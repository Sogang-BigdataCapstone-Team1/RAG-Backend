import re
from dotenv import load_dotenv
from langchain_teddynote.korean import stopwords
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_teddynote.community.pinecone import (
    preprocess_documents,
    create_sparse_encoder,
    fit_sparse_encoder,
    upsert_documents_parallel,
    delete_namespace,
)
from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_teddynote import logging


def clean_and_filter_text(content):
    """
    텍스트를 정리하면서 불필요한 줄바꿈 및 공백을 제거합니다.
    1. 줄바꿈이 3번 이상 반복되면 2번으로 축소.
    2. 줄바꿈 이후 줄바꿈이 아닌 문자가 나올 때까지 제거.
    3. 줄바꿈 앞뒤 공백 제거.
    4. 여러 공백을 하나로 축소.
    5. 텍스트 양 끝 공백 제거.
    """
    # 3번 이상 반복되는 줄바꿈을 2번으로 축소
    pattern = r"(?:\n){3,}"

    def replacer(match):
        """
        매칭된 패턴 이후 줄바꿈 제거
        """
        text = match.group(0)  # 매칭된 문자열 가져오기
        # 반복된 줄바꿈 이후 남은 텍스트 가져오기
        remainder = text.split("\n", 3)[-1]
        # 줄바꿈이 아닌 문자가 나올 때까지 삭제
        cleaned_remainder = re.sub(r"^\n+", "", remainder)
        return "\n\n\n" + cleaned_remainder

    # 줄바꿈 패턴 처리
    content = re.sub(pattern, replacer, content)

    # 각 줄 앞뒤 공백 제거
    content = re.sub(r" *\n+ *", "\n", content)

    # 여러 공백을 하나의 공백으로 축소
    content = re.sub(r"\s{2,}", " ", content)


    # 텍스트 양 끝 공백 제거
    content = content.strip()

    return content




def main():
    # API 키 정보 로드
    load_dotenv()

    # LangSmith 추적을 설정합니다. https://smith.langchain.com
    logging.langsmith("SeniorMTS-RAG")

    # Pinecone 초기화 및 인스턴스 생성
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)

    # Pinecone 인덱스 연결
    index_name = "seniormts"  # 기존에 생성된 Pinecone 인덱스 이름
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=4096,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="gcp",
                region="us-east1-gcp"
            )
        )
        print(f"Created Pinecone index: {index_name}")
    pc_index = pc.Index(index_name)

    # 데이터 경로 설정
    data_path = "../data/cyclereports/*.pdf"

    # 폴더명 추출 -> namespace로 사용
    namespace = os.path.basename(os.path.dirname(data_path))
    print(f"Namespace: {namespace}")

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

    split_docs = []

    # 텍스트 파일을 load -> List[Document] 형태로 변환
    files = sorted(glob.glob(data_path))

    for file in files:
        loader = PyMuPDFLoader(file)
        split_docs.extend(loader.load_and_split(text_splitter))

    # 문서 개수 확인
    print("청크개수: ", len(split_docs))

    # 필터링 조건 적용
    filtered_docs = [
        doc for doc in split_docs
        if len(clean_and_filter_text(doc.page_content.replace("\n", ""))) >= 200

    ]

    # 필터링된 문서를 preprocess_documents로 전달
    contents, metadatas = preprocess_documents(
        split_docs=filtered_docs,
        metadata_keys=["source", "page", "author"],
        min_length=5,  # 이 값은 필터링된 데이터에도 적용 가능
        use_basename=True,
    )

    # 최종 데이터 확인
    print("필터링 후 청크 개수: ", len(contents))

    # 문서 개수 확인, 소스 개수 확인, 페이지 개수 확인
    print(f"문서 개수 (contents): {len(contents)}")
    print(f"소스 개수 (metadatas['source']): {len(metadatas['source'])}")
    print(f"페이지 개수 (metadatas['page']): {len(metadatas['page'])}")

    # 한글 불용어 사전 + Kiwi 형태소 분석기를 사용합니다.
    sparse_encoder = create_sparse_encoder(stopwords(), mode="kiwi")

    # Sparse Encoder 를 사용하여 contents 를 학습
    saved_path = fit_sparse_encoder(
        sparse_encoder=sparse_encoder, contents=contents, save_path=f"./{namespace}_sparse_encoder.pkl"
    )

    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    upstage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")

    # 기존 namespace 삭제
    delete_namespace(
        pinecone_index=pc_index,
        namespace=namespace,
    )

    # Pinecone 인덱스 생성 및 업로드
    upsert_documents_parallel(
        index=pc_index,  # Pinecone 인덱스
        namespace=namespace,  # 폴더명으로 설정된 namespace
        contents=contents,  # 이전에 전처리한 문서 내용
        metadatas=metadatas,  # 이전에 전처리한 문서 메타데이터
        sparse_encoder=sparse_encoder,  # Sparse encoder
        embedder=upstage_embeddings,
        batch_size=64,
        max_workers=30,
    )


if __name__ == "__main__":
    main()
