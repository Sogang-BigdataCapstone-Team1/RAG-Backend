import pandas as pd
from dotenv import load_dotenv
from langchain_teddynote.korean import stopwords
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_teddynote.community.pinecone import (
    preprocess_documents,
    create_sparse_encoder,
    fit_sparse_encoder,
)
from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_teddynote.community.pinecone import upsert_documents_parallel
from langchain_teddynote.community.pinecone import delete_namespace


def main():
    # API 키 정보 로드
    load_dotenv()

    # LangSmith 추적을 설정합니다.
    from langchain_teddynote import logging
    logging.langsmith("SeniorMTS-RAG")

    # 파일 경로 및 네임스페이스 설정
    csv_path = "../data/stock_news.csv"
    namespace = "stocknews"

    # Pinecone 초기화 및 인스턴스 생성
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)

    # Pinecone 인덱스 연결
    index_name = "seniormts"  # 기존에 생성된 Pinecone 인덱스 이름
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # 임베딩 모델에 맞는 차원 설정
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-1"
            )
        )
        print(f"Created Pinecone index: {index_name}")
    pc_index = pc.Index(index_name)

    # CSV 파일 읽기
    df = pd.read_csv(csv_path)

    # 본문 텍스트 추출
    texts = df['content'].tolist()

    # 메타데이터 생성
    metadatas = [
        {
            "title": row["title"],
            "time": row["time"],
            "url": row["url"]
        }
        for _, row in df.iterrows()
    ]

    # 텍스트 분할 설정
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=700)

    # 텍스트를 청크로 분할
    split_docs = []
    split_metadatas = []  # 청크와 메타데이터의 수를 동일하게 유지

    for text, metadata in zip(texts, metadatas):
        docs = text_splitter.create_documents([text], metadatas=[metadata])
        split_docs.extend(docs)
        split_metadatas.extend([metadata] * len(docs))  # 각 청크에 대해 동일한 메타데이터 복제

    # 문서 개수 확인
    print("청크 개수: ", len(split_docs))
    print("메타데이터 개수 (split_metadatas): ", len(split_metadatas))

    # Preprocess 문서를 전달
    contents, processed_metadatas = preprocess_documents(
        split_docs=split_docs,
        metadata_keys=["title", "time", "url"],  # 메타데이터 키 설정
        min_length=5,  # 필터링 최소 길이
        use_basename=False,  # 파일 기반이 아니므로 False
    )

    # 최종 데이터 확인
    print("처리된 청크 개수: ", len(contents))
    print(f"문서 개수 (contents): {len(contents)}")
    print(f"메타데이터 개수 (processed_metadatas): {len(processed_metadatas)}")

    # 한글 불용어 사전 + Kiwi 형태소 분석기를 사용합니다.
    sparse_encoder = create_sparse_encoder(stopwords(), mode="kiwi")

    # Sparse Encoder 를 사용하여 contents 를 학습
    saved_path = fit_sparse_encoder(
        sparse_encoder=sparse_encoder, contents=contents, save_path=f"./{namespace}_sparse_encoder.pkl"
    )

    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    upstage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")

    # 기존 네임스페이스 삭제
    delete_namespace(
        pinecone_index=pc_index,
        namespace=namespace,
    )

    # Pinecone 인덱스 생성 및 업로드
    upsert_documents_parallel(
        index=pc_index,  # Pinecone 인덱스
        namespace=namespace,  # 네임스페이스
        contents=contents,  # 전처리한 문서 내용
        metadatas=processed_metadatas,  # 전처리된 문서 메타데이터
        sparse_encoder=sparse_encoder,  # Sparse encoder
        embedder=upstage_embeddings,
        batch_size=64,
        max_workers=30,
    )


if __name__ == "__main__":
    main()
