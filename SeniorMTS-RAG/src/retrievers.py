from dotenv import load_dotenv
from langchain_teddynote.community.pinecone import init_pinecone_index, PineconeKiwiHybridRetriever
from langchain_upstage import UpstageEmbeddings
import os

# .env 파일 로드
load_dotenv()

from pathlib import Path

# 현재 파일의 디렉토리 경로
BASE_DIR = Path(__file__).resolve().parent

def stopwords():
    """
    로컬 파일로부터 'korean_stopwords.txt'를 읽어 불용어 리스트를 반환합니다.
    """
    # 현재 파일의 디렉터리를 기준으로 경로 설정
    local_path = Path(__file__).parent / "korean_stopwords.txt"

    # 파일이 존재하는지 확인
    if not local_path.exists():
        raise FileNotFoundError(f"'{local_path}' 경로에 'korean_stopwords.txt' 파일이 없습니다.")

    # 파일 읽기
    with local_path.open("r", encoding="utf-8") as file:
        return [word.strip() for word in file.readlines()]

def setup_retrievers():
    """
    검색기 생성 및 설정
    """
    stopwords_list = stopwords()  # 로컬 파일에서 불용어 리스트 로드

    # Cycle 검색기 설정
    cycle_params = init_pinecone_index(
        index_name="seniormts",
        namespace="cyclereports",
        api_key=os.environ["PINECONE_API_KEY"],
        sparse_encoder_path=str(BASE_DIR / "cyclereports_sparse_encoder.pkl"),
        stopwords=stopwords_list,
        tokenizer="kiwi",
        embeddings=UpstageEmbeddings(model="solar-embedding-1-large-query"),
        top_k=5,
        alpha=0.5,
    )
    cycle_retriever = PineconeKiwiHybridRetriever(**cycle_params)

    # Stock 검색기 설정
    stock_params = init_pinecone_index(
        index_name="seniormts",
        namespace="stockreports",
        api_key=os.environ["PINECONE_API_KEY"],
        sparse_encoder_path=str(BASE_DIR / "stockreports_sparse_encoder.pkl"),
        stopwords=stopwords_list,
        tokenizer="kiwi",
        embeddings=UpstageEmbeddings(model="solar-embedding-1-large-query"),
        top_k=4,
        alpha=0.5,
    )
    stock_retriever = PineconeKiwiHybridRetriever(**stock_params)

    # News 검색기 설정
    news_params = init_pinecone_index(
        index_name="seniormts",
        namespace="stocknews",
        api_key=os.environ["PINECONE_API_KEY"],
        sparse_encoder_path=str(BASE_DIR / "stocknews_sparse_encoder.pkl"),
        stopwords=stopwords_list,
        tokenizer="kiwi",
        embeddings=UpstageEmbeddings(model="solar-embedding-1-large-query"),
        top_k=3,
        alpha=0.5,
    )
    news_retriever = PineconeKiwiHybridRetriever(**news_params)

    return cycle_retriever, stock_retriever, news_retriever
