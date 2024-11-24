from langchain_teddynote.community.pinecone import init_pinecone_index, PineconeKiwiHybridRetriever
from langchain_upstage import UpstageEmbeddings
from langchain_teddynote.korean import stopwords
import os

def setup_retrievers():
    """
    검색기 생성 및 설정
    """
    # Cycle 검색기 설정
    cycle_params = init_pinecone_index(
        index_name="seniormts",
        namespace="cyclereports",
        api_key=os.environ["PINECONE_API_KEY"],
        sparse_encoder_path="./cyclereports_sparse_encoder.pkl",
        stopwords=stopwords(),
        tokenizer="kiwi",
        embeddings=UpstageEmbeddings(model="solar-embedding-1-large-query"),
        top_k=3,
        alpha=0.5,
    )
    cycle_retriever = PineconeKiwiHybridRetriever(**cycle_params)

    # Stock 검색기 설정
    stock_params = init_pinecone_index(
        index_name="seniormts",
        namespace="stockreports",
        api_key=os.environ["PINECONE_API_KEY"],
        sparse_encoder_path="./stockreports_sparse_encoder.pkl",
        stopwords=stopwords(),
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
        sparse_encoder_path="./stocknews_sparse_encoder.pkl",
        stopwords=stopwords(),
        tokenizer="kiwi",
        embeddings=UpstageEmbeddings(model="solar-embedding-1-large-query"),
        top_k=3,
        alpha=0.5,
    )
    news_retriever = PineconeKiwiHybridRetriever(**news_params)

    return cycle_retriever, stock_retriever, news_retriever
