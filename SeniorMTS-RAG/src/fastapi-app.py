from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from langchain.schema import HumanMessage, AIMessage
from utils.config import load_environment
from utils.session import get_session_history
from retrievers import setup_retrievers
from tools import setup_tools, real_time_stock_tool
from llm import setup_llm
from agent import setup_agent

# FastAPI 인스턴스 생성
app = FastAPI(
    title="AI Agent and Tools API",
    description="Provides AI agent responses and tools for stock analysis and information retrieval."
)

# 환경 변수 로드
load_environment()

# 검색기, 도구, LLM 및 에이전트 설정
try:
    cycle_retriever, stock_retriever, news_retriever = setup_retrievers()
    tools = setup_tools(cycle_retriever, stock_retriever, news_retriever)
    llm, prompt = setup_llm()
    agent_executor = setup_agent(llm, tools, prompt)
except Exception as e:
    raise RuntimeError(f"Initialization failed: {e}")

# 요청 및 응답 모델 정의
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

class ChatResponse(BaseModel):
    output: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    AI Agent API 엔드포인트: 사용자 입력과 세션 ID를 받아 AI 응답을 반환합니다.
    Tools의 기능도 함께 사용 가능합니다.
    """
    try:
        # 세션 기록 가져오기
        session_history = get_session_history(request.session_id)
        chat_history = session_history.messages

        # 현재 시간 계산
        current_time = datetime.utcnow().isoformat()

        # 에이전트 실행
        response = agent_executor.invoke({
            "input": request.user_input,
            "chat_history": chat_history,
            "current_time": current_time,
        })

        # 에이전트 응답 처리
        agent_output = response.get("output", str(response))

        # 사용자 요청이 특정 도구와 관련된 경우 직접 호출
        if "주가" in request.user_input:
            try:
                # 주식 코드 추출 및 real_time_stock_tool 호출
                stock_code = extract_stock_code(request.user_input)
                stock_data = real_time_stock_tool(code=stock_code)
                agent_output += f"\n추가 주가 정보:\n{stock_data}"
            except Exception as e:
                agent_output += f"\n추가 주가 정보를 가져오는 중 오류가 발생했습니다: {e}"

        elif "인터넷 검색" in request.user_input:
            try:
                # TavilySearchResults 도구 호출
                search_results = tools[1].run(request.user_input)  # internet_search_tool
                agent_output += f"\n인터넷 검색 결과:\n{search_results}"
            except Exception as e:
                agent_output += f"\n인터넷 검색 도중 오류가 발생했습니다: {e}"

        # 세션 기록 갱신
        session_history.add_message(HumanMessage(content=request.user_input))
        session_history.add_message(AIMessage(content=agent_output))

        # 응답 반환
        return ChatResponse(output=agent_output)

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


def extract_stock_code(user_input: str) -> str:
    """
    사용자 입력에서 종목 코드를 추출합니다.
    """
    # 간단한 예시: 사용자 입력에서 특정 키워드를 기반으로 종목 코드 매핑
    stock_mapping = {
        "삼성전자": "005930",
        "SK하이닉스": "000660",
        "LG에너지솔루션": "373220",
        "현대차": "005380",
    }
    for stock_name, stock_code in stock_mapping.items():
        if stock_name in user_input:
            return stock_code
    raise ValueError("사용자 입력에서 종목 코드를 찾을 수 없습니다.")


@app.get("/")
async def root():
    """
    루트 엔드포인트: 서버 상태를 확인합니다.
    """
    return {"message": "AI Agent and Tools API is running successfully!"}
