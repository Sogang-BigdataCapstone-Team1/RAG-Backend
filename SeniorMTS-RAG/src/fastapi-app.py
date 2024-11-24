from fastapi import FastAPI, HTTPException
from retrievers import setup_retrievers
from tools import setup_tools
from llm import setup_llm
from agent import setup_agent
from utils.config import load_environment
from utils.session import get_session_history
from langchain.schema import HumanMessage, AIMessage
from pydantic import BaseModel

# FastAPI 인스턴스 생성
app = FastAPI()

# 환경 변수 로드
load_environment()

# 검색기 및 도구 설정
cycle_retriever, stock_retriever, news_retriever = setup_retrievers()
tools = setup_tools(cycle_retriever, stock_retriever, news_retriever)

# LLM 및 에이전트 설정
llm, prompt = setup_llm()
agent_executor = setup_agent(llm, tools, prompt)

# 요청 데이터 모델 정의
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

# 응답 데이터 모델 정의
class ChatResponse(BaseModel):
    output: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    사용자 입력과 세션 ID를 받아 AI 응답을 반환하는 API 엔드포인트
    """
    try:
        # 세션 관리
        session_history = get_session_history(request.session_id)

        # 기존 대화 기록 가져오기
        chat_history = [{"type": message.type, "content": message.content} for message in session_history.messages]

        # 에이전트 실행
        response = agent_executor.invoke({"input": request.user_input, "chat_history": chat_history})

        # 응답 처리
        if isinstance(response, dict) and 'output' in response:
            agent_output = response['output']
        else:
            agent_output = str(response)

        # 세션 기록 갱신
        session_history.add_message(HumanMessage(content=request.user_input))
        session_history.add_message(AIMessage(content=agent_output))

        # 응답 반환
        return ChatResponse(output=agent_output)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {e}")

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    특정 세션의 대화 기록을 반환하는 API 엔드포인트
    """
    try:
        session_history = get_session_history(session_id)
        return {
            "session_id": session_id,
            "messages": [
                {"type": "Human" if isinstance(message, HumanMessage) else "AI", "content": message.content}
                for message in session_history.messages
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {e}")
