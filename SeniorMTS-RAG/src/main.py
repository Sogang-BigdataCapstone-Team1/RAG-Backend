from retrievers import setup_retrievers
from tools import setup_tools
from llm import setup_llm
from agent import setup_agent
from parsers import setup_parser
from utils.config import load_environment
from utils.session import get_session_history
from langchain.schema import HumanMessage, AIMessage

from datetime import datetime
from retrievers import setup_retrievers
from tools import setup_tools
from llm import setup_llm
from agent import setup_agent
from parsers import setup_parser
from utils.config import load_environment
from utils.session import get_session_history
from langchain.schema import HumanMessage, AIMessage


from retrievers import setup_retrievers
from tools import setup_tools
from llm import setup_llm
from agent import setup_agent
from parsers import setup_parser
from utils.config import load_environment
from utils.session import get_session_history
from langchain.schema import HumanMessage, AIMessage
from datetime import datetime


def main():
    # 환경 변수 로드
    load_environment()

    # 검색기 및 도구 설정
    cycle_retriever, stock_retriever, news_retriever = setup_retrievers()
    tools = setup_tools(cycle_retriever, stock_retriever, news_retriever)

    # LLM 및 에이전트 설정
    llm, prompt = setup_llm()
    agent_executor = setup_agent(llm, tools, prompt)

    # 세션 ID
    session_id = input("세션 ID를 입력하세요 (기본값: 'default_session'): ").strip() or "default_session"

    # 대화 기록 관리
    session_history = get_session_history(session_id)

    while True:
        user_input = input("질문을 입력하세요 (종료하려면 'exit' 입력): ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("프로그램을 종료합니다.")
            break

        try:
            # 현재 시간을 계산
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 대화 기록 전달
            chat_history = [{"type": message.type, "content": message.content} for message in session_history.messages]

            # 에이전트 실행
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
                "current_time": current_time  # 현재 시간을 전달
            })

            # 딕셔너리 형태의 응답 처리
            if isinstance(response, dict) and 'output' in response:
                agent_output = response['output']
            else:
                agent_output = str(response)

            print("\n[에이전트 응답]:", agent_output)

            # 세션 기록 저장
            session_history.add_message(HumanMessage(content=user_input))
            session_history.add_message(AIMessage(content=agent_output))

        except Exception as e:
            print(f"오류 발생: {e}")

    # 세션 기록 출력
    print("\n[세션 기록]:")
    for message in session_history.messages:
        if isinstance(message, HumanMessage):
            print(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"AI: {message.content}")


if __name__ == "__main__":
    main()
