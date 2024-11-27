from langchain.agents import create_tool_calling_agent, AgentExecutor

from datetime import datetime

from datetime import datetime


def execute_agent(agent_executor, user_input, chat_history):
    """
    에이전트를 실행하고 응답을 반환합니다.
    :param agent_executor: 에이전트 실행기
    :param user_input: 사용자 입력
    :param chat_history: 대화 기록
    :return: 에이전트의 응답
    """
    # 현재 시간 계산
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 에이전트 실행
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history,
        "current_time": current_time  # 현재 시간 전달
    })
    return response


def setup_agent(llm, tools, prompt):
    """
    에이전트와 실행기 생성
    """
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return agent_executor
