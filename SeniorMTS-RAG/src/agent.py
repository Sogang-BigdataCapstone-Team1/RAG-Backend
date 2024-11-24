from langchain.agents import create_tool_calling_agent, AgentExecutor

def setup_agent(llm, tools, prompt):
    """
    에이전트와 실행기 생성
    """
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return agent_executor
