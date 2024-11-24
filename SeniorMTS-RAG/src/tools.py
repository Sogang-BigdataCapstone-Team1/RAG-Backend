from langchain.tools.retriever import create_retriever_tool

def setup_tools(cycle_retriever, stock_retriever, news_retriever):
    """
    검색 도구 설정
    """
    cycle_search_tool = create_retriever_tool(
        cycle_retriever,
        name="cycle_search",
        description="use this tool to search information about the economic cycle",
    )

    stockinformation_search_tool = create_retriever_tool(
        stock_retriever,
        name="stock_information_search",
        description="use this tool to search information about the stock in question",
    )

    newsinformation_search_tool = create_retriever_tool(
        news_retriever,
        name="news_information_search",
        description="use this tool to search information about the news related to the stock I asked about",
    )

    return [cycle_search_tool, stockinformation_search_tool, newsinformation_search_tool]
