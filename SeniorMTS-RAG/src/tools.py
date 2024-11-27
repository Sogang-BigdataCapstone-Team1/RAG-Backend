from langchain.tools.retriever import create_retriever_tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool

from urllib import parse
from ast import literal_eval
import requests
from datetime import datetime, timedelta
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool


# 1. 기본 날짜 범위를 계산하는 함수
def get_default_date_range():
    """
    Return default start and end dates (last 7 days).
    """
    today = datetime.now()
    start_date = today - timedelta(days=7)
    return start_date.strftime("%Y%m%d"), today.strftime("%Y%m%d")


# 2. 실시간 주가 및 거래량 데이터를 가져오는 함수
def get_sise(code, start_time, end_time, time_from='day'):
    """
    실시간 주가 및 거래량 데이터를 가져오는 함수
    :param code: (str) 종목 코드 (e.g., '005930' for 삼성전자)
    :param start_time: (str) 시작 날짜 (YYYYMMDD 형식)
    :param end_time: (str) 종료 날짜 (YYYYMMDD 형식)
    :param time_from: (str) 날짜 간격 ('day', 'week', 'month' 중 하나)
    :return: (list) 주가 및 거래량 데이터
    """
    get_param = {
        'symbol': code,
        'requestType': 1,
        'startTime': start_time,
        'endTime': end_time,
        'timeframe': time_from
    }
    get_param = parse.urlencode(get_param)
    url = f"https://api.finance.naver.com/siseJson.naver?{get_param}"
    response = requests.get(url)
    return literal_eval(response.text.strip())


# 3. 실시간 주가 수집 도구
def real_time_stock_tool(code, start_time=None, end_time=None, time_from='day'):
    """
    Retrieve real-time stock price and volume data.
    If no dates are provided, defaults to the last 7 days.
    """
    if not start_time or not end_time:
        start_time, end_time = get_default_date_range()

    data = get_sise(code, start_time, end_time, time_from)
    if not data:
        return "No data available for the given stock code and time range."

    # 표 데이터를 문자열로 변환
    header = data[0]
    rows = data[1:]
    formatted_data = [", ".join(map(str, header))]
    for row in rows:
        formatted_data.append(", ".join(map(str, row)))

    return "\n".join(formatted_data)


def setup_tools(cycle_retriever, stock_retriever, news_retriever):
    """
    검색 도구들을 설정합니다.
    """
    tools = []

    # 1. 실시간 주가 수집 도구
    real_time_stock_tool_instance = Tool.from_function(
        func=real_time_stock_tool,
        name="real_time_stock_data",  # 공백 없는 유효한 이름
        description=(
            "Retrieve accurate and up-to-date stock price and volume data for a given stock. "
            "If no date range is provided, defaults to the last 7 days. "
            "Parameters:\n"
            "- `code`: Stock code (e.g., Samsung Electronics: '005930').\n"
            "- `start_time`: Start date in YYYYMMDD format.\n"
            "- `end_time`: End date in YYYYMMDD format.\n"
            "- `time_from`: Timeframe ('day', 'week', 'month').\n"
            "This tool is the most accurate and reliable source for stock data.\n\n"
            "Examples of stock codes for KOSPI top 30 companies:\n"
            "2. SK Hynix (000660)\n"
            "3. LG Energy Solution (373220)\n"
            "11. NAVER (035420)\n" 
            "6. Hyundai Motor (005380)\n"

        )
    )
    tools.append(real_time_stock_tool_instance)

    # 2. Tavily 인터넷 검색 도구
    internet_search_tool = Tool.from_function(
        func=lambda input: TavilySearchResults(k=6).run(input),  # 람다 함수로 명시적 호출
        name="real_time_internet_search",  # 공백 없는 유효한 이름
        description=(
            "Use this tool to search for information from the web when relevant data cannot be found in the documents. "
            "This tool provides real-time information but may not always be accurate or verified. "
            "When using this tool, always include a disclaimer in the response, such as:\n"
            "'This information was retrieved from the internet and may not be fully accurate or verified. "
            "Please use caution when interpreting these results.'"
        )
    )
    tools.append(internet_search_tool)

    # 3. Cycle search tool
    cycle_search_tool = create_retriever_tool(
        cycle_retriever,
        name="cycle_search",  # 공백 없는 유효한 이름
        description="Use this tool to search information about the economic cycle.",
    )
    tools.append(cycle_search_tool)

    # 4. Stock search tool
    stock_search_tool = create_retriever_tool(
        stock_retriever,
        name="stock_information_search",  # 공백 없는 유효한 이름
        description="Use this tool to search information about the stock in question.",
    )
    tools.append(stock_search_tool)

    # 5. News search tool
    news_search_tool = create_retriever_tool(
        news_retriever,
        name="news_information_search",  # 공백 없는 유효한 이름
        description="Use this tool to search information about the news related to the stock in question.",
    )
    tools.append(news_search_tool)

    return tools
