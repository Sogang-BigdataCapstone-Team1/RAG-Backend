from langchain_community.chat_message_histories import ChatMessageHistory

# 세션 기록 저장 딕셔너리
store = {}

def get_session_history(session_id):
    """
    세션 ID로 기록 관리
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
