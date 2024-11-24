# Python 3.10 slim 이미지를 기반으로 생성
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# Python 경로 설정 (src 디렉터리를 모듈로 인식)
ENV PYTHONPATH=/app/SeniorMTS-RAG/src

# FastAPI 앱 실행
CMD ["uvicorn", "fastapi-app:app", "--host", "0.0.0.0", "--port", "8001"]
