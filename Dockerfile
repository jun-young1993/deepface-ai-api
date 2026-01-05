# Python 이미지 사용 (3.11 권장 - 안정성 + 패키지 호환성)
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# OpenCV 런타임 의존성만 설치 (빌드 도구 불필요!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사
COPY requirements.txt ./

# 의존성 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY main.py ./
COPY api/ ./api/

# 포트 노출 (Cloud Run용)
EXPOSE 8080

# Cloud Run의 PORT 환경 변수를 사용하도록 설정
ENV PORT=8080

# 컨테이너 시작 시 실행할 명령어 (Cloud Run 호환)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}