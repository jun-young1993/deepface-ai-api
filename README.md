## 실행 방법

### 방법 1: Docker 사용 (권장)

**빌드 및 실행:**

Windows:
```bash
docker-run.bat
```

Linux/Mac/Git Bash:
```bash
chmod +x docker-run.sh
./docker-run.sh
```

**수동 실행:**

1. Docker 이미지 빌드:
```bash
docker build -t deep-face-ai .
```

2. Docker 컨테이너 실행:

**방법 A: Foreground 모드 (모든 출력을 실시간으로 확인)**
```bash
docker run --name deep-face-ai -p 8080:8080 gcr.io/ai-hub-483500/deep-face-ai-api:v1.1
```
- 모든 로그가 실시간으로 출력됩니다
- 컨테이너를 중지하려면 `Ctrl+C`를 누르세요

**방법 B: Background 모드 (백그라운드에서 실행)**
```bash
docker run -d --name deep-face-ai -p 8080:8080 deep-face-ai-api:latest
```
- 백그라운드에서 실행됩니다
- 로그를 보려면 아래 명령어를 사용하세요

3. 서버 확인:
- API 문서: http://localhost:8080/docs
- 헬스체크: http://localhost:8080/health


도커 쉘 오픈
```bash
docker exec -it deep-face-ai sh
```

**컨테이너 관리:**
```bash
# 로그 확인 (실시간 출력)
docker logs -f deep-face-ai

# 로그 확인 (마지막 100줄)
docker logs --tail 100 deep-face-ai

# 컨테이너 중지
docker stop deep-face-ai

# 컨테이너 재시작
docker start deep-face-ai

# 컨테이너 삭제
docker rm deep-face-ai

# 이미지 삭제
docker rmi deep-face-ai-api:latest
```

### 방법 2: 로컬 실행 (가상환경 사용)

**실행 스크립트 사용:**

**Windows (CMD/PowerShell):**
```bash
start_server.bat
```

**Git Bash/Linux/Mac:**
```bash
chmod +x start_server.sh
./start_server.sh
```

### 방법 2: 수동 실행

**1단계: 가상환경 활성화**

Windows CMD/PowerShell:
```bash
deepface\Scripts\activate
```

Git Bash:
```bash
source deepface/Scripts/activate
```

**2단계: API 서버 실행**

가상환경 활성화 후:
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

또는 가상환경 활성화 없이:
```bash
deepface/Scripts/python.exe -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```



**3단계: 서버 확인**

브라우저에서 다음 주소로 접속:
- API 문서: http://127.0.0.1:8000/docs
- 헬스체크: http://127.0.0.1:8000/health

## API 엔드포인트

### 1. 헬스체크
```
GET /health
```

### 1-1. 임시 파일 저장 (raw 바이너리 업로드)
```
POST /files/temp/raw?dir_path=C:/tmp&filename=a.jpg&ttl_seconds=600
Content-Type: application/octet-stream
Body: (바이너리)
```

### 2. 얼굴 검증 (이미지 경로 사용)
```
POST /verify
Content-Type: application/json

{
  "img1_path": "C:/path/to/image1.jpg",
  "img2_path": "C:/path/to/image2.jpg"
}
```

### 3. 얼굴 분석 (이미지 경로 사용)
```
POST /analyze
Content-Type: application/json

{
  "img_path": "C:/path/to/image.jpg",
  "actions": ["age", "gender", "race", "emotion"]
}
```

### 4. 얼굴 검증 (파일 업로드)
```
POST /verify/upload
Content-Type: multipart/form-data

img1: (파일)
img2: (파일)
```

### 5. 얼굴 분석 (파일 업로드)
```
POST /analyze/upload
Content-Type: multipart/form-data

img: (파일)
actions: ["age", "gender", "race", "emotion"] (선택사항)
```

## 테스트 예시

### Python으로 테스트
```python
import requests

# Analyze 테스트
response = requests.post(
    "http://127.0.0.1:8000/analyze",
    json={
        "img_path": "C:/Users/junyoung/junyoung/source/deep-face-ai/tests/1.jpg",
        "actions": ["age", "gender", "race", "emotion"]
    },
    timeout=300
)
print(response.json())

# Verify 테스트
response = requests.post(
    "http://127.0.0.1:8000/verify",
    json={
        "img1_path": "C:/Users/junyoung/junyoung/source/deep-face-ai/tests/1.jpg",
        "img2_path": "C:/Users/junyoung/junyoung/source/deep-face-ai/tests/2.jpg"
    },
    timeout=300
)
print(response.json())
```

### cURL로 테스트
```bash
# Health check
curl http://127.0.0.1:8000/health

# Raw 바이너리 업로드(임시 저장)
curl -X POST "http://127.0.0.1:8000/files/temp/raw?dir_path=C:/Users/junyoung/OneDrive/Desktop/막쓰는%20폴더/새%20폴더/tmp&filename=1.jpg" \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@C:/Users/junyoung/junyoung/source/deep-face-ai/tests/1.jpg"

# Analyze
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"img_path": "C:/Users/junyoung/junyoung/source/deep-face-ai/tests/1.jpg", "actions": ["age", "gender"]}'
```