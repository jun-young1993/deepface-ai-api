# 자동 requirements.txt 생성
```shell
pip freeze > requirements.txt
```


# gcloud 로 docker 빌드하여 올리기
profject id = ai-hub-483500

## 로그인

```shell
gcloud auth login gcloud config set project ai-hub-483500 gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

## 빋드
```shell
docker build -t gcr.io/ai-hub-483500/deep-face-ai-api:v1.2 .
```

# ✅ 2) GCP Artifact Registry(또는 GCR)에 푸시

먼저 구글 로그인 되어 있어야 함:

```shell
gcloud auth configure-docker
```

그다음 push:
```shell
docker push gcr.io/ai-hub-483500/deep-face-ai-api:v1.1
```
