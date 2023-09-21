# Stage 1: Compile image
FROM python:3.10-slim-bullseye AS compile-image
WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y wget libpq-dev gcc g++ python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Stage 2: Build image
FROM python:3.10-slim-bullseye AS build-image

ARG BUILD_DATE
ARG BUILD_VERSION
ARG BUILD_NUMBER
ARG GIT_COMMIT
ARG BUILD_AUTHORS
ARG JENKINS_URL
ARG COMPOSE
ARG K8SYML

LABEL org.opencontainers.image.authors=$BUILD_AUTHORS \
      org.opencontainers.image.source="https://github.com/timothymugayi/sentimentGPT" \
      org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.build_number=$BUILD_NUMBER \
      org.opencontainers.image.commit=$GIT_COMMIT \
      org.opencontainers.image.build.server=$JENKINS_URL \
      org.opencontainers.image.title="timothymugayi/sentimentGPT" \
      org.opencontainers.image.description="The repository contains sentimentGPT FastAPI application" \
      org.opencontainers.image.version=$BUILD_VERSION \
      org.opencontainers.image.build="docker build -t timothymugayi/sentimentgpt --build-arg BUILD_DATE='$(date +%D)' --build-arg BUILD_VERSION='0.0.1' --build-arg COMPOSE='$(cat docker-compose.yml|base64)' --build-arg K8SYML='$(cat fast_k8s.yml|base64)' --build-arg BUILD_AUTHORS=$BUILD_AUTHORS --build-arg JENKINS_URL=$JENKINS_URL --build-arg BUILD_NUMBER=$BUILD_NUMBER --build-arg GIT_COMMIT=$(git log -n 1 --pretty=format:'%h') . " \
      org.zdocker.compose=$COMPOSE \
      org.zdocker.k8s=$K8SYML

WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=compile-image /opt/venv /opt/venv
COPY --from=compile-image /app /app

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000