FROM python:3.11-slim

ARG RCCIA_GIT_REF=multicancer-v1
ARG RCCIA_GIT_COMMIT=a51772da5d50eb8d758c01d737ea09f2f388cea6

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_HEADLESS=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    RCCIA_REPO_ROOT=/home/user/app/source

WORKDIR /home/user/app

RUN git clone --depth 1 --branch "${RCCIA_GIT_REF}" \
        https://github.com/amiralfefe/R.C.C.I.A.git source \
    && test "$(git -C source rev-parse HEAD)" = "${RCCIA_GIT_COMMIT}"

COPY --chown=user requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY --chown=user download_models.py entrypoint.py ./

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=3).read()"]

CMD ["python", "entrypoint.py"]
