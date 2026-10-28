FROM python:3.12-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN python -m pip wheel --no-cache-dir --no-deps . -w /wheels
ARG WITH_LLM=0
RUN if [ "$WITH_LLM" = "1" ]; then CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_METAL=OFF" python -m pip wheel --no-cache-dir --no-deps "llama-cpp-python==0.3.14" -w /wheels; fi

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.12.0"
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels && useradd --uid 10001 --create-home tiemu && mkdir /data && chown tiemu:tiemu /data
USER tiemu
WORKDIR /data
VOLUME ["/data"]
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-u", "-m", "main"]
CMD ["--no-interactive", "--device", "cpu", "--llm-device", "cpu", "--disable-llm", "--backend", "cpp", "--output-dir", "/data"]
