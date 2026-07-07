# Build from the repo root: docker build -f demo/Dockerfile --build-arg CUDA_ARCH=75 -t redraft-space .
# CUDA_ARCH must match the target GPU: 75 (T4), 86 (A10G), 89 (L4/Ada).

ARG CUDA_ARCH=75
ARG LLAMA_COMMIT=9777256c3130fa3201327bfab44bae187f7caea2

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
ARG CUDA_ARCH
ARG LLAMA_COMMIT

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake git build-essential libcurl4-openssl-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git init -q . \
    && git remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git fetch -q --depth 1 origin "${LLAMA_COMMIT}" \
    && git checkout -q FETCH_HEAD

COPY engine/llama-server-prompt-probs.patch \
     engine/llama-server-redraft.patch \
     /tmp/
# redraft depends on the prompt-probs patch (same three server files); order matters.
RUN git apply /tmp/llama-server-prompt-probs.patch \
    && git apply /tmp/llama-server-redraft.patch

COPY engine/stabilize/stabilize.hpp \
     engine/stabilize/batched_stabilize.hpp \
     engine/stabilize/redraft_stabilize.hpp \
     tools/server/

# HF Spaces' Docker builder has no GPU attached, so nothing here may assume one:
# CMAKE_CUDA_ARCHITECTURES is pinned explicitly (nvcc cross-compiles for a given
# compute capability without a live device) and GGML_NATIVE=OFF stops ggml from
# baking in -march=native for whatever CPU the *builder* happens to have, which
# would silently crash with "illegal instruction" on a *runtime* host with a
# different CPU. LLAMA_CURL=ON so the server can resolve `-hf <repo>:<quant>`
# itself at startup (llama.cpp's own downloader), caching into $LLAMA_CACHE.
# ggml-cuda's CMakeLists only links libcuda.so (the driver API) for its VMM
# (virtual-memory-management) allocator path; that lib normally comes from the
# host driver bind-mount nvidia-docker does at *runtime*, so it's absent here at
# build time. GGML_CUDA_NO_VMM=ON is llama.cpp's own documented way to skip that
# link entirely rather than fighting to satisfy it on a GPU-less builder.
RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA_NO_VMM=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
    && cmake --build build -t llama-server -j"$(nproc)"

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libcurl4 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

# pyproject.toml requires Python >=3.11, newer than Ubuntu 22.04's system python3,
# so uv downloads a managed interpreter. By default that lands under whichever
# user's home ran `uv sync` (root's, at build time) -- unreadable to the `user`
# account this image actually runs as. Point it at a shared, world-readable dir.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN useradd -u 1000 -m user

WORKDIR /app
COPY --from=builder /build/build/bin/llama-server /app/bin/llama-server
COPY --from=builder /build/build/bin/*.so* /app/bin/

COPY clients/python/redraft_client /clients/python/redraft_client
COPY demo/pyproject.toml demo/uv.lock /app/
RUN uv sync --frozen --no-dev
COPY demo/app /app/app
COPY demo/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /data && chown -R user:user /app /data

ENV LD_LIBRARY_PATH=/app/bin
ENV LLAMA_CACHE=/data
ENV MODEL_HF_REPO=Qwen/Qwen2.5-14B-Instruct-GGUF
ENV MODEL_HF_QUANT=Q4_K_M
ENV REDRAFT_BASE=http://127.0.0.1:8080
ENV REDRAFT_MODEL=Qwen2.5-14B-Instruct-GGUF

USER user
EXPOSE 7860
CMD ["/entrypoint.sh"]
