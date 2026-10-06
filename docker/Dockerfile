FROM ubuntu:24.04 AS build
ARG DEBIAN_FRONTEND=noninteractive
ARG USE_ONNX=OFF
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ca-certificates curl python3 libboost-dev \
    nlohmann-json3-dev libgtest-dev && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/onnxruntime/lib && \
    if [ "$USE_ONNX" = "ON" ]; then \
      case "$TARGETARCH" in \
        amd64) sdk_arch=x64; sdk_sha=8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3 ;; \
        arm64) sdk_arch=aarch64; sdk_sha=bb76395092d150b52c7092dc6b8f2fe4d80f0f3bf0416d2f269193e347e24702 ;; \
        *) echo "Unsupported ONNX architecture: $TARGETARCH" >&2; exit 1 ;; \
      esac; \
      curl -fL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-${sdk_arch}-1.22.0.tgz" -o /tmp/onnxruntime.tgz && \
      echo "$sdk_sha  /tmp/onnxruntime.tgz" | sha256sum -c - && \
      tar -xzf /tmp/onnxruntime.tgz -C /opt/onnxruntime --strip-components=1 && \
      rm /tmp/onnxruntime.tgz; \
    fi
WORKDIR /source
COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DINFERENCE_RUNTIME_USE_ONNX=${USE_ONNX} -DONNXRUNTIME_ROOT=/opt/onnxruntime \
    -DINFERENCE_RUNTIME_WARNINGS_AS_ERRORS=ON \
    -DINFERENCE_RUNTIME_FETCH_DEPENDENCIES=OFF && \
    cmake --build build -j 2 && ctest --test-dir build --output-on-failure

FROM ubuntu:24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 ca-certificates && \
    rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 10001 inference
COPY --from=build /source/build/inference_server /usr/local/bin/inference_server
COPY --from=build /source/build/inference_benchmark /usr/local/bin/inference_benchmark
COPY --from=build /opt/onnxruntime/lib/ /usr/local/lib/
RUN ldconfig
USER inference
WORKDIR /home/inference
EXPOSE 8080
STOPSIGNAL SIGTERM
ENTRYPOINT ["inference_server"]
CMD ["--address", "0.0.0.0", "--port", "8080", "--reuse-buffers"]
