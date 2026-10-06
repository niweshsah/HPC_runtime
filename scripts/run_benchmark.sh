#!/usr/bin/env bash
set -euo pipefail

project_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_executable="${INFERENCE_BENCHMARK_EXECUTABLE:-${project_directory}/build/inference_benchmark}"
if [[ ! -x "$benchmark_executable" ]]; then
    echo "Build inference_benchmark first or set INFERENCE_BENCHMARK_EXECUTABLE." >&2
    exit 1
fi
exec "$benchmark_executable" --compare "$@"
