cmake_minimum_required(VERSION 3.24)
project(high_performance_inference_runtime VERSION 1.0.0 LANGUAGES CXX)

option(INFERENCE_RUNTIME_BUILD_TESTS "Build GoogleTest tests" ON)
option(INFERENCE_RUNTIME_BUILD_BENCHMARKS "Build benchmarks" ON)
option(INFERENCE_RUNTIME_BUILD_SERVER "Build the HTTP server" ON)
option(INFERENCE_RUNTIME_USE_ONNX "Use the ONNX Runtime CPU backend" OFF)
option(INFERENCE_RUNTIME_FETCH_DEPENDENCIES "Download missing pinned dependencies" ON)
option(INFERENCE_RUNTIME_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)
set(INFERENCE_RUNTIME_SANITIZER "" CACHE STRING "Sanitizer: address or thread")
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime SDK directory")

include(FetchContent)
find_package(Threads REQUIRED)
add_library(runtime_build_options INTERFACE)
target_compile_features(runtime_build_options INTERFACE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(runtime_build_options INTERFACE -Wall -Wextra -Wpedantic)
  if(INFERENCE_RUNTIME_WARNINGS_AS_ERRORS)
    target_compile_options(runtime_build_options INTERFACE -Werror)
  endif()
  if(INFERENCE_RUNTIME_SANITIZER STREQUAL "address")
    target_compile_options(runtime_build_options INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(runtime_build_options INTERFACE -fsanitize=address,undefined)
  elseif(INFERENCE_RUNTIME_SANITIZER STREQUAL "thread")
    target_compile_options(runtime_build_options INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(runtime_build_options INTERFACE -fsanitize=thread)
  elseif(NOT INFERENCE_RUNTIME_SANITIZER STREQUAL "")
    message(FATAL_ERROR "Unknown sanitizer: ${INFERENCE_RUNTIME_SANITIZER}")
  endif()
endif()

add_library(inference_runtime
  src/inference/inference_engine.cpp
  src/inference/batch_scheduler.cpp
  src/inference/prediction_result.cpp
  src/metrics/latency_tracker.cpp
  src/metrics/runtime_metrics.cpp
  src/server/inference_service.cpp)
target_sources(inference_runtime PRIVATE src/memory/reusable_buffer_pool.cpp)
target_include_directories(inference_runtime PUBLIC include)
target_link_libraries(inference_runtime PUBLIC runtime_build_options Threads::Threads)

if(INFERENCE_RUNTIME_USE_ONNX)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include" REQUIRED)
  find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)
  add_library(onnxruntime_sdk SHARED IMPORTED)
  set_target_properties(onnxruntime_sdk PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
  target_link_libraries(inference_runtime PRIVATE onnxruntime_sdk)
  target_compile_definitions(inference_runtime PRIVATE INFERENCE_RUNTIME_HAS_ONNX=1)
endif()

if(INFERENCE_RUNTIME_BUILD_BENCHMARKS)
  add_executable(inference_benchmark benchmarks/benchmark_main.cpp
    benchmarks/queue_benchmark.cpp benchmarks/batching_benchmark.cpp
    benchmarks/inference_runtime_benchmark.cpp benchmarks/buffer_benchmark.cpp)
  target_link_libraries(inference_benchmark PRIVATE inference_runtime)
endif()

if(INFERENCE_RUNTIME_BUILD_SERVER)
  include(cmake/HttpDependencies.cmake)
  add_library(inference_http src/server/http_server.cpp src/server/http_routes.cpp)
  target_link_libraries(inference_http PUBLIC inference_runtime
    PRIVATE runtime_boost_headers nlohmann_json::nlohmann_json)
  add_executable(inference_server src/main.cpp src/shutdown/shutdown_controller.cpp)
  target_link_libraries(inference_server PRIVATE inference_http)
endif()

if(INFERENCE_RUNTIME_BUILD_TESTS)
  find_package(GTest 1.14 QUIET)
  if(NOT GTest_FOUND)
    if(NOT INFERENCE_RUNTIME_FETCH_DEPENDENCIES)
      message(FATAL_ERROR "GoogleTest is required; install it or enable dependency downloads")
    endif()
    FetchContent_Declare(googletest
      URL https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.15.2
      URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926)
    set(BUILD_GMOCK OFF CACHE BOOL "Build GoogleMock" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  enable_testing()
  add_executable(inference_tests tests/blocking_queue_test.cpp tests/thread_pool_test.cpp
    tests/batch_scheduler_test.cpp tests/inference_service_test.cpp tests/metrics_test.cpp
    tests/reusable_buffer_pool_test.cpp tests/inference_engine_test.cpp)
  target_link_libraries(inference_tests PRIVATE GTest::gtest_main)
  if(INFERENCE_RUNTIME_BUILD_SERVER)
    target_sources(inference_tests PRIVATE tests/http_routes_test.cpp tests/http_server_test.cpp)
    target_include_directories(inference_tests PRIVATE src/server)
    target_link_libraries(inference_tests PRIVATE inference_http runtime_boost_headers nlohmann_json::nlohmann_json)
  else()
    target_link_libraries(inference_tests PRIVATE inference_runtime)
  endif()
  include(GoogleTest)
  gtest_discover_tests(inference_tests DISCOVERY_TIMEOUT 30
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTIES TIMEOUT 30)
  if(INFERENCE_RUNTIME_BUILD_SERVER)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
      add_test(NAME HttpServer.SmokeAndSignals COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/http_smoke_test.py $<TARGET_FILE:inference_server>)
      set_tests_properties(HttpServer.SmokeAndSignals PROPERTIES TIMEOUT 45)
    endif()
  endif()
endif()

foreach(runtime_target inference_runtime inference_http inference_server inference_tests inference_benchmark)
  if(TARGET ${runtime_target})
    set_target_properties(${runtime_target} PROPERTIES
      CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
  endif()
endforeach()
