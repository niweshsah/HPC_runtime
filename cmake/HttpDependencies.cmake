find_package(Boost 1.80 CONFIG QUIET)
if(Boost_FOUND)
  add_library(runtime_boost_headers INTERFACE)
  if(TARGET Boost::headers)
    target_link_libraries(runtime_boost_headers INTERFACE Boost::headers)
  else()
    target_include_directories(runtime_boost_headers SYSTEM INTERFACE ${Boost_INCLUDE_DIRS})
  endif()
else()
  if(NOT INFERENCE_RUNTIME_FETCH_DEPENDENCIES)
    message(FATAL_ERROR "Boost >= 1.80 is required for the server")
  endif()
  FetchContent_Declare(boost_headers
    URL https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.bz2
    URL_HASH SHA256=af57be25cb4c4f4b413ed692fe378affb4352ea50fbe294a11ef548f4d527d89
    SOURCE_SUBDIR headers_only)
  FetchContent_MakeAvailable(boost_headers)
  add_library(runtime_boost_headers INTERFACE)
  target_include_directories(runtime_boost_headers SYSTEM INTERFACE ${boost_headers_SOURCE_DIR})
endif()

find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
  if(NOT INFERENCE_RUNTIME_FETCH_DEPENDENCIES)
    message(FATAL_ERROR "nlohmann_json >= 3.11 is required for the server")
  endif()
  FetchContent_Declare(nlohmann_json
    URL https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.11.3
    URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406)
  FetchContent_MakeAvailable(nlohmann_json)
endif()
