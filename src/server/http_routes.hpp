#pragma once

#include <boost/beast/http.hpp>
#include <nlohmann/json.hpp>

#include "inference_runtime/server/inference_service.hpp"

namespace inference_runtime {
using HttpRequest = boost::beast::http::request<boost::beast::http::string_body>;
using HttpResponse = boost::beast::http::response<boost::beast::http::string_body>;
HttpResponse createJsonResponse(boost::beast::http::status status, nlohmann::json body);
HttpResponse buildHttpResponse(const HttpRequest& request, InferenceService& service);
}  // namespace inference_runtime
