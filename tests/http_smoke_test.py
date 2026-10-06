"""Exercise the real listener and process-level signal shutdown with stdlib only."""

import concurrent.futures
import http.client
import json
import signal
import socket
import subprocess
import sys


def exchange(port, method, path, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_server(executable, shutdown_signal, exercise_routes):
    process = subprocess.Popen(
        [executable, "--port", "0", "--max-body-bytes", "1024", "--reuse-buffers"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    idle_connection = None
    try:
        startup_line = process.stdout.readline().strip()
        if not startup_line.startswith("Listening on "):
            _, diagnostics = process.communicate(timeout=5)
            raise RuntimeError(f"Server failed to start: {startup_line}\n{diagnostics}")
        port = int(startup_line.rsplit(":", 1)[1])
        assert exchange(port, "GET", "/health")[0] == 200
        if exercise_routes:
            assert exchange(port, "POST", "/predict", '{"input":[0.1,0.2,0.3]}')[0] == 200
            assert exchange(port, "POST", "/predict", "invalid")[0] == 400
            assert exchange(port, "POST", "/predict", '{"input":[1]}')[0] == 400
            assert exchange(port, "GET", "/predict")[0] == 405
            assert exchange(port, "GET", "/missing")[0] == 404
            assert exchange(port, "POST", "/predict", " " * 2048)[0] == 413
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as clients:
                responses = list(clients.map(
                    lambda _: exchange(port, "POST", "/predict", '{"input":[1,2,3]}'), range(64)
                ))
            assert all(status == 200 and body["status"] == "success" for status, body in responses)
            request_ids = {body["request_id"] for _, body in responses}
            assert len(request_ids) == 64
            metrics = exchange(port, "GET", "/metrics")[1]
            assert metrics["completed_requests"] == 65, metrics
            assert metrics["failed_requests"] == 0, metrics
        idle_connection = socket.create_connection(("127.0.0.1", port), timeout=10)
        idle_connection.sendall(b"GET /health HTTP/1.1\r\n")
        process.send_signal(shutdown_signal)
        _, diagnostics = process.communicate(timeout=10)
        assert process.returncode == 0, diagnostics
        assert "event=service_stopped" in diagnostics, diagnostics
    finally:
        if idle_connection is not None:
            idle_connection.close()
        if process.poll() is None:
            process.kill()
            process.communicate()


if __name__ == "__main__":
    test_server(sys.argv[1], signal.SIGTERM, True)
    test_server(sys.argv[1], signal.SIGINT, False)
    print("HTTP endpoints, concurrent predictions, limits, SIGTERM and SIGINT passed")
