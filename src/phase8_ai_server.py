"""
BankInsight AI — Phase 8: AI API Server
========================================
Lightweight HTTP server exposing the AI layer via a simple REST endpoint.
Used by the dashboard's AI Insights chat panel.

Endpoints:
  POST /api/ask   { "question": "..." }
                  -> { "answer": "...", "source": "...", "model": "..." }
  GET  /api/ping  -> { "status": "ok", "ai_available": true/false }
  GET  /          -> serves dashboard/index.html (+ static files)

Usage:
  python src/phase8_ai_server.py [--port 8050]

Combines Phase 7 dashboard serving with the AI endpoint on one port.
"""

import os
import sys
import json
import argparse
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase8_ai_context import get_context
from phase8_ai_engine  import ask, ai_available

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASH_DIR  = os.path.join(BASE_DIR, "dashboard")

# Pre-build context once at startup (fast, ~6800 chars)
_CTX = None

def _get_ctx():
    global _CTX
    if _CTX is None:
        _CTX = get_context()
    return _CTX


class BankInsightHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress access log noise

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")  # allow dashboard fetch
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_preflight(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._send_cors_preflight()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"

        if path == "/api/ping":
            self._send_json({"status": "ok", "ai_available": ai_available()})
            return

        # Serve static files from dashboard/
        if path == "/" or path == "/index.html":
            file_path = os.path.join(DASH_DIR, "index.html")
        else:
            # Strip leading slash
            rel = path.lstrip("/")
            file_path = os.path.join(DASH_DIR, rel)

        if not os.path.isfile(file_path):
            self.send_error(404, f"File not found: {path}")
            return

        # Detect MIME
        ext = os.path.splitext(file_path)[1].lower()
        mime = {
            ".html": "text/html; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".js":   "application/javascript",
            ".css":  "text/css",
            ".png":  "image/png",
        }.get(ext, "application/octet-stream")

        with open(file_path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")

        if path != "/api/ask":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            data   = json.loads(body.decode("utf-8"))
            question = str(data.get("question", "")).strip()
        except Exception as e:
            self._send_json({"error": f"Bad request: {e}"}, status=400)
            return

        if not question:
            self._send_json({"error": "Missing 'question' field"}, status=400)
            return

        ctx    = _get_ctx()
        result = ask(question, ctx)
        # Return only what the dashboard needs
        self._send_json({
            "answer":  result["answer"],
            "source":  result["source"],
            "model":   result.get("model", ""),
            "error":   result.get("error", ""),
        })


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    # Pre-build context
    print("Building analytical context...")
    _ = _get_ctx()
    print(f"  Context ready: {len(_CTX)} chars")

    url = f"http://localhost:{args.port}"
    print("=" * 50)
    print("BankInsight AI Server")
    print("=" * 50)
    print(f"Dashboard : {url}")
    print(f"AI ping   : {url}/api/ping")
    print(f"AI ask    : POST {url}/api/ask")
    print(f"AI mode   : {'LIVE (OpenAI)' if ai_available() else 'FALLBACK (rule-based)'}")
    print("Press Ctrl+C to stop\n")

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    server = HTTPServer(("", args.port), BankInsightHandler)
    server.allow_reuse_address = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
