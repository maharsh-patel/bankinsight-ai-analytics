"""
BankInsight AI — Dashboard Local Server
========================================
Run this script to serve the dashboard at http://localhost:8050

Usage:
    python src/serve_dashboard.py

Opens the browser automatically.
Press Ctrl+C to stop.
"""

import os
import sys
import webbrowser
import http.server
import socketserver
import threading

PORT      = 8050
DASH_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")

class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASH_DIR, **kwargs)
    def log_message(self, fmt, *args):
        pass  # suppress request logs; keep console clean

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    url = f"http://localhost:{PORT}"
    print("=" * 50)
    print("BankInsight AI Dashboard")
    print("=" * 50)
    print(f"Serving: {DASH_DIR}")
    print(f"URL    : {url}")
    print("Press Ctrl+C to stop")
    print()

    # Open browser after a short delay
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    with socketserver.TCPServer(("", PORT), _Handler) as httpd:
        httpd.allow_reuse_address = True
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()
