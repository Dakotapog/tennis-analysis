#!/usr/bin/env python3
"""
Graphify visualization server — port 7779.
Serves graphify-out/ with Cache-Control: no-cache for graph.json
so F5 reload always fetches fresh data after `graphify update .`.

Nodo-83: Graphify Local HTTP Server (2026-07-11)
"""
import http.server
import os
import sys

PORT = 7779
SERVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphify-out")
NO_CACHE_EXTS = {".json", ".html"}


class GraphifyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVE_DIR, **kwargs)

    def end_headers(self):
        ext = os.path.splitext(self.path.split("?")[0])[1].lower()
        if ext in NO_CACHE_EXTS:
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        else:
            self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Suppress routine GETs; only log errors (4xx/5xx)
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)


def main():
    if not os.path.isdir(SERVE_DIR):
        print(f"ERROR: {SERVE_DIR} not found. Run `graphify update .` first.", file=sys.stderr)
        sys.exit(1)
    server = http.server.HTTPServer(("0.0.0.0", PORT), GraphifyHandler)
    import socket
    wsl_ip = socket.gethostbyname(socket.gethostname())
    print(f"Graphify server: http://localhost:{PORT}/graph.html", flush=True)
    print(f"Desde Windows:   http://{wsl_ip}:{PORT}/graph.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
