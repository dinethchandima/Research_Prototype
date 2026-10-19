"""Serve the web app on http://localhost:8000 (this PC only) with caching turned off,
so the browser always loads the latest page, styles and models."""
import functools
import http.server
from pathlib import Path


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".wasm": "application/wasm"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        # cross-origin isolation: lets ONNX Runtime Web use several CPU threads (SharedArrayBuffer)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


if __name__ == "__main__":
    handler = functools.partial(NoCacheHandler, directory=str(Path(__file__).parent))
    print("Open http://localhost:8000 in Chrome or Edge. Press Ctrl+C to stop.")
    http.server.ThreadingHTTPServer(("127.0.0.1", 8000), handler).serve_forever()
