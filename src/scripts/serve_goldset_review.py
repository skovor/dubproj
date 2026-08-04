"""Serve a read-only JSON review queue; labels are written through SQLite API."""
from __future__ import annotations
import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing_pipeline.goldset import GoldsetStore

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("database"); parser.add_argument("--port", type=int, default=8765); args = parser.parse_args(); store = GoldsetStore(args.database)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = {"clips": [clip.review_payload() for clip in store.clips()]}
            data = json.dumps(payload, ensure_ascii=False).encode(); self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def log_message(self, *_): pass
    try: ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    finally: store.close()
    return 0
if __name__ == "__main__": raise SystemExit(main())
