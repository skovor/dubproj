"""Serve a human-only gold-set review queue.

The API intentionally exposes no ASR/CTC scores or pipeline verdicts.  A
reviewer claims a clip, submits one or more human defect labels, and a lead
reviewer can adjudicate disagreements.  All mutations go through
``GoldsetStore`` so the SQLite audit trail remains authoritative.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing_pipeline.goldset import GoldsetStore, HumanLabel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value

        def do_GET(self):
            query = parse_qs(urlparse(self.path).query)
            split = (query.get("split") or [None])[0]
            if split == "hidden_test":
                self._send(403, {"error": "hidden test is not exposed through the normal review API"})
                return
            with GoldsetStore(args.database) as store:
                clips = [clip.review_payload() for clip in store.clips() if clip.split != "hidden_test" and (split is None or clip.split == split)]
            self._send(200, {"schema": "goldset-review-v2", "clips": clips})

        def do_POST(self):
            try:
                body = self._body()
                path = urlparse(self.path).path
                with GoldsetStore(args.database) as store:
                    self._dispatch_post(path, body, store)
                return
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})

        def _dispatch_post(self, path: str, body: dict, store: GoldsetStore) -> None:
            if path == "/claim":
                clip = store.claim(str(body.get("reviewer_id", "")), split=body.get("split"), lease_seconds=int(body.get("lease_seconds", 900)))
                self._send(200, {"clip": clip.review_payload() if clip else None})
                return
            if path == "/release":
                store.release_claim(str(body["clip_id"]), str(body["reviewer_id"]))
                self._send(200, {"released": True})
                return
            if path == "/label":
                allowed = {"clip_id", "reviewer_id", "label", "labels", "severity", "region_start", "region_end", "affected_tokens", "comment", "confidence", "needs_context"}
                unknown = set(body) - allowed
                if unknown:
                    raise ValueError(f"unknown label fields: {sorted(unknown)}")
                value = dict(body)
                value["affected_tokens"] = tuple(value.get("affected_tokens") or ())
                value["labels"] = tuple(value.get("labels") or ())
                store.save_label(HumanLabel(**value))
                self._send(200, {"saved": True})
                return
            if path == "/adjudicate":
                store.adjudicate(str(body["clip_id"]), str(body["adjudicator_id"]), body.get("consensus_labels") or (), comment=str(body.get("comment", "")))
                self._send(200, {"adjudicated": True})
                return
            raise ValueError("unknown endpoint")

        def log_message(self, *_):
            pass

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
