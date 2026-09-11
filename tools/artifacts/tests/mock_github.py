#!/usr/bin/env python3
"""mock_github -- CPU-only stand-in for the GitHub Releases API used by the release-publish tests.

    mock_github.py [--fault MODE] [--port 0] [--state FILE]

Implements only what tools/artifacts/github_release_publish.py calls: commit lookup, release by tag, create
release, upload asset, list assets, download asset, get release, patch release. `--fault` injects one failure so
that the error paths can be tested without touching the real API:
  none unauthorized forbidden notfound-commit tag-exists unprocessable server-error-upload invalid-json
  wrong-id truncated-asset asset-exists timeout incomplete-set publish-not-applied
Prints "MOCK_READY <port>" on stdout when listening.
"""
import argparse
import hashlib
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"releases": {}, "assets": {}, "tags": {}, "next_id": 1000, "fault": "none"}
LOCK = threading.Lock()


def nid():
    STATE["next_id"] += 1
    return STATE["next_id"]


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj=None, raw=None, ctype="application/json"):
        body = raw if raw is not None else json.dumps(obj if obj is not None else {}).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self):
        f = STATE["fault"]
        if f == "unauthorized":
            self._send(401, {"message": "Bad credentials"}); return False
        if f == "forbidden":
            self._send(403, {"message": "Resource not accessible"}); return False
        if not (self.headers.get("Authorization") or "").startswith("Bearer "):
            self._send(401, {"message": "Requires authentication"}); return False
        return True

    def do_GET(self):
        f = STATE["fault"]
        if not self._auth_ok():
            return
        if f == "timeout":
            time.sleep(5)
        m = re.match(r"/repos/([^/]+)/([^/]+)/commits/([0-9a-f]+)$", self.path)
        if m:
            if f == "notfound-commit":
                return self._send(404, {"message": "No commit found"})
            return self._send(200, {"sha": m.group(3)})
        m = re.match(r"/repos/[^/]+/[^/]+/releases/tags/(.+)$", self.path)
        if m:
            tag = m.group(1)
            if f == "tag-exists" or tag in STATE["tags"]:
                return self._send(200, STATE["releases"].get(STATE["tags"].get(tag), {"id": 1, "tag_name": tag}))
            return self._send(404, {"message": "Not Found"})
        m = re.match(r"/repos/[^/]+/[^/]+/releases/(\d+)/assets$", self.path)
        if m:
            rid = int(m.group(1))
            items = [{"id": a["id"], "name": a["name"], "size": a["size"], "state": "uploaded"}
                     for a in STATE["assets"].values() if a["release"] == rid]
            if f == "incomplete-set" and items:
                items = items[:-1]
            return self._send(200, items)
        m = re.match(r"/repos/[^/]+/[^/]+/releases/assets/(\d+)$", self.path)
        if m:
            a = STATE["assets"].get(int(m.group(1)))
            if not a:
                return self._send(404, {"message": "Not Found"})
            data = a["data"]
            if f == "truncated-asset":
                data = data[: max(0, len(data) - 1)]
            return self._send(200, raw=data, ctype="application/octet-stream")
        m = re.match(r"/repos/[^/]+/[^/]+/releases/(\d+)$", self.path)
        if m:
            r = STATE["releases"].get(int(m.group(1)))
            return self._send(200, r) if r else self._send(404, {"message": "Not Found"})
        self._send(404, {"message": "Not Found"})

    def do_POST(self):
        f = STATE["fault"]
        if not self._auth_ok():
            return
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        if re.match(r"/repos/[^/]+/[^/]+/releases$", self.path):
            if f == "unprocessable":
                return self._send(422, {"message": "Validation Failed"})
            if f == "invalid-json":
                return self._send(201, raw=b"<html>not json</html>")
            req = json.loads(body.decode() or "{}")
            rid = nid()
            # "wrong-id": the response names an id that does not exist server-side (the release is stored under rid)
            rid_reported = rid + 7 if f == "wrong-id" else rid
            rel = {"id": rid_reported, "tag_name": req.get("tag_name"), "target_commitish": req.get("target_commitish"),
                   "draft": bool(req.get("draft")), "prerelease": bool(req.get("prerelease")),
                   "upload_url": f"http://127.0.0.1:{STATE['port']}/uploads/releases/{rid}/assets{{?name,label}}"}
            with LOCK:
                STATE["releases"][rid] = dict(rel, id=rid)
                STATE["tags"][req.get("tag_name")] = rid
                if f == "asset-exists":
                    aid = nid()
                    STATE["assets"][aid] = {"id": aid, "release": rid_reported, "name": "SHA256SUMS", "size": 3, "data": b"old"}
            return self._send(201, rel)
        m = re.match(r"/uploads/releases/(\d+)/assets$", self.path.split("?")[0])
        if m:
            if f == "server-error-upload":
                return self._send(500, {"message": "Server Error"})
            rid = int(m.group(1))
            q = re.search(r"[?&]name=([^&]+)", self.path)
            name = q.group(1) if q else "unnamed"
            with LOCK:
                if any(a["release"] == rid and a["name"] == name for a in STATE["assets"].values()):
                    return self._send(422, {"message": "Validation Failed: already_exists"})
                aid = nid()
                STATE["assets"][aid] = {"id": aid, "release": rid, "name": name, "size": len(body), "data": body}
            return self._send(201, {"id": aid, "name": name, "size": len(body), "state": "uploaded"})
        self._send(404, {"message": "Not Found"})

    def do_PATCH(self):
        f = STATE["fault"]
        if not self._auth_ok():
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads((self.rfile.read(n) if n else b"{}").decode() or "{}")
        m = re.match(r"/repos/[^/]+/[^/]+/releases/(\d+)$", self.path)
        if not m:
            return self._send(404, {"message": "Not Found"})
        rid = int(m.group(1))
        r = STATE["releases"].get(rid)
        if not r:
            return self._send(404, {"message": "Not Found"})
        with LOCK:
            if f != "publish-not-applied":
                r.update({k: bool(v) for k, v in req.items() if k in ("draft", "prerelease")})
        return self._send(200, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", default="none"); ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args()
    STATE["fault"] = a.fault
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    STATE["port"] = srv.server_address[1]
    print(f"MOCK_READY {srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
