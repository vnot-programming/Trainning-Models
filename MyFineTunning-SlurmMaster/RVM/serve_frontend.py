#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serve_frontend.py — Static file server untuk frontend Visual Evaluation
========================================================================
Berjalan di LOGIN NODE. Diakses publik via:
    https://front-rvm.penelitian.my.id (Cloudflare Tunnel → localhost:8501)

PENGGUNAAN:
    source /data/programs/anaconda3/bin/activate yolo_env
    cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster
    python RVM/serve_frontend.py
"""

import os
import sys
import urllib.request
import urllib.error
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config_shared import EVAL_API_HOST, EVAL_API_FRONTEND_PORT, EVAL_API_BACKEND_PORT

def get_cf_headers():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    cf_id = ""
    cf_secret = ""
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("CF-Access-Client-Id="):
                    cf_id = line.split("=", 1)[1]
                elif line.startswith("CF-Access-Client-Secret="):
                    cf_secret = line.split("=", 1)[1]
    return cf_id, cf_secret



class CORSHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + CORS header + API Proxy untuk menghindari blokir CORS Cloudflare."""
    
    def proxy_api(self):
        """Proxy request API langsung ke localhost:8502 (backend RVM) menggunakan Service Token dari .env"""
        url = f"http://127.0.0.1:{EVAL_API_BACKEND_PORT}{self.path}"
        body = None
        if self.command == "POST":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                body = self.rfile.read(content_length)
                
        req = urllib.request.Request(url, data=body, method=self.command)
        
        # Forward original headers except Host
        for key, value in self.headers.items():
            if key.lower() not in ["host", "content-length"]:
                req.add_header(key, value)
                
        # Inject Cloudflare Service Token untuk backend
        cf_id, cf_secret = get_cf_headers()
        if cf_id and cf_secret:
            req.add_header("CF-Access-Client-Id", cf_id)
            req.add_header("CF-Access-Client-Secret", cf_secret)
                
        try:
            with urllib.request.urlopen(req) as response:
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in ["transfer-encoding", "connection"]:
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for key, value in e.headers.items():
                if key.lower() not in ["transfer-encoding", "connection"]:
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f'{{"error": "API Proxy Error", "details": "{str(e)}" }}'.encode("utf-8"))

    def do_OPTIONS(self):
        if self.path.startswith('/api/'):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()
            return
        super().do_OPTIONS()

    def do_POST(self):
        if self.path.startswith('/api/'):
            return self.proxy_api()
        self.send_response(405)
        self.end_headers()

    def do_GET(self):
        if self.path.startswith('/api/'):
            return self.proxy_api()
        if self.path == '/js/env.js':
            self.send_response(200)
            self.send_header("Content-type", "application/javascript")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            # CF tokens tidak perlu lagi terekspos ke sisi klien
            js_content = f'window.RVM_ENV = {{}};'
            self.wfile.write(js_content.encode("utf-8"))
            return
        super().do_GET()

    def end_headers(self):
        if not self.path.startswith('/api/'):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format, *args):
        print(f"[Frontend] {self.address_string()} — {args[0]}", flush=True)


def main():
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    if not os.path.isdir(frontend_dir):
        print(f"[Frontend] ❌ Direktori frontend tidak ditemukan: {frontend_dir}")
        sys.exit(1)

    handler = partial(CORSHandler, directory=frontend_dir)
    server = HTTPServer((EVAL_API_HOST, EVAL_API_FRONTEND_PORT), handler)

    print("=" * 60)
    print("  📱 Visual Evaluation Frontend Server")
    print("=" * 60)
    print(f"  Host    : {EVAL_API_HOST}")
    print(f"  Port    : {EVAL_API_FRONTEND_PORT}")
    print(f"  Serving : {frontend_dir}")
    print(f"  Publik  : https://front-rvm.penelitian.my.id")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Frontend] Server dihentikan.")
        server.server_close()


if __name__ == "__main__":
    main()
