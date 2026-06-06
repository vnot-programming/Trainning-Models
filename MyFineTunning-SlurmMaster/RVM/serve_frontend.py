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
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config_shared import EVAL_API_HOST, EVAL_API_FRONTEND_PORT


class CORSHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + CORS header untuk cross-origin access."""

    def end_headers(self):
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
