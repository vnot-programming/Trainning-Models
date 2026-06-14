import asyncio
import os
import time
import subprocess
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import httpx
import logging
from logging.handlers import RotatingFileHandler

# ==== Konfigurasi Logging ====
LOG_DIR = "/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/asprigate_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "asprigate.log")

logger = logging.getLogger("AspriGate")
logger.setLevel(logging.INFO)
# Rotasi log maksimal 50MB, simpan 1 backup
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=50*1024*1024, backupCount=10)
formatter = logging.Formatter('%(asctime)s: %(levelname)s - %(message)s', datefmt='%d-%m-%Y, %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Log juga ke stdout agar terlihat di tmux
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

templates = Jinja2Templates(directory="/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/templates")

# Global State
LAST_ACTIVITY_TIME = time.time()
IDLE_TIMEOUT_SEC = 300  # 5 minutes
IS_BOOTING = False
IS_IDLE = False
STATUS_MESSAGE = "Inisialisasi sistem..."

def update_activity():
    global LAST_ACTIVITY_TIME, IS_IDLE
    LAST_ACTIVITY_TIME = time.time()
    IS_IDLE = False

def check_gpu_status():
    """Mengecek apakah ada job RUNNING di squeue."""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "vnot"), "-t", "R", "-h", "-o", "%i"],
            capture_output=True, text=True
        )
        return len(result.stdout.strip()) > 0
    except Exception:
        return False

def check_port_listening(port):
    """Mengecek apakah Engine lokal sudah listen."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_engine_scripts():
    """Eksekusi skrip engine."""
    global STATUS_MESSAGE
    STATUS_MESSAGE = "Memuat konfigurasi mesin kecerdasan buatan..."
    logger.info(">>> MEMULAI PROSES BOOTING ENGINE (Scale-to-Zero Wake Up)...")
    
    # RVM Backend & Frontend
    subprocess.run(["bash", "/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/RVM/start_rvm.sh", "backend"])
    subprocess.run(["bash", "/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/RVM/start_rvm.sh", "frontend"])
    
    # ComfyUI Daemon
    comfui_dir = "/data/users/g6717500336/singularity/comfui"
    if os.path.exists(comfui_dir):
        # Jalankan via nohup agar background
        subprocess.Popen(
            ["bash", f"{comfui_dir}/run_comfui_daemon.sh"],
            cwd=comfui_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )

def kill_engine_scripts():
    """Mematikan proses engine untuk hemat VRAM, tanpa scancel."""
    logger.info(">>> IDLE TIMEOUT TERCAPAI. Mematikan Engine RVM/ComfyUI secara aman...")
    # RVM Tmux
    subprocess.run(["tmux", "kill-session", "-t", "rvm_backend"], stderr=subprocess.DEVNULL)
    subprocess.run(["tmux", "kill-session", "-t", "rvm_frontend"], stderr=subprocess.DEVNULL)
    
    # ComfyUI processes
    subprocess.run(["pkill", "-f", "run_comfui_daemon.sh"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "python main.py --listen 0.0.0.0"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "ssh -N.*-R"], stderr=subprocess.DEVNULL) # Kill reverse tunnels

async def booting_manager():
    """Background task yang memantau proses booting dan idle timeout."""
    global IS_BOOTING, STATUS_MESSAGE, LAST_ACTIVITY_TIME, IS_IDLE
    while True:
        await asyncio.sleep(2)
        
        # Idle check
        if time.time() - LAST_ACTIVITY_TIME > IDLE_TIMEOUT_SEC:
            if not IS_IDLE:
                # Pkill engines
                kill_engine_scripts()
                IS_IDLE = True
        
        if not IS_BOOTING:
            continue
            
        # Jika sedang booting, cek GPU
        if not check_gpu_status():
            STATUS_MESSAGE = "Menunggu alokasi Node komputasi dari Slurm..."
        else:
            if STATUS_MESSAGE == "Menunggu alokasi Node komputasi dari Slurm..." or STATUS_MESSAGE == "Inisialisasi sistem...":
                start_engine_scripts()
            STATUS_MESSAGE = "Menghangatkan model AI... (Menunggu Port Aktif)"
            
            # Cek kesiapan port RVM Frontend (8601), Backend (8602), ComfyUI (19195)
            # Asumsikan jika 8601 & 8602 siap, sistem cukup ready.
            if check_port_listening(8601) and check_port_listening(8602):
                STATUS_MESSAGE = "Proses Berhasil, mengalihkan halaman!"
                IS_BOOTING = False

# ==== FastAPI Factory ====
def create_proxy_app(target_port: int):
    app = FastAPI()
    http_client = httpx.AsyncClient(verify=False, timeout=60.0)
    
    @app.get("/asprigate-status")
    async def get_status(request: Request):
        update_activity()
        ready = check_port_listening(target_port)
        msg = "Proses Berhasil, mengalihkan halaman!" if ready else STATUS_MESSAGE
        
        client_ip = request.headers.get("x-real-ip", request.client.host if request.client else "Unknown")
        logger.info(f"[POLL] {client_ip} -> GET /asprigate-status | TargetPort: {target_port} | Ready: {ready}")
        
        return JSONResponse(
            {"message": msg, "ready": ready},
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
        )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    async def proxy(request: Request, path: str):
        global IS_BOOTING
        update_activity()
        
        client_ip = request.headers.get("x-real-ip", request.client.host if request.client else "Unknown")
        
        target_url = f"http://127.0.0.1:{target_port}{request.url.path}"
        if request.url.query:
            target_url += f"?{request.url.query}"
            
        logger.info(f"[PROXY] {client_ip} -> {request.method} {request.url.path} -> {target_url}")
            
        try:
            # Proxy request
            headers = dict(request.headers)
            headers.pop("host", None) # Jangan overwrite host header ke localhost jika tidak perlu, atau biarkan httpx handle
            
            req = http_client.build_request(
                request.method,
                target_url,
                headers=headers,
                content=await request.body()
            )
            resp = await http_client.send(req, stream=True)
            
            # Extract response
            headers = dict(resp.headers)
            headers.pop("content-length", None)
            headers.pop("content-encoding", None)
            
            return Response(
                content=await resp.aread(),
                status_code=resp.status_code,
                headers=headers
            )
        except httpx.RequestError:
            # Target is Down!
            if not IS_BOOTING:
                IS_BOOTING = True
            
            # Jika Accept HTML, tampilkan Loading Screen. Jika tidak, return 503.
            accept = request.headers.get("accept", "")
            if "text/html" in accept or path == "":
                # i18n detection
                country = request.headers.get("cf-ipcountry", "US").upper()
                lang = "id" if country == "ID" else ("th" if country == "TH" else "en")
                
                return templates.TemplateResponse("loading_template.html", {"request": request, "lang": lang})
            else:
                return JSONResponse({"error": "Service Booting", "status": STATUS_MESSAGE}, status_code=503)

    return app

# Bikin 3 aplikasi
app_rvm_front = create_proxy_app(8601)  # Target RVM Frontend yang baru
app_rvm_back = create_proxy_app(8602)   # Target RVM Backend yang baru
app_comfyui = create_proxy_app(8188)    # Target ComfyUI SSH Reverse Tunnel

async def main():
    asyncio.create_task(booting_manager())
    
    config1 = uvicorn.Config(app_rvm_front, port=8501, host="0.0.0.0", log_level="error")
    config2 = uvicorn.Config(app_rvm_back, port=8502, host="0.0.0.0", log_level="error")
    config3 = uvicorn.Config(app_comfyui, port=19095, host="0.0.0.0", log_level="error")
    
    server1 = uvicorn.Server(config1)
    server2 = uvicorn.Server(config2)
    server3 = uvicorn.Server(config3)
    
    print("🚀 AspriGate berjalan melayani Port 8501, 8502, 19095...")
    await asyncio.gather(server1.serve(), server2.serve(), server3.serve())

if __name__ == "__main__":
    asyncio.run(main())
