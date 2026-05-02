import subprocess
import threading
import time
import os
import logging

# Configure logging to be visible in terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - GPU-FAN - %(message)s')

def get_gpu_data():
    """Returns list of (index, temp) for all GPUs."""
    try:
        res = subprocess.check_output([
            'nvidia-smi', 
            '--query-gpu=index,temperature.gpu', 
            '--format=csv,noheader,nounits'
        ])
        lines = res.decode().strip().split('\n')
        data = []
        for line in lines:
            if line.strip():
                parts = line.split(',')
                if len(parts) == 2:
                    idx, temp = parts
                    data.append((int(idx.strip()), int(temp.strip())))
        return data
    except Exception as e:
        logging.error(f"Error getting GPU data: {e}")
        return []

def set_fan_speed(gpu_idx, speed_percent):
    """Sets fan speed for a specific GPU using nvidia-settings."""
    try:
        # Note: nvidia-settings requires a display. 
        # In headless environments, this might need X-server running.
        env = os.environ.copy()
        if "DISPLAY" not in env:
            env["DISPLAY"] = ":0"
            
        # 1. Enable manual control (CoolBits must be enabled in Xorg)
        # We try to apply to all fans for that GPU (usually fan:0, fan:1 etc)
        # For simplicity and most common cases (1-2 fans), we target fan:0 and fan:1
        subprocess.run([
            'nvidia-settings', '-a', f'[gpu:{gpu_idx}]/GPUFanControlState=1'
        ], env=env, capture_output=True, check=False)
        
        # Apply to fans (we'll try indices 0 and 1)
        for fan_idx in [0, 1]:
            subprocess.run([
                'nvidia-settings', '-a', f'[fan:{fan_idx}]/GPUTargetFanSpeed={speed_percent}'
            ], env=env, capture_output=True, check=False)
            
    except Exception as e:
        logging.error(f"Error setting fan speed for GPU {gpu_idx}: {e}")

def fan_manager_loop():
    logging.info("Starting GPU Fan Manager loop (30s interval)...")
    while True:
        gpu_data = get_gpu_data()
        if not gpu_data:
            logging.warning("No GPU data found. Retrying in 60s...")
            time.sleep(60)
            continue

        for idx, temp in gpu_data:
            # Logic provided by user:
            # >= 70 ==> fan speed 90%
            # >= 60 ==> fan speed 75%
            # >= 50 ==> fan speed 60%
            # >= 30 ==> fan speed 30%
            # >= 0  ==> fan speed 0%
            
            speed = 0
            if temp >= 70:
                speed = 90
            elif temp >= 60:
                speed = 75
            elif temp >= 50:
                speed = 60
            elif temp >= 30:
                speed = 30
            else:
                speed = 0
            
            set_fan_speed(idx, speed)
            # logging.info(f"GPU {idx}: {temp}C -> {speed}%")
            
        time.sleep(30)

def start_fan_manager():
    """Starts the fan manager in a daemon thread so it doesn't block main script."""
    thread = threading.Thread(target=fan_manager_loop, daemon=True)
    thread.start()
    logging.info("GPU Fan Manager background thread initialized.")

if __name__ == "__main__":
    # Manual test run
    print("Testing GPU Fan Manager... Press Ctrl+C to stop.")
    start_fan_manager()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")
