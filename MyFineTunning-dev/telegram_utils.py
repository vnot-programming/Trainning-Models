# -*- coding: utf-8 -*-
import os
import requests
import socket
import time

# Jika Anda ingin mengetes koneksi bot, Anda bisa menjalankan perintah ini di terminal:

# bash
# python3 telegram_utils.py

# Konfigurasi Telegram (Dikelola via .env)
_load_dotenv()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")

# Global state untuk heartbeat (10 menit)
_LAST_NOTIF_TIME = 0
HEARTBEAT_INTERVAL = 600  # 10 menit dalam detik

def send_telegram_msg(message: str, force: bool = True):
    """
    Kirim pesan teks ke Telegram.
    """
    global _LAST_NOTIF_TIME
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    current_time = time.time()
    if not force and (current_time - _LAST_NOTIF_TIME < HEARTBEAT_INTERVAL):
        return False

    hostname = socket.gethostname()
    prefix   = f"<b>[MyFineTunning @ {hostname}]</b>\n"
    full_msg = prefix + message

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": full_msg,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            _LAST_NOTIF_TIME = current_time
            return True
    except:
        pass
    return False

def get_yolo_callbacks(model_name: str):
    """
    Callback helper untuk Ultralytics YOLO.
    """
    def on_train_start(trainer):
        send_telegram_msg(f"🚀 <b>Training Started</b>\nModel: <code>{model_name}</code>")

    def on_train_epoch_end(trainer):
        epoch = trainer.epoch + 1
        total = trainer.args.epochs
        loss  = trainer.loss_items[0] if trainer.loss_items else 0
        msg = (f"📈 <b>Progress Update</b>\nModel: {model_name}\nEpoch: {epoch}/{total}\nLoss: {loss:.4f}")
        send_telegram_msg(msg, force=False)

    def on_train_end(trainer):
        send_telegram_msg(f"✅ <b>Training Finished</b>\nModel: <code>{model_name}</code>")

    return {
        "on_train_start": on_train_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_train_end": on_train_end,
    }

if __name__ == "__main__":
    print(f"[Telegram] Mengirim pesan tes ke chat {TELEGRAM_CHAT_ID}...")
    success = send_telegram_msg("🚀 Telegram Notification System is Online!")
    if success:
        print("✅ Berhasil terkirim! Silakan cek Telegram Anda.")
    else:
        print("❌ Gagal mengirim. Cek BOT_TOKEN, CHAT_ID, atau koneksi internet.")
