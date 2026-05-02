#!/usr/bin/env python3
"""
===============================================================================
  Monitor Training Log — Telegram Notification
===============================================================================
  Memantau file log training Mask R-CNN dan mengirim notifikasi Telegram
  ketika epoch mencapai threshold tertentu (default: 98/100).

  Cara mendapatkan Bot Token & Chat ID:
  1. Buka Telegram, cari @BotFather
  2. Kirim /newbot, ikuti instruksi → dapatkan BOT_TOKEN
  3. Kirim pesan apa saja ke bot yang baru dibuat
  4. Buka: https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
  5. Cari "chat":{"id": 123456789} → itu CHAT_ID kamu

  Jalankan:
    # Test notifikasi dulu
      cd /root/MyFineTunning/mask-r-cnn && python3 monitor_training.py --test

  Atau jalankan di background:
    cd /root/MyFineTunning/mask-r-cnn && nohup python3 monitor_training.py > monitor.log 2>&1 &
===============================================================================
"""

import os
import re
import time
import socket
import datetime
import requests
import argparse

# ═══════════════════════════════════════════════════════════════════════════════
#  KONFIGURASI — Isi sebelum menjalankan!
# ═══════════════════════════════════════════════════════════════════════════════

# Telegram Bot
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8208480895:AAHBZfjMzqfUXMwfyvxvtX-vkOQ4wuF5vTo")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "7174876589")

# File log yang dipantau
LOG_FILES = [
    "/root/MyFineTunning/mask-r-cnn/maskrcnn_train.log",
    "/root/MyFineTunning/mask-r-cnn/train_multigpu.log",
]

# Threshold epoch untuk notifikasi
NOTIFY_EPOCHS = [95, 98, 99, 100]

# Interval polling (detik)
POLL_INTERVAL = 10

# Interval heartbeat — kirim status ke Telegram (menit)
HEARTBEAT_INTERVAL = 30

# ═══════════════════════════════════════════════════════════════════════════════
#  REGEX PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Mencocokkan: Epoch [98/100]  Batch [xxx/xxx]  Loss: x.xxxx
# Mencocokkan: Epoch [98/100]  Train Loss: x.xxxx | Val Loss: x.xxxx | LR: x.xxxx
EPOCH_PATTERN = re.compile(
    r"Epoch\s*\[(\d+)/(\d+)\]"
)

# Pattern untuk summary line (end of epoch)
EPOCH_SUMMARY_PATTERN = re.compile(
    r"Epoch\s*\[(\d+)/(\d+)\]\s+Train Loss:\s*([\d.]+)\s*\|\s*Val Loss:\s*([\d.]+)\s*\|\s*LR:\s*([\d.]+)"
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TELEGRAM NOTIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def send_telegram(message: str) -> bool:
    """Kirim pesan ke Telegram Bot."""
    if "ISI_" in TELEGRAM_BOT_TOKEN or "ISI_" in TELEGRAM_CHAT_ID:
        print(f"⚠️  Telegram belum dikonfigurasi! Pesan ditampilkan di console saja:")
        print(f"   {message}")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            print(f"✅ Telegram terkirim: Epoch notifikasi berhasil dikirim")
            return True
        else:
            print(f"❌ Telegram gagal: {resp.status_code} — {resp.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  LOG MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class LogMonitor:
    """Monitor satu file log training."""

    def __init__(self, filepath: str, notify_epochs: list[int]):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.notify_epochs = set(notify_epochs)
        self.notified = set()           # epoch yang sudah dikirim notifikasi
        self.last_position = 0          # posisi baca terakhir
        self.last_epoch = 0             # epoch terakhir yang terdeteksi
        self.last_total = 0             # total epoch
        self.last_loss = ""             # info loss terakhir

        # Mulai dari akhir file (skip history)
        if os.path.exists(filepath):
            self.last_position = os.path.getsize(filepath)
            print(f"📄 Monitoring: {filepath}")
            print(f"   Mulai dari posisi: {self.last_position} bytes (skip history)")
        else:
            print(f"⚠️  File belum ada: {filepath} — menunggu file dibuat...")

    def check(self) -> list[str]:
        """
        Baca baris baru dari log file.
        Return list of notification messages (jika ada).
        """
        if not os.path.exists(self.filepath):
            return []

        current_size = os.path.getsize(self.filepath)

        # File di-truncate/recreate?
        if current_size < self.last_position:
            print(f"🔄 File {self.filename} direset, baca dari awal")
            self.last_position = 0

        if current_size == self.last_position:
            return []

        messages = []

        try:
            with open(self.filepath, "r", errors="replace") as f:
                f.seek(self.last_position)
                new_content = f.read()
                self.last_position = f.tell()
        except Exception as e:
            print(f"❌ Error membaca {self.filename}: {e}")
            return []

        for line in new_content.splitlines():
            # Cek summary line (akhir epoch)
            summary_match = EPOCH_SUMMARY_PATTERN.search(line)
            if summary_match:
                epoch = int(summary_match.group(1))
                total = int(summary_match.group(2))
                train_loss = summary_match.group(3)
                val_loss = summary_match.group(4)
                lr = summary_match.group(5)

                self.last_epoch = epoch
                self.last_total = total
                self.last_loss = f"Train Loss: {train_loss} | Val Loss: {val_loss} | LR: {lr}"

                if epoch in self.notify_epochs and epoch not in self.notified:
                    self.notified.add(epoch)
                    msg = self._build_message(epoch, total, train_loss, val_loss, lr)
                    messages.append(msg)
                continue

            # Cek epoch biasa (batch line) — hanya update tracker
            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                total = int(epoch_match.group(2))
                self.last_epoch = epoch
                self.last_total = total

                # Untuk file yang tidak punya summary line,
                # kirim notifikasi saat pertama kali epoch terdeteksi
                if epoch in self.notify_epochs and epoch not in self.notified:
                    self.notified.add(epoch)
                    # Extract loss jika ada
                    loss_match = re.search(r"Loss:\s*([\d.]+)", line)
                    loss_info = loss_match.group(1) if loss_match else "N/A"
                    msg = self._build_message_simple(epoch, total, loss_info)
                    messages.append(msg)

        return messages

    def _build_message(self, epoch, total, train_loss, val_loss, lr):
        """Format pesan notifikasi lengkap."""
        hostname = socket.gethostname()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        progress = epoch / total * 100

        icon = "🏁" if epoch == total else "🔔"

        return (
            f"{icon} *Training Update*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📄 Log: `{self.filename}`\n"
            f"🖥 Host: `{hostname}`\n"
            f"⏰ Waktu: {now}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📊 *Epoch {epoch}/{total}* ({progress:.0f}%)\n"
            f"📉 Train Loss: `{train_loss}`\n"
            f"📈 Val Loss: `{val_loss}`\n"
            f"🔧 LR: `{lr}`\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"{'✅ *Training Selesai!*' if epoch == total else f'⏳ Sisa {total - epoch} epoch lagi'}"
        )

    def _build_message_simple(self, epoch, total, loss_info):
        """Format pesan notifikasi sederhana (tanpa val loss)."""
        hostname = socket.gethostname()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        progress = epoch / total * 100

        icon = "🏁" if epoch == total else "🔔"

        return (
            f"{icon} *Training Update*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📄 Log: `{self.filename}`\n"
            f"🖥 Host: `{hostname}`\n"
            f"⏰ Waktu: {now}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📊 *Epoch {epoch}/{total}* ({progress:.0f}%)\n"
            f"📉 Loss: `{loss_info}`\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"{'✅ *Training Selesai!*' if epoch == total else f'⏳ Sisa {total - epoch} epoch lagi'}"
        )

    def status(self):
        """Return current status string."""
        if self.last_epoch > 0:
            return f"{self.filename}: Epoch [{self.last_epoch}/{self.last_total}]"
        return f"{self.filename}: Menunggu data..."

    def status_detail(self):
        """Return detailed status for heartbeat message."""
        if self.last_epoch > 0:
            progress = self.last_epoch / self.last_total * 100 if self.last_total > 0 else 0
            bar_filled = int(progress / 5)  # 20 char bar
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            loss_line = f"   📉 {self.last_loss}" if self.last_loss else ""
            return (
                f"📄 `{self.filename}`\n"
                f"   📊 Epoch *{self.last_epoch}/{self.last_total}* ({progress:.0f}%)\n"
                f"   {bar}\n"
                f"{loss_line}"
            )
        return f"📄 `{self.filename}`\n   ⏳ Menunggu data..."


# ═══════════════════════════════════════════════════════════════════════════════
#  HEARTBEAT
# ═══════════════════════════════════════════════════════════════════════════════

def _send_heartbeat(monitors: list[LogMonitor]):
    """Kirim status periodik ke Telegram."""
    hostname = socket.gethostname()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    details = "\n\n".join(m.status_detail() for m in monitors)

    msg = (
        f"💓 *Heartbeat — Training Status*\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🖥 Host: `{hostname}`\n"
        f"⏰ Waktu: {now}\n"
        f"━━━━━━━━━━━━━━━━━━━\n\n"
        f"{details}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📡 Laporan otomatis setiap 30 menit"
    )

    send_telegram(msg)
    print(f"[{now}] 💓 Heartbeat terkirim ke Telegram")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Monitor training log dan kirim notifikasi Telegram"
    )
    parser.add_argument(
        "--from-start", action="store_true",
        help="Baca log dari awal (bukan dari akhir). Berguna untuk testing."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Kirim pesan test ke Telegram lalu keluar."
    )
    parser.add_argument(
        "--epochs", nargs="+", type=int, default=NOTIFY_EPOCHS,
        help=f"Epoch yang akan dikirim notifikasi (default: {NOTIFY_EPOCHS})"
    )
    parser.add_argument(
        "--interval", type=int, default=POLL_INTERVAL,
        help=f"Interval polling dalam detik (default: {POLL_INTERVAL})"
    )
    parser.add_argument(
        "--heartbeat", type=int, default=HEARTBEAT_INTERVAL,
        help=f"Interval heartbeat ke Telegram dalam menit (default: {HEARTBEAT_INTERVAL}). Set 0 untuk nonaktif."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  🔍 Mask R-CNN Training Monitor")
    print("=" * 60)
    print(f"  Notify epochs : {args.epochs}")
    print(f"  Poll interval : {args.interval}s")
    print(f"  Heartbeat     : {'setiap ' + str(args.heartbeat) + ' menit' if args.heartbeat > 0 else '❌ Nonaktif'}")
    print(f"  Telegram      : {'✅ Configured' if 'ISI_' not in TELEGRAM_BOT_TOKEN else '⚠️  NOT configured'}")
    print("=" * 60)

    # Test mode
    if args.test:
        hostname = socket.gethostname()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        test_msg = (
            f"🧪 *Test Notification*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🖥 Host: `{hostname}`\n"
            f"⏰ Waktu: {now}\n"
            f"✅ Monitor berjalan dengan baik!"
        )
        send_telegram(test_msg)
        return

    # Inisialisasi monitor
    monitors = []
    for logfile in LOG_FILES:
        mon = LogMonitor(logfile, args.epochs)
        if args.from_start and os.path.exists(logfile):
            mon.last_position = 0
            print(f"   ↪ Membaca dari awal: {logfile}")
        monitors.append(mon)

    print(f"\n🚀 Monitoring dimulai... (Ctrl+C untuk berhenti)\n")

    try:
        last_heartbeat = time.time()
        heartbeat_seconds = args.heartbeat * 60  # convert menit ke detik
        status_counter = 0

        while True:
            for mon in monitors:
                messages = mon.check()
                for msg in messages:
                    send_telegram(msg)
                    print(f"\n{'='*50}")
                    print(msg)
                    print(f"{'='*50}\n")

            # Heartbeat — kirim status ke Telegram setiap N menit
            if heartbeat_seconds > 0 and (time.time() - last_heartbeat) >= heartbeat_seconds:
                last_heartbeat = time.time()
                _send_heartbeat(monitors)

            # Print status ke console setiap 30 kali poll (~5 menit jika interval=10s)
            status_counter += 1
            if status_counter % 30 == 0:
                now = datetime.datetime.now().strftime("%H:%M:%S")
                statuses = " | ".join(m.status() for m in monitors)
                print(f"[{now}] 💓 {statuses}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n🛑 Monitor dihentikan oleh user.")
        statuses = "\n".join(f"   • {m.status()}" for m in monitors)
        print(f"Status terakhir:\n{statuses}")


if __name__ == "__main__":
    main()
