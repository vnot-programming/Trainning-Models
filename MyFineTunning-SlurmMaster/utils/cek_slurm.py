#!/usr/bin/env python3

"""
# Untuk menjalankan dan melihat antarmuka bio-digital-nya:
python3 /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/cek_slurm.py

"""
import subprocess
import time
import sys
import re

# ============================================================================
# UI/UX Bio-Digital Minimalism (CLI Edition)
# ----------------------------------------------------------------------------
# Menggunakan ANSI Escape Codes untuk warna hangat/dingin (low sensory load), 
# fluid output (micro-animations), dan pemanfaatan negative space.
# ============================================================================

C_PRIMARY = "\033[94m"   # Soft Blue (Circadian friendly)
C_ACCENT = "\033[36m"    # Cyan
C_SUCCESS = "\033[92m"   # Soft Green
C_WARN = "\033[93m"      # Soft Yellow
C_DANGER = "\033[91m"    # Soft Red
C_DIM = "\033[2m"        # Dimmed text (Kurangi kontras)
C_RESET = "\033[0m"
C_BOLD = "\033[1m"

def smooth_print(text, delay=0.015):
    """Menghasilkan efek mengetik yang fluid untuk kenyamanan mata."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def divider():
    print(f"\n{C_DIM}─────────────────────────────────────────────────────────────{C_RESET}\n")

def get_output(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return ""

def parse_scontrol():
    output = get_output("scontrol show node")
    nodes = {}
    current_node = None
    
    for line in output.split('\n'):
        if line.startswith("NodeName="):
            parts = line.split()
            current_node = parts[0].split('=')[1]
            nodes[current_node] = {'name': current_node, 'gres': '0', 'alloc_gpu': 0, 'state': '', 'reason': '', 'cpu': 'N/A', 'ram': 'N/A', 'os': 'N/A', 'arch': 'N/A'}
            
            match_arch = re.search(r'Arch=([^\s]+)', line)
            if match_arch:
                nodes[current_node]['arch'] = match_arch.group(1)
        
        if current_node:
            if "Gres=" in line:
                match = re.search(r'Gres=.*?gpu:(\d+)', line)
                if match:
                    nodes[current_node]['gres'] = match.group(1)
            
            if "AllocTRES=" in line:
                match = re.search(r'gres/gpu=(\d+)', line)
                if match:
                    nodes[current_node]['alloc_gpu'] = int(match.group(1))
            
            if "State=" in line:
                match = re.search(r'State=([A-Z\+]+)', line)
                if match:
                    nodes[current_node]['state'] = match.group(1)
            
            if "Reason=" in line:
                # Mengambil teks reason
                match = re.search(r'Reason=(.*?)( \[|$)', line)
                if match:
                    nodes[current_node]['reason'] = match.group(1)
                    
            if "CPUTot=" in line:
                match = re.search(r'CPUTot=(\d+)', line)
                if match:
                    nodes[current_node]['cpu'] = match.group(1)
                    
            if "RealMemory=" in line:
                match = re.search(r'RealMemory=(\d+)', line)
                if match:
                    mb = int(match.group(1))
                    nodes[current_node]['ram'] = f"{mb / 1024:.0f} GB"
                    
            if "OS=" in line:
                match = re.search(r'OS=(.*?)( #|$)', line)
                if match:
                    nodes[current_node]['os'] = match.group(1).strip()
                    
    return nodes

def main():
    # Clear screen & Reset Cursor
    print(f"\033[2J\033[H", end="") 
    
    smooth_print(f"{C_PRIMARY}{C_BOLD}✧ SLURM MONITOR | VnoT ✧{C_RESET}")
    smooth_print(f"{C_DIM}Menyinkronkan metrik cluster dengan ritme sirkadian...{C_RESET}", 0.02)
    time.sleep(0.4)
    divider()

    # ---------------------------------------------------------
    # 1. GPU & NODE STATUS
    # ---------------------------------------------------------
    smooth_print(f"{C_ACCENT}❖ Status Komputasi Node & GPU{C_RESET}")
    nodes = parse_scontrol()
    
    total_gpus = 0
    total_alloc = 0
    total_avail = 0

    if not nodes:
        print(f"  {C_DIM}Tidak dapat terhubung ke Slurm Master.{C_RESET}")
    else:
        for name, data in nodes.items():
            state = data['state']
            total = int(data['gres']) if data['gres'].isdigit() else 0
            alloc = data['alloc_gpu']
            
            total_gpus += total
            total_alloc += alloc
            
            # Aksesibilitas: Indikator Teks + Warna (Multi-Visual)
            state_color = C_SUCCESS
            icon = "✅"
            if "DRAIN" in state or "DOWN" in state:
                state_color = C_DANGER
                icon = "⚠️"
            elif "MIXED" in state:
                state_color = C_WARN
                icon = "🔄"
                
            print(f"\n  {C_BOLD}Node: {name}{C_RESET}")
            
            # --- Bio-Digital Hardware Specs ---
            cpu = data['cpu']
            ram = data['ram']
            arch = data['arch']
            os_info = data['os']
            if len(os_info) > 20:
                os_info = os_info[:17] + "..."
                
            # Identitas GPU dari Cluster AI_KU_V100
            gpu_model = "NVIDIA V100"
            
            print(f"  {C_DIM}Spesifikasi: 🖥️ {cpu} Cores | 🧠 {ram} | 🎮 {total}x {gpu_model} | ⚙️ {arch} | 🐧 {os_info}{C_RESET}")
            
            reason_text = f"({data['reason']})" if data['reason'] else ""
            print(f"  Status    : {state_color}{icon} {state}{C_RESET} {C_DIM}{reason_text}{C_RESET}")
            
            # Visual Bar untuk Beban Kognitif Rendah
            bar = ""
            for i in range(total):
                if i < alloc:
                    bar += f"{C_DANGER}■ {C_RESET}" # Terpakai
                else:
                    bar += f"{C_SUCCESS}□ {C_RESET}" # Kosong
                    if "DRAIN" not in state and "DOWN" not in state:
                        total_avail += 1
            
            print(f"  Alokasi   : {bar} ({alloc}/{total} GPU Terpakai)")

    # Rangkuman GPU
    avail_color = C_SUCCESS if total_avail > 0 else C_DANGER
    avail_icon = "🌿" if total_avail > 0 else "🛑"
    print()
    smooth_print(f"  {C_BOLD}Kapasitas Aktif:{C_RESET} {avail_color}{avail_icon} {total_avail} GPU Siap Digunakan{C_RESET} {C_DIM}(Dari total {total_gpus} terpasang){C_RESET}")
    
    divider()

    # ---------------------------------------------------------
    # 2. DISK QUOTA /DATA
    # ---------------------------------------------------------
    smooth_print(f"{C_ACCENT}❖ Kapasitas Penyimpanan Ekologis (/data){C_RESET}")
    df_output = get_output("df -h /data | tail -n 1")
    
    if df_output:
        parts = df_output.split()
        if len(parts) >= 5:
            total = parts[1]
            used = parts[2]
            avail = parts[3]
            use_pct = parts[4]
            
            # Progress bar untuk disk (Visual)
            pct = int(use_pct.replace('%', ''))
            filled_blocks = int(pct / 5)
            empty_blocks = 20 - filled_blocks
            disk_bar = f"{C_WARN}" + "■" * filled_blocks + f"{C_DIM}" + "□" * empty_blocks + f"{C_RESET}"
            
            print(f"\n  Total Quota : {C_BOLD}{total}{C_RESET}")
            print(f"  Terpakai    : {used} [{disk_bar}] {use_pct}")
            
            color_avail = C_SUCCESS if pct < 90 else C_DANGER
            print(f"  Tersedia    : {color_avail}{C_BOLD}{avail}{C_RESET} {C_DIM}(Ruang bernapas untuk dataset){C_RESET}")
    else:
        print(f"\n  {C_DIM}Tidak dapat membaca metrik disk.{C_RESET}")

    divider()
    smooth_print(f"{C_DIM}Penyelarasan sistem selesai. Tetap jaga ritme kerja yang sehat.{C_RESET}", 0.02)
    print()

if __name__ == '__main__':
    main()
