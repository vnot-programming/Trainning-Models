#!/usr/bin/env python3
import subprocess
import os

def format_time(time_str):
    if time_str == 'INVALID' or time_str == 'N/A' or time_str == '0:00':
        return '0m'
    
    days = 0
    if '-' in time_str:
        parts = time_str.split('-')
        days = int(parts[0])
        time_str = parts[1]
    
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h = 0
        m, s = int(parts[0]), int(parts[1])
    else:
        return time_str

    res = []
    if days > 0:
        res.append(f"{days}d")
    if h > 0:
        res.append(f"{h}h")
    if m > 0:
        res.append(f"{m}m")
    if s > 0 and days == 0 and h == 0:
        res.append(f"{s}s")
        
    if not res:
        return "0s"
    return " ".join(res[:2]) # return top 2 most significant parts

def get_squeue_data(user=None):
    cmd = ["squeue", "-h", "-o", "%i|%u|%M|%j|%t|%R"]
    if user:
        cmd.extend(["-u", user])
    
    try:
        output = subprocess.check_output(cmd, universal_newlines=True)
    except Exception:
        return []

    lines = output.strip().split('\n')
    data = []
    for line in lines:
        if not line.strip(): continue
        parts = line.split('|')
        if len(parts) >= 6:
            data.append(parts)
    return data

def print_table(data):
    if not data:
        print("  ⚠️ Anda tidak memiliki antrean yang sedang berjalan/pending.")
        print("  " + "-"*75)
        return

    # Header
    print(f"  {'JOBID':<10} {'USER':<14} {'TIME':<12} {'NAME':<18} {'STATE':<10} {'NODES'}")
    print("  " + "-"*75)
    for row in data:
        jobid, user, time_str, name, state, nodes = row
        f_time = format_time(time_str)
        
        padded_state = f"{state:<8}"
        if state == 'R':
            c_padded_state = f"\033[92m{padded_state}\033[0m" # Green
        elif state == 'PD':
            c_padded_state = f"\033[93m{padded_state}\033[0m" # Yellow
        elif state == 'CG':
            c_padded_state = f"\033[91m{padded_state}\033[0m" # Red
        else:
            c_padded_state = padded_state

        # Truncate name if too long
        if len(name) > 17:
            name = name[:14] + "..."
            
        print(f"  {jobid:<10} {user:<14} {f_time:<12} {name:<18} {c_padded_state} {nodes}")
    print("  " + "-"*75)

if __name__ == "__main__":
    current_user = os.environ.get("USER", "")
    
    print("╭─────────────────────────────────────────────────────────────────────────╮")
    print("│                       DAFTAR ANTREAN SLURM GLOBAL                       │")
    print("╰─────────────────────────────────────────────────────────────────────────╯")
    all_data = get_squeue_data()
    print_table(all_data)
    
    print("\n╭─────────────────────────────────────────────────────────────────────────╮")
    print("│                     DAFTAR ANTREAN SLURM ANDA (USER)                    │")
    print("╰─────────────────────────────────────────────────────────────────────────╯")
    if current_user:
        user_data = get_squeue_data(current_user)
        print_table(user_data)
    else:
        print("  ⚠️ User tidak terdeteksi.")
        print("  " + "-"*75)
