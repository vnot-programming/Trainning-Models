import os
import subprocess
import sys

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def run_command(command):
    try:
        print(f"\nExcecuting: {command}")
        result = subprocess.run(command, shell=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"\nError executing command: {e}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

def show_menu():
    print("="*50)
    print("      SLURM COMMAND HELPER - AI_KU_V100")
    print("="*50)
    print("1. [sinfo]    Cek Status Node & Partisi")
    print("2. [squeue]   Cek Antrian Job (User)")
    print("3. [squeue]   Cek Semua Antrian Job")
    print("4. [scancel]  Batalkan Job")
    print("5. [sacctmgr] Cek Kuota & QOS")
    print("6. [srun]     Masuk ke Sesi Interaktif (GPU)")
    print("7. [sbatch]   Submit Script Batch")
    print("8. [Template] Buat Script Batch Baru")
    print("9. [Info]     Lihat Daftar Directive #SBATCH")
    print("0. Keluar")
    print("="*50)

def show_directives():
    directives = [
        ("#SBATCH --job-name=test", "Menentukan nama job"),
        ("#SBATCH --output=output%j.out", "Log standar output dan error (%j = Job ID)"),
        ("#SBATCH --ntasks=4", "Jumlah task yang akan dijalankan"),
        ("#SBATCH --time=00:01:00", "Batas waktu (jam:menit:detik)"),
        ("#SBATCH --cpus-per-task=1", "Jumlah core CPU per task"),
        ("#SBATCH --gres=gpu:1", "Jumlah resource GPU"),
        ("#SBATCH --mem=16GB", "Jumlah memori RAM per node"),
        ("#SBATCH --partition=gpu", "Nama partisi (contoh: gpu)"),
        ("#SBATCH --test-only", "Validasi script tanpa benar-benar menjalankannya")
    ]
    print("\n--- Daftar Directive #SBATCH ---")
    print(f"{'Directive':<30} | {'Deskripsi'}")
    print("-" * 60)
    for d, desc in directives:
        print(f"{d:<30} | {desc}")

def create_template():
    filename = input("Masukkan nama file (contoh: my_job.sh): ")
    if not filename.endswith('.sh'):
        filename += '.sh'
    
    job_name = input("Nama Job (default: test): ") or "test"
    cpus = input("Jumlah CPU Cores (max 8, default: 4): ") or "4"
    mem = input("Jumlah RAM (max 64GB, default: 16GB): ") or "16GB"
    time = input("Batas Waktu (HH:MM:SS, default: 01:00:00): ") or "01:00:00"
    
    content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=output_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:1
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --partition=gpu

# Tulis perintah Anda di bawah ini
echo "Job dimulai pada: $(date)"
echo "Berjalan di node: $(hostname)"
nvidia-smi
python3 your_script.py
"""
    with open(filename, 'w') as f:
        f.write(content)
    print(f"\nTemplate berhasil dibuat: {filename}")
    print(f"Gunakan menu nomor 7 untuk menjalankan script ini.")

def main():
    while True:
        show_menu()
        choice = input("Pilih menu (0-8): ")
        
        if choice == '1':
            run_command("sinfo")
        elif choice == '2':
            user = os.getenv('USER')
            run_command(f"squeue -u {user}")
        elif choice == '3':
            run_command("squeue")
        elif choice == '4':
            job_id = input("Masukkan Job ID yang ingin dibatalkan: ")
            if job_id:
                run_command(f"scancel {job_id}")
        elif choice == '5':
            print("\n--- Ringkasan QOS ---")
            run_command("sacctmgr show qos format=name%15,MaxTRESPU%35,MaxWall,MaxTRES,MaxJobsPU")
        elif choice == '6':
            print("\nMembuka sesi interaktif (4 Core, 1 GPU, Partition: gpu)...")
            # Note: srun --pty bash will take over the terminal
            os.system("srun -c 4 -p gpu --gres=gpu:1 --pty bash")
        elif choice == '7':
            filename = input("Masukkan nama file script (.sh): ")
            if os.path.exists(filename):
                run_command(f"sbatch {filename}")
            else:
                print(f"File {filename} tidak ditemukan.")
        elif choice == '8':
            create_template()
        elif choice == '9':
            show_directives()
        elif choice == '0':
            print("Keluar...")
            break
        else:
            print("Pilihan tidak valid.")
        
        input("\nTekan Enter untuk kembali ke menu...")
        clear_screen()

if __name__ == "__main__":
    main()
