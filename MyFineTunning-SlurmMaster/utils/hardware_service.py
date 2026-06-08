import psutil
import platform
import json
import subprocess
from datetime import datetime

class HardwareService:
    """
    Layanan untuk mengambil spesifikasi hardware sistem menggunakan psutil dan perintah sistem.
    """

    def get_size(self, bytes, suffix="B"):
        """
        Mengonversi bytes ke format yang mudah dibaca (e.g. GB, MB).
        """
        factor = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if bytes < factor:
                return f"{bytes:.2f}{unit}{suffix}"
            bytes /= factor

    def get_system_info(self):
        """
        Mengambil informasi sistem operasi.
        """
        uname = platform.uname()
        return {
            "System": uname.system,
            "Node Name": uname.node,
            "Release": uname.release,
            "Version": uname.version,
            "Machine": uname.machine,
            "Processor": uname.processor
        }

    def get_cpu_info(self):
        """
        Mengambil informasi CPU.
        """
        cpufreq = psutil.cpu_freq()
        return {
            "Physical cores": psutil.cpu_count(logical=False),
            "Total cores": psutil.cpu_count(logical=True),
            "Max Frequency": f"{cpufreq.max:.2f}Mhz" if cpufreq else "N/A",
            "Min Frequency": f"{cpufreq.min:.2f}Mhz" if cpufreq else "N/A",
            "Current Frequency": f"{cpufreq.current:.2f}Mhz" if cpufreq else "N/A",
            "Total CPU Usage": f"{psutil.cpu_percent()}%"
        }

    def get_memory_info(self):
        """
        Mengambil informasi Memori (RAM).
        """
        svmem = psutil.virtual_memory()
        return {
            "Total": self.get_size(svmem.total),
            "Available": self.get_size(svmem.available),
            "Used": self.get_size(svmem.used),
            "Percentage": f"{svmem.percent}%"
        }

    def get_disk_info(self):
        """
        Mengambil informasi Disk.
        """
        partitions = psutil.disk_partitions()
        disk_data = []
        for partition in partitions:
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                disk_data.append({
                    "Device": partition.device,
                    "Mountpoint": partition.mountpoint,
                    "File System Type": partition.fstype,
                    "Total Size": self.get_size(partition_usage.total),
                    "Used": self.get_size(partition_usage.used),
                    "Free": self.get_size(partition_usage.free),
                    "Percentage": f"{partition_usage.percent}%"
                })
            except PermissionError:
                continue
        return disk_data

    def get_gpu_info(self):
        """
        Mengambil informasi GPU/VGA menggunakan perintah sistem.
        """
        gpu_info = "Not Found"
        # Coba beberapa lokasi lspci
        lspci_paths = ['lspci', '/usr/sbin/lspci', '/sbin/lspci']
        
        for path in lspci_paths:
            try:
                result = subprocess.run([path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if "VGA" in line or "3D" in line or "NVIDIA" in line:
                            gpu_info = line.split(": ")[-1].strip()
                            break
                    if gpu_info != "Not Found":
                        break
            except Exception:
                continue
        
        # Cek apakah ada nvidia-smi
        has_nvidia = False
        try:
            subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            has_nvidia = True
        except Exception:
            pass

        return {
            "GPU/VGA Controller": gpu_info,
            "NVIDIA Driver Installed": has_nvidia
        }

    def run(self):
        """
        Menjalankan layanan dan mengembalikan laporan lengkap.
        """
        report = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "System Info": self.get_system_info(),
            "CPU Info": self.get_cpu_info(),
            "Memory Info": self.get_memory_info(),
            "Disk Info": self.get_disk_info(),
            "GPU Info": self.get_gpu_info()
        }
        return report

if __name__ == "__main__":
    service = HardwareService()
    specs = service.run()
    print(json.dumps(specs, indent=4))
