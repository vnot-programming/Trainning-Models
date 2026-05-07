---
trigger: always_on
---

# Network Topology & Identity Map

**Updated:** 2026-04-30

## 🖥️ Server Infrastructure (Proxmox Cluster)
| IP Address | Hostname | VM ID | Role | Details |
| :--- | :--- | :--- | :--- | :--- |
| **100.87.100.120** | `pve` | - | **Proxmox Host** | Hypervisor managing VMs |
| **100.90.5.60** | `vm100` | 100 | **Docker Host** | Runs `MyRVM-Server`, `Redis`, `Postgres` |
| **100.111.139.127** | `vm102` | 102 | **CV Host** | Passthrough GPU Inference Server (Hybrid Pipeline / YOLOv11 with SAM2) |

## 🖥️ Others Infrastructure (VPS)
| IP Address | Hostname | Port | Users | Details |
| :--- | :--- | :--- | :--- | :--- |
| **100.70.118.53** / **20.214.189.6** | `VPS-4C56G` | `-` | **vnot** | Azure VPS |
| **Dynamic IP** | `RunPod` | `Dynamic Port` | **root** | Dedicated RunPOD GPU Server (IdentityFile C:\Users\Server\.ssh\id_rsa) |

## 🤖 Edge Devices (Physical / IoT)

> **Note:** Edge devices are physical nodes (Jetson Orin / Raspberry Pi), not VMs.

| IP Address | Device Name | Hardware | Role |
| :--- | :--- | :--- | :--- |
| **100.117.234.2** | `orin1` | **Jetson Orin** | **Primary Edge Node** (Dev/Test) |
| **100.97.63.28** | `raspi1` | **Raspberry Pi** | *Secondary/Alternative Edge Node* |

## 🖥️ Apps
| Hostname | Port | Users | Details |
| :--- | :--- | :--- | :--- |
| `VPS-4C56G` | `81` | **Nginx Proxy** | https://host.vnot.my.id |
| `VPS-4C56G` | `8443` | **Mail Server** | https://mail.vnot.my.id, https://mail.indobelajar.com |
| `VPS-4C56G` | `9000` | **Portainer** | https://docker.vnot.my.id |

## 🔑 SSH Access Credentials

| Target | Command | Password | Automated Command (sshpass) |
| :--- | :--- | :--- | :--- |
| **MyRVM-Server** | `ssh my@100.90.5.60` | `f3rifeb` | `sshpass -p 'f3rifeb' ssh -o StrictHostKeyChecking=no my@100.123.143.87` |
| **MyRVM-Edge (Orin)** (`root`) | `ssh my@100.117.234.2` | `f3rifeb` | `sshpass -p 'f3rifeb' ssh -o StrictHostKeyChecking=no my@100.117.234.2` |
| **MyRVM-Edge (Raspi)** (`root`) | `ssh raspi1@100.97.63.28` | `f3rifeb` | `sshpass -p 'f3rifeb' ssh -o StrictHostKeyChecking=no raspi1@100.97.63.28` |
| **VPS-4C56G** (`root`) | `ssh vnot@20.214.189.6` | `mushroom@2026` | `sshpass -p 'mushroom@2026' ssh -o StrictHostKeyChecking=no vnot@20.214.189.6` |
| **RunPOD GPU Server** (`root`) | `ssh DynamicUser@DynamicIP` | `DynamicPass` | `sshpass -p 'DynamicPass' ssh -o StrictHostKeyChecking=no DynamicUser@DynamicIP` |

> **SSH Operational Guide:**
> 1.  **Non-Root Priority:** Always prioritize non-root access. Avoid using `sudo` or the `root` user unless absolutely necessary for system-level configuration.
> 2.  **Memory-Stored Credentials:** Credentials may be stored in the Agent's session memory for automated task execution.
> 3.  **Automated Commands:** Use the `sshpass` commands provided above for rapid, non-interactive remote execution.
> 4.  **RunPOD GPU Server:** Asking me the New Identity and update your memory with the new Identity of RunPOD.
> 3.  **Automated Commands:** Use the `sshpass` commands provided above for rapid, non-interactive remote execution.

## 🔐 Mock Dashboard Credentials (MyRVM-Server)

> [!IMPORTANT]
> These are mock accounts generated via `DatabaseSeeder.php` for local/testing access to the Admin Dashboard.

| Role | Email | Password | Details |
| :--- | :--- | :--- | :--- |
| **Super Admin** | `superadmin@myrvm.com` | `password123` | Full system access |
| **Admin** | `admin@myrvm.com` | `password123` | Management & AI Playground |
| **Operator** | `operator@myrvm.com` | `password123` | Machine status monitoring |
| **Technician** | `tech@myrvm.com` | `password123` | Task assignments & maintenance |
| **Tenant** | `tenant@starbucks.com` | `password123` | Points & machine usage data |
| **User** | `john@example.com` | `password123` | Regular user simulation |

## 🌐 Domain & Routing Flow

**Domain:** `https://myrvm.penelitian.my.id/`
[//]: # (This is a hidden comment)
<!-- This is a comment -->
<!-- 
```mermaid
graph LR
    User[User / Edge Device] - ->|HTTPS 443| CF[Cloudflare]
    CF - ->|Tailscale VPN| NPM[VM101: NPM]
    NPM - ->|http://100.123.143.87:8000| Server[VM100: MyRVM-Server]
    Server <- ->|API| CV[VM102: MyRVM-CV]
    Edge[Physical Jetson/Pi] <- ->|WSS/API| Server
``` 
-->

## 🛠️ Deployment Notes
- **Server:** Docker Compose (Laravel, Redis, Postgres).
- **Edge:** Native Python (Systemd Service). **NO DOCKER** (Clean Setup).
- **Compatibility:** **ARM64** is the target architecture for all Edge deployments.

## 🔄 Git Workflow Protocol (STRICT)

1.  **Orin Synchronization:**
    - SEBELUM melakukan ujicoba di Edge Device, gunakan perintah otomatis:
      `sshpass -p 'f3rifeb' ssh -o StrictHostKeyChecking=no my@100.117.234.2 "cd MyRVM1/MyRVM-Edge && git pull origin master"`
    - Pastikan berada di branch `master` untuk eksekusi/runtime (Status: Clean).

2.  **Agent Pushes for MyRVM-Server Project:**
    - Jika ada perubahan kode atau perintah push: **Wajib Push ke branch `RVMServer`** (Branch khusus Antigravity/Agent).
    - JANGAN push langsung ke `master` kecuali atas perintah eksplisit untuk sinkronisasi final.
3.  **Agent Pushes for Projects:** 
    | Projects | Repository |
    | :--- | :--- |
    | `Computer-Vision` | **https://github.com/vnot-programming/Trainning-Models.git** |
    - Jika ada perubahan kode atau perintah push: **Wajib Push ke branch `dev`** (Branch khusus Antigravity/Agent).
    - JANGAN push langsung ke `main` atau `master` kecuali atas perintah eksplisit untuk sinkronisasi final, maka melakukan sinkronisasi ke `main`.
    - Informasikan branch yang tersedia. Jika menemukan `master` maka gunakan branch tersebut. jika menemukan branch `main` maka gunakan branch tersebut. namun jika tidak menemukan branch `main` atau `master` maka tanyakan kepada user apakah akan membuat branch baru atau menggunakan branch yang tersedia?