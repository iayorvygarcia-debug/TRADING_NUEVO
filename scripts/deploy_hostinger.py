"""
Script de Despliegue Automatizado a Hostinger (mysmartdomains.com)
Sincroniza el codigo fuente por SFTP y reinicia el demonio UDS.
"""
import os
from pathlib import Path
import paramiko
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("SSH_HOST", "82.197.80.115")
port = int(os.getenv("SSH_PORT", 65002))
user = os.getenv("SSH_USER", "u114856156")
pwd = os.getenv("SSH_PASSWORD", "")

if not pwd:
    print("[-] Error: SSH_PASSWORD no esta configurada en .env")
    exit(1)

print(f"[+] Conectando a {host}:{port} como {user}...")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username=user, password=pwd, timeout=15)
sftp = client.open_sftp()

remote_base = "/home/u114856156/trading_platform"
dirs_to_sync = ["apps", "core", "services", "strategies"]

uploaded = 0
for d in dirs_to_sync:
    for root, _, files in os.walk(d):
        if "__pycache__" in root or ".pytest_cache" in root or ".ruff_cache" in root:
            continue
        rel_root = Path(root).as_posix()
        rem_dir = f"{remote_base}/{rel_root}"
        try:
            sftp.stat(rem_dir)
        except IOError:
            sftp.mkdir(rem_dir)
        for f in files:
            if f.endswith(".pyc") or f.endswith(".pyo"):
                continue
            local_p = Path(root) / f
            rem_p = f"{remote_base}/{local_p.as_posix()}"
            sftp.put(str(local_p), rem_p)
            uploaded += 1

print(f"[+] Total archivos sincronizados: {uploaded}")
sftp.close()

# Reiniciar daemon
print("[+] Reiniciando AI Quant Platform en Hostinger...")
cmd = (
    "pkill -9 -f apps.api.main; "
    "pkill -9 -f uvicorn; "
    "rm -f /home/u114856156/trading_platform/uvicorn.sock; "
    "/home/u114856156/trading_platform/.venv/bin/python3 /home/u114856156/trading_platform/start_daemon.py"
)
stdin, stdout, stderr = client.exec_command(cmd)
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    print("Stderr:", err)

client.close()
print("[OK] Despliegue completado con exito. Visita https://mysmartdomains.com/trading/dashboard")
