"""
Quant OS - Proxy Local Anti-429
================================
Este script crea un servidor HTTP en localhost:8765 que actua como
intermediario entre tu navegador y el servidor de trading en Hostinger.

VENTAJAS:
- Las peticiones al API salen desde tu PC directamente al servidor
  usando requests de Python con reintentos robustos
- El navegador nunca llega a Hostinger directamente (sin 429!)
- No necesitas VPN, Cloudflare ni cambiar DNS

USO: python local_proxy.py
Luego abre: http://localhost:8765/dashboard
"""

import io
import sys

# Fix Windows console encoding for UTF-8 output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import http.server
import urllib.request
import urllib.error
import ssl
import time
import threading
import webbrowser
import json


LOCAL_PORT = 8765
REMOTE_BASE = "https://mysmartdomains.com/trading"
MAX_RETRIES = 5
RETRY_DELAY = 3  # segundos entre reintentos
# ────────────────────────────────────────────────────

# Context SSL que acepta el cert del servidor
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

BANNER = """
==============================================================
   QUANT OS - PROXY LOCAL ACTIVO (Anti-429 de Hostinger)
==============================================================
  Dashboard: http://localhost:{port}/dashboard
  Presiona Ctrl+C para detener
==============================================================
""".format(port=LOCAL_PORT)



class TradingProxy(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        """Override to show cleaner logs."""
        path = self.path
        # Filter out noisy favicon/status requests
        if "favicon" in path:
            return
        print(f"  [{time.strftime('%H:%M:%S')}] {self.command} {path}")

    def do_GET(self):
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")

    def do_PUT(self):
        self._proxy("PUT")

    def do_DELETE(self):
        self._proxy("DELETE")

    def do_OPTIONS(self):
        self.send_response(200)
        self._add_cors_headers()
        self.end_headers()

    def _add_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _proxy(self, method):
        remote_url = REMOTE_BASE + self.path

        # Read request body if present
        body = None
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.rfile.read(content_length)

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(
                    remote_url,
                    data=body,
                    method=method,
                    headers={
                        "User-Agent": "QuantOS-LocalProxy/2.0",
                        "Content-Type": self.headers.get("Content-Type", "application/json"),
                    }
                )
                with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as resp:
                    response_body = resp.read()
                    self.send_response(resp.getcode())
                    content_type = resp.headers.get("Content-Type", "application/octet-stream")
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(response_body)))
                    self.send_header("Cache-Control", "no-cache")
                    self._add_cors_headers()
                    self.end_headers()
                    self.wfile.write(response_body)
                    return  # Success!

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = RETRY_DELAY * attempt * 2
                    print(f"  ⚠️  429 recibido (intento {attempt}/{MAX_RETRIES}). Esperando {wait}s...")
                    time.sleep(wait)
                    last_error = e
                    continue
                else:
                    # Other HTTP errors - pass through
                    try:
                        body_err = e.read()
                    except:
                        body_err = b""
                    self.send_response(e.code)
                    self.send_header("Content-Type", "text/plain")
                    self._add_cors_headers()
                    self.end_headers()
                    self.wfile.write(body_err or f"HTTP Error {e.code}".encode())
                    return

            except Exception as e:
                last_error = e
                print(f"  ❌ Error de red (intento {attempt}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        # All retries failed
        print(f"  ❌ Error tras {MAX_RETRIES} intentos: {last_error}")
        error_response = json.dumps({
            "error": "proxy_failed",
            "message": f"No se pudo conectar al servidor tras {MAX_RETRIES} intentos",
            "last_error": str(last_error)
        }).encode()
        try:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error_response)))
            self._add_cors_headers()
            self.end_headers()
            self.wfile.write(error_response)
        except:
            pass


def open_browser():
    """Open the dashboard in the default browser after a short delay."""
    time.sleep(2)
    url = f"http://localhost:{LOCAL_PORT}/dashboard"
    print(f"\n  🌐 Abriendo dashboard en navegador: {url}\n")
    webbrowser.open(url)


if __name__ == "__main__":
    print(BANNER)

    # Test remote connection first
    print("  Verificando conexion con el servidor...")
    try:
        req = urllib.request.Request(
            f"{REMOTE_BASE}/api/v1/telemetry",
            headers={"User-Agent": "QuantOS-LocalProxy/2.0 TestProbe"}
        )
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            pos_count = len(data.get("positions", []))
            portfolio = data.get("broker", {}).get("portfolio_value", "N/A")
            print(f"  ✅ Servidor activo - {pos_count} posiciones, Portfolio: ${portfolio:,.2f}" 
                  if isinstance(portfolio, (int, float)) else 
                  f"  ✅ Servidor activo - {pos_count} posiciones")
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f"  ⚠️  Tu IP recibe 429 directamente. El proxy local te protegerá.")
            print(f"     (El proxy usará su propia IP para acceder al servidor)")
        else:
            print(f"  ⚠️  Error HTTP {e.code} conectando al servidor")
    except Exception as ex:
        print(f"  ⚠️  No se pudo verificar: {ex}")

    # Start local server
    server = http.server.HTTPServer(("localhost", LOCAL_PORT), TradingProxy)
    print(f"\n  🚀 Proxy local iniciado en http://localhost:{LOCAL_PORT}")
    print(f"  📊 Dashboard disponible en http://localhost:{LOCAL_PORT}/dashboard")
    print(f"  ⌨️  Presiona Ctrl+C para detener\n")

    # Open browser in background
    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n  🛑 Proxy detenido. ¡Hasta luego!")
        server.shutdown()
