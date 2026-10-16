import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from apps.api.main import app

def run_tests():
    client = TestClient(app)
    results = []

    def check(name, method, url, **kwargs):
        try:
            if method == "GET":
                res = client.get(url, **kwargs)
            elif method == "POST":
                res = client.post(url, **kwargs)
            status_ok = res.status_code < 400
            print(f"[{'PASS' if status_ok else 'FAIL'}] {method} {url} -> Status {res.status_code}")
            results.append((name, url, res.status_code, "OK" if status_ok else "ERROR"))
            return res
        except Exception as e:
            print(f"[ERROR] {method} {url} -> Exception: {e}")
            results.append((name, url, 500, str(e)))
            return None

    print("\n" + "="*60)
    print("PROBANDO ENDPOINTS DE LA API (AI QUANT TRADING PLATFORM)")
    print("="*60)

    # 1. Sistema y Monitoreo
    check("Health Check", "GET", "/health")
    check("System Status", "GET", "/status")
    check("Market Session", "GET", "/session")
    check("OpenAPI JSON", "GET", "/openapi.json")

    # 2. Vistas HTML / Dashboard
    check("Dashboard HTML", "GET", "/dashboard")
    check("Portfolio HTML", "GET", "/portfolio")
    check("Orders HTML", "GET", "/orders")
    check("Scanner HTML", "GET", "/scanner")

    # 3. Telemetría y Datos Cuantitativos
    check("Telemetry", "GET", "/api/v1/telemetry")
    check("Trade History", "GET", "/api/v1/trade-history")
    check("Portfolio History", "GET", "/api/v1/portfolio-history")
    check("Market Bars AAPL", "GET", "/api/v1/market-data/bars/AAPL")

    # 4. Controles y Motor de Trading
    check("Scan de Oportunidades", "POST", "/api/v1/control/scan")
    check("Kill Switch Activar", "POST", "/api/v1/control/kill-switch", json={"reason": "Test de verificación"})
    
    # 5. Reanudar (prueba de seguridad: pass incorrecto -> debe dar 403)
    res_reject = client.post("/api/v1/control/resume", json={"passphrase": "CLAVE_INCORRECTA"})
    if res_reject.status_code == 403:
        print("[PASS] POST /api/v1/control/resume (Rechazo con clave incorrecta) -> Status 403 (Seguridad OK)")
        results.append(("Resume Clave Invalida", "/api/v1/control/resume", 403, "OK (Esperado)"))
    else:
        print(f"[FAIL] POST /api/v1/control/resume no devolvió 403: {res_reject.status_code}")
        results.append(("Resume Clave Invalida", "/api/v1/control/resume", res_reject.status_code, "FAIL"))

    # 6. Reanudar con clave correcta
    check("Kill Switch Reanudar", "POST", "/api/v1/control/resume", json={"passphrase": "CONFIRM_RESUME"})

    # 7. Auto-trade toggle
    check("Control Auto-Trade", "POST", "/api/v1/control/auto-trade", json={"enabled": False})

    print("="*60)
    total = len(results)
    passed = sum(1 for r in results if "OK" in r[3])
    print(f"RESUMEN: {passed}/{total} endpoints respondieron satisfactoriamente.")
    print("="*60 + "\n")

if __name__ == "__main__":
    with TestClient(app) as _:
        run_tests()
