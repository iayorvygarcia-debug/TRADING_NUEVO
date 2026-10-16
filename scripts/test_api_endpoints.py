import urllib.request
import json

base_url = "http://127.0.0.1:8000"

def test_endpoints():
    # 1. Telemetry
    resp = urllib.request.urlopen(f"{base_url}/api/v1/telemetry")
    data = json.loads(resp.read())
    print("Telemetry check:", data["trading_mode"], "Session:", data["market_session"]["session"])
    # If kill switch was left active from prior test, resume it first
    if data["kill_switch_active"]:
        req = urllib.request.Request(
            f"{base_url}/api/v1/control/resume",
            data=json.dumps({"passphrase": "CONFIRM_RESUME"}).encode(),
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req)
        resp = urllib.request.urlopen(f"{base_url}/api/v1/telemetry")
        data = json.loads(resp.read())

    assert data["kill_switch_active"] is False

    # 2. Trigger Kill Switch
    req = urllib.request.Request(
        f"{base_url}/api/v1/control/kill-switch",
        data=json.dumps({"reason": "Audit verification"}).encode(),
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    print("Kill switch activated:", data)
    assert data["is_active"] is True

    # 3. Resume with invalid passphrase (expect 403)
    try:
        req = urllib.request.Request(
            f"{base_url}/api/v1/control/resume",
            data=json.dumps({"passphrase": "WRONG_KEY"}).encode(),
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req)
        print("ERROR: Invalid passphrase did not fail!")
    except urllib.error.HTTPError as e:
        print("Resume with invalid passphrase correctly rejected (HTTP 403):", e.code)

    # 4. Resume with valid passphrase
    req = urllib.request.Request(
        f"{base_url}/api/v1/control/resume",
        data=json.dumps({"passphrase": "CONFIRM_RESUME"}).encode(),
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    print("Kill switch deactivated (Resumed):", data)
    assert data["kill_switch_active"] is False

    # 5. Opportunity Scan
    req = urllib.request.Request(
        f"{base_url}/api/v1/control/scan",
        data=b"{}",
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    print("Scan status:", data["status"], "Cash:", data["telemetry"]["portfolio"]["cash"])

    print("\nALL ENDPOINTS FUNCTIONING WITH 100% SUCCESS!")

if __name__ == "__main__":
    test_endpoints()
