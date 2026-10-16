<?php
/**
 * AI Quant Trading Platform — Reverse Proxy Gateway (UDS Edition)
 * Versión optimizada con Micro-Cache y entrega estática para inmunidad contra 429 de LiteSpeed.
 */

$ip = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
$uri = $_SERVER['REQUEST_URI'] ?? '';
$t = microtime(true);
@file_put_contents('/home/u114856156/trading_platform/requests.log', sprintf("%.3f | %s | %s\n", $t, $ip, $uri), FILE_APPEND);

$sock_file = '/home/u114856156/trading_platform/uvicorn.sock';
$daemon_script = '/home/u114856156/trading_platform/start_daemon.py';
$python_bin = '/home/u114856156/trading_platform/.venv/bin/python3';
$static_dashboard = '/home/u114856156/trading_platform/apps/api/static/dashboard.html';
$cache_file = '/tmp/trading_telemetry_cache.json';

if (!function_exists('getallheaders')) {
    function getallheaders() {
        $headers = [];
        foreach ($_SERVER as $name => $value) {
            if (substr($name, 0, 5) == 'HTTP_') {
                $headers[str_replace(' ', '-', ucwords(strtolower(str_replace('_', ' ', substr($name, 5)))))] = $value;
            } elseif ($name == 'CONTENT_TYPE') {
                $headers['Content-Type'] = $value;
            } elseif ($name == 'CONTENT_LENGTH') {
                $headers['Content-Length'] = $value;
            }
        }
        return $headers;
    }
}

function wake_daemon_in_background($python_bin, $daemon_script) {
    $lock_file = '/tmp/trading_daemon_starting.lock';
    if (file_exists($lock_file) && (time() - filemtime($lock_file) < 30)) {
        return;
    }
    @touch($lock_file);
    @exec("nohup $python_bin $daemon_script > /dev/null 2>&1 &");
}

function show_starting_page($is_api = false) {
    http_response_code(503);
    if ($is_api) {
        header("Content-Type: application/json; charset=utf-8");
        echo json_encode([
            "status" => "connecting",
            "message" => "El motor de trading se esta sincronizando. Reconectando en breve...",
            "retry_after" => 2
        ]);
        exit;
    }
    header("Content-Type: text/html; charset=utf-8");
    echo '<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant OS — Conectando</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-[#0d1117] text-white flex items-center justify-center min-h-screen font-sans">
    <div class="text-center p-8 max-w-md bg-[#161b22] border border-[#30363d] rounded-2xl shadow-2xl">
        <div class="text-4xl mb-4 animate-bounce">⚡</div>
        <h1 class="text-xl font-bold mb-2 text-[#00e676]">Sincronizando Motor Cuántico...</h1>
        <p class="text-sm text-[#8b9bb4] mb-6">El servicio se está reconectando automáticamente. Reintentando en breve...</p>
        <div class="w-full bg-[#21262d] h-2 rounded-full overflow-hidden">
            <div class="bg-[#00e676] h-full w-2/3 animate-pulse"></div>
        </div>
    </div>
    <script>
        let checking = false;
        setInterval(async () => {
            if (checking) return;
            checking = true;
            try {
                const res = await fetch("/trading/status");
                if (res.ok) {
                    location.reload();
                }
            } catch(e) {}
            finally {
                checking = false;
            }
        }, 6000);
    </script>
</body>
</html>';
    exit;
}

$request_uri = $_SERVER['REQUEST_URI'] ?? '/trading/';
$parsed = parse_url($request_uri);
$path = $parsed['path'] ?? '/';
$query = isset($parsed['query']) ? '?' . $parsed['query'] : '';

$clean_path = preg_replace('#^/trading/#', '/', $path);
if ($clean_path === '/trading' || empty($clean_path)) {
    $clean_path = '/dashboard';
}

$is_api      = (strpos($clean_path, '/api/') === 0);
$is_telemetry = ($clean_path === '/api/v1/telemetry' && $_SERVER['REQUEST_METHOD'] === 'GET');
$is_sse       = ($clean_path === '/api/v1/stream'    && $_SERVER['REQUEST_METHOD'] === 'GET');

// OPTIMIZACIÓN 1: Entrega estática ultrarrápida del Dashboard (0ms PHP, sin cURL, sin rate limit)
if (($clean_path === '/dashboard' || $clean_path === '/') && $_SERVER['REQUEST_METHOD'] === 'GET' && file_exists($static_dashboard)) {
    header('Content-Type: text/html; charset=utf-8');
    header('Cache-Control: public, max-age=30');
    readfile($static_dashboard);
    exit;
}

// OPTIMIZACIÓN 2: Micro-Cache de 3s para Telemetría (previene sobrecarga y rechaza saturaciones)
if ($is_telemetry && file_exists($cache_file) && (time() - filemtime($cache_file) < 3)) {
    header('Content-Type: application/json; charset=utf-8');
    header('Cache-Control: no-cache');
    header('X-Quant-Cache: HIT');
    readfile($cache_file);
    exit;
}

$target_url = 'http://localhost' . $clean_path . $query;

if (!file_exists($sock_file)) {
    wake_daemon_in_background($python_bin, $daemon_script);
    show_starting_page($is_api);
}

// ── Función proxy estándar (requests normales) ──────────────────────────────
function proxy_request($target_url, $sock_file) {
    $ch = curl_init($target_url);
    curl_setopt($ch, CURLOPT_UNIX_SOCKET_PATH, $sock_file);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HEADER, true);
    curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);
    curl_setopt($ch, CURLOPT_CUSTOMREQUEST, $_SERVER['REQUEST_METHOD']);
    curl_setopt($ch, CURLOPT_TIMEOUT, 20);

    $headers = [];
    foreach (getallheaders() as $name => $value) {
        if (strtolower($name) !== 'host') {
            $headers[] = "$name: $value";
        }
    }
    $headers[] = "X-Forwarded-Prefix: /trading";
    curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);

    if (in_array($_SERVER['REQUEST_METHOD'], ['POST', 'PUT', 'PATCH', 'DELETE'])) {
        $body = file_get_contents('php://input');
        curl_setopt($ch, CURLOPT_POSTFIELDS, $body);
    }

    $raw = curl_exec($ch);
    $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $size = curl_getinfo($ch, CURLINFO_HEADER_SIZE);
    curl_close($ch);

    return [$raw, $code, $size];
}

// ── Proxy SSE en modo streaming (sin buffer — imprescindible para EventSource) ──
function sse_proxy_stream($target_url, $sock_file) {
    // Desactivar todos los buffers PHP para streaming inmediato
    while (ob_get_level()) ob_end_clean();
    set_time_limit(0);
    ignore_user_abort(false);

    header('Content-Type: text/event-stream; charset=utf-8');
    header('Cache-Control: no-cache');
    header('X-Accel-Buffering: no');   // Desactiva buffer de Nginx/LiteSpeed
    header('Access-Control-Allow-Origin: *');

    $ch = curl_init($target_url);
    curl_setopt($ch, CURLOPT_UNIX_SOCKET_PATH, $sock_file);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, false);  // Modo streaming
    curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, 600);            // 10 minutos max
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Accept: text/event-stream',
        'Cache-Control: no-cache',
        'X-Forwarded-Prefix: /trading',
    ]);
    curl_setopt($ch, CURLOPT_WRITEFUNCTION, function($ch, $data) {
        echo $data;
        if (ob_get_level()) ob_flush();
        flush();
        // Si el cliente se desconectó, abortar
        return connection_aborted() ? 0 : strlen($data);
    });

    curl_exec($ch);
    curl_close($ch);
    exit;
}

// ── Routing ──────────────────────────────────────────────────────────────────
// Ruta SSE → streaming inmediato, sin buffer, sin caché
if ($is_sse && file_exists($sock_file)) {
    sse_proxy_stream($target_url, $sock_file);
}

// Resto de requests → proxy estándar con buffer
list($raw_response, $http_code, $header_size) = proxy_request($target_url, $sock_file);

if ($http_code === 0 || $raw_response === false) {
    usleep(300000);
    list($raw_response, $http_code, $header_size) = proxy_request($target_url, $sock_file);
}

if ($http_code > 0 && $raw_response !== false) {
    http_response_code($http_code);
    $header_text = substr($raw_response, 0, $header_size);
    $body = substr($raw_response, $header_size);

    foreach (explode("\r\n", $header_text) as $hdr) {
        if (stripos($hdr, 'Content-Type:') === 0) {
            header($hdr);
        }
    }

    // Guardar en micro-cache si fue exitoso
    if ($http_code === 200 && $is_telemetry && !empty($body)) {
        @file_put_contents($cache_file, $body);
    }

    echo $body;
} else {
    // Si cURL no pudo conectar y es telemetría, servir caché reciente (hasta 60s)
    if ($is_telemetry && file_exists($cache_file) && (time() - filemtime($cache_file) < 60)) {
        header('Content-Type: application/json; charset=utf-8');
        header('X-Quant-Cache: STALE');
        readfile($cache_file);
        exit;
    }
    wake_daemon_in_background($python_bin, $daemon_script);
    show_starting_page($is_api);
}
