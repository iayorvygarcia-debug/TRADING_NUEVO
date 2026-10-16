/**
 * AI Quant Trading Platform — Node.js Launcher Helper
 * Permite levantar el servidor Python FastAPI / Uvicorn directamente con `node server.js`.
 */
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

const PORT = 8000;
const HOST = '127.0.0.1';
const isWindows = process.platform === 'win32';
const pythonVenvUvicorn = path.join(
    __dirname,
    '.venv',
    isWindows ? 'Scripts' : 'bin',
    isWindows ? 'uvicorn.exe' : 'uvicorn'
);

// Verificar si el servidor ya está activo
const req = http.get(`http://${HOST}:${PORT}/health`, (res) => {
    console.log(`\n======================================================`);
    console.log(` [OK] El servidor ya está corriendo en: http://${HOST}:${PORT}`);
    console.log(` Dashboard Web disponible en:          http://localhost:${PORT}/dashboard`);
    console.log(`======================================================\n`);
    process.exit(0);
});

req.on('error', () => {
    // Si no está corriendo, levantarlo con uvicorn
    console.log(`\n[+] Iniciando AI Quant Trading Platform con Uvicorn...`);
    console.log(`[+] Dashboard: http://localhost:${PORT}/dashboard\n`);

    const child = spawn(
        pythonVenvUvicorn,
        ['apps.api.main:app', '--reload', '--host', HOST, '--port', String(PORT)],
        {
            cwd: __dirname,
            stdio: 'inherit',
            shell: true,
        }
    );

    child.on('error', (err) => {
        console.error('[-] Error al arrancar uvicorn:', err);
    });

    child.on('exit', (code) => {
        console.log(`[i] Servidor finalizado con código: ${code}`);
    });
});
