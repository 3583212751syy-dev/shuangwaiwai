@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting image-match local preview...
echo Open http://127.0.0.1:8788 in your browser. Keep this window open.
node preview.mjs
pause
