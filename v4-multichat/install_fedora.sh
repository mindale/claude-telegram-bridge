#!/usr/bin/env bash
# Установка Node.js + Claude Code на Fedora / RHEL-совместимых системах.
# Запускать на сервере от root: bash install_fedora.sh
set -euo pipefail

echo "==> Обновление пакетов"
dnf -y update

echo "==> Установка Node.js 20 LTS через NodeSource"
dnf -y install curl
curl -fsSL https://rpm.nodesource.com/setup_20.x | bash -
dnf -y install nodejs

echo "==> Node.js: $(node -v), npm: $(npm -v)"

echo "==> Установка Claude Code CLI"
npm install -g @anthropic-ai/claude-code

echo "==> Claude Code: $(claude --version)"

echo "==> Установка Python, pip, ffmpeg (нужны для моста с Telegram)"
dnf -y install python3 python3-pip ffmpeg

echo "==> Установка PyTorch (CPU-версия, для озвучки ответов через Silero TTS)"
# Обычный `pip install torch` тянет CUDA-сборку (гигабайты, не нужны без
# GPU) — берём отдельно с официального CPU-индекса PyTorch.
python3 -m pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu

echo
echo "Готово. Дальше:"
echo "1) На своём компьютере (с браузером) выполните: claude setup-token"
echo "2) Скопируйте полученный токен и на сервере выполните:"
echo "   echo 'export CLAUDE_CODE_OAUTH_TOKEN=\"вставьте_токен\"' >> ~/.bashrc && source ~/.bashrc"
echo "3) Проверьте: claude --version && claude -p 'скажи привет'"

