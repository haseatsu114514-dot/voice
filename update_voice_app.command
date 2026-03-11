#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_PATH="${HOME}/Applications/Voice Input.app"
PROCESS_PATH="${APP_PATH}/Contents/MacOS/VoiceMacApp"

cd "${ROOT_DIR}"

echo "Voice Input を更新します。"
echo ""

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ローカルに未保存の変更があるため、GitHub からの pull はスキップします。"
  echo "このMac上の内容を使ってアプリだけ再生成します。"
else
  echo "GitHub から最新版を取得しています..."
  git pull --ff-only origin main
fi

echo ""
echo "アプリを再生成しています..."
"${ROOT_DIR}/build_mac_app.command"

echo ""
echo "古いアプリを終了しています..."
pkill -f "${PROCESS_PATH}" || true
sleep 1

echo "最新版を起動しています..."
open "${APP_PATH}"

echo ""
echo "更新が終わりました。"
