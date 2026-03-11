#!/bin/zsh
set -euo pipefail

APP_PATH="${HOME}/Applications/Voice Input.app"

echo "Voice Input のアクセシビリティ権限を再設定します。"
echo ""
echo "1. 古い権限情報をリセット"
tccutil reset Accessibility com.haseatsu.voiceinput.macapp || true

echo "2. Voice Input を終了"
pkill -x "VoiceMacApp" || true
sleep 1

echo "3. アクセシビリティ設定を開く"
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"

echo ""
echo "次の手順:"
echo " - 一覧にある Voice Input を OFF -> ON"
echo " - 一覧になければ + から ${APP_PATH} を追加"
echo " - その後、${APP_PATH} を開き直す"
