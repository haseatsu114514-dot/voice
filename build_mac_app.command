#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_DIR="${ROOT_DIR}/VoiceMacApp"
APP_NAME="Voice Input"
APP_DIR="${HOME}/Applications/${APP_NAME}.app"
BIN_PATH="$(cd "${PACKAGE_DIR}" && swift build -c release --show-bin-path)"
EXECUTABLE_PATH="${BIN_PATH}/VoiceMacApp"

cd "${PACKAGE_DIR}"
swift build -c release

mkdir -p "${APP_DIR}/Contents/MacOS"
mkdir -p "${APP_DIR}/Contents/Resources"

cp "${EXECUTABLE_PATH}" "${APP_DIR}/Contents/MacOS/VoiceMacApp"

cat > "${APP_DIR}/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>ja</string>
  <key>CFBundleExecutable</key>
  <string>VoiceMacApp</string>
  <key>CFBundleIdentifier</key>
  <string>com.haseatsu.voiceinput.macapp</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>${APP_NAME}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>NSMicrophoneUsageDescription</key>
  <string>音声入力のためにマイクを使います。</string>
</dict>
</plist>
EOF

touch "${APP_DIR}/Contents/PkgInfo"
codesign --force --deep -s - "${APP_DIR}" >/dev/null 2>&1 || true

ln -sfn "${APP_DIR}" "${HOME}/Desktop/${APP_NAME}.app"

echo "Built app:"
echo "${APP_DIR}"
echo ""
echo "Desktop shortcut:"
echo "${HOME}/Desktop/${APP_NAME}.app"
