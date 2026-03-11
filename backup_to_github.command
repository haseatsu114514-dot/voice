#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"

current_branch="$(git branch --show-current)"

echo "Current branch:"
echo "${current_branch}"
echo ""

echo "Working tree status:"
git status --short
echo ""

read "commit_message?バックアップ用メッセージを入れてください: "

if [[ -z "${commit_message}" ]]; then
  commit_message="backup $(date '+%Y-%m-%d %H:%M:%S')"
fi

git add .

if git diff --cached --quiet; then
  echo ""
  echo "変更がないため、commit は作りませんでした。"
  exit 0
fi

git commit -m "${commit_message}"

echo ""
echo "Remote の最新状態を取り込みます..."
git pull --rebase origin "${current_branch}"
git push origin "${current_branch}"

echo ""
echo "GitHub へのバックアップが完了しました。"
