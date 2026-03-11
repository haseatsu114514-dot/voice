# 実装手順メモ（初心者向け）

このメモは「どう作ったか」を、なるべくやさしく書いたものです。

## 今回追加した機能

1. 無音で自動停止
2. 学習（修正内容を辞書に反映）
3. 小さいマイクボタンUI
4. 文字起こし履歴の自動保存
5. 再貼り付け / 再コピー
6. macメニューバー常駐 + ログイン起動
7. 設定GUI
8. 初回起動時の `config.toml` 自動作成

## 1. 無音で自動停止

やったこと:
- 録音中に音量(RMS)を毎回計測
- 一定時間、音量がしきい値より低ければ自動停止

使う設定:
- `auto_stop_silence_enabled`
- `auto_stop_silence_sec`
- `auto_stop_min_record_sec`
- `silence_level_threshold`

## 2. 学習機能

課題:
- 自動変換を強くするには「あなたの言い間違い」を学ぶ必要がある

実装:
- 最後の出力結果を保持
- ユーザーが修正した文をコピー
- 学習ホットキーで差分比較
- 置換ペアを `user_replacements_ja.toml` へ保存

補足:
- 共通辞書: `common_replacements_ja.toml`
- 個人学習辞書: `user_replacements_ja.toml`
- 手動辞書: `config.toml` の `[normalization.replacements]`
- 優先順位は「手動辞書 > 学習辞書 > 共通辞書」

## 3. 小さいマイクボタンUI

Typelessっぽく使うには、ホットキーだけでは弱い。  
そこで、常に前面に置ける小さい `MIC` ボタンを追加した。

できること:
- `MIC` で録音開始 / 停止
- `Paste Last` で最後の結果を再貼り付け
- `Copy Last` で最後の結果を再コピー
- `Open History` で履歴を確認
- `Learn` で修正を学習

## 4. 履歴保存

Typeless系で一番つらいのは「喋った内容が消えること」なので、  
毎回の文字起こし結果を `transcript_history.jsonl` に保存するようにした。

保存内容:
- タイムスタンプ
- 生の認識結果
- 整形後の結果

## 5. 再貼り付け / 再コピー

貼り付け先アプリ側で失敗しても、話し直さなくていいようにした。

- `Paste Last`: 最後の結果をもう一度貼る
- `Copy Last`: 最後の結果をクリップボードへ戻す

## 6. macメニューバー常駐

`rumps` を使ってメニューバーアプリ化。

メニュー項目:
- 録音開始/停止
- 修正を学習(クリップボード)
- 設定を開く
- 設定を再読込
- ログイン起動 ON/OFF
- 終了

録音中はタイトルを `Voice●` にして状態が見えるようにした。

## 7. ログイン起動

`launchd` の `LaunchAgent` を使う形にした。

- 有効化で plist を作成して `launchctl load`
- 無効化で `launchctl unload` + plist削除
- ログイン時には `MIC` ボタンUIを起動する

## 8. 設定GUI

`tkinter` でシンプルな設定画面を追加。

編集できる項目:
- ホットキー
- 学習ホットキー
- モデル
- 無音自動停止
- マイクボタンのタイトル
- 辞書ON/OFF
- フィラー一覧

保存時は TOML を更新するだけの最小構成。

## 9. 初回起動を簡単にした理由

初心者向けでは `config.toml` が無いだけで起動失敗するのがかなりつらい。  
そのため、初回起動時に `config.example.toml` から自動生成するようにした。

## 10. 使い分け

- 通常CLI: `python main.py --config config.toml`
- マイクボタンUI: `python main.py --config config.toml --mic-button`
- メニューバー: `python main.py --config config.toml --tray`
- GUI設定: `python main.py --config config.toml --settings`
- ログイン起動ON: `python main.py --config config.toml --install-login-item`
- ログイン起動OFF: `python main.py --uninstall-login-item`
- ダブルクリック起動: `start_mic_button.command`
