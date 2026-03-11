# voice

PC向けの「自作 音声入力ソフト」です。  
OS標準より使いやすくするために、次の機能を実装しています。

- ホットキーで録音開始/停止
- `faster-whisper` で高精度の音声認識
- 言い間違いの自動変換（共通辞書 + 個人辞書）
- フィラー（「えーと」「あの」など）除去
- 文字起こし結果を自動で貼り付け（またはコピーのみ）
- 無音を検知して自動停止
- 修正内容を学習して辞書に反映
- 小さいマイクボタンをクリックするだけの簡単UI
- 文字起こし履歴の自動保存
- 最後の結果の再貼り付け / 再コピー
- macメニューバー常駐
- ログイン時の自動起動
- 設定GUI

## いちばん簡単な使い方

macなら、まずはこれで十分です。

迷ったら、まず [BEGINNER_MANUAL_JA.md](/Users/hasegawaatsuki/Documents/New%20project/voice/BEGINNER_MANUAL_JA.md) を見てください。

macアプリ版を使う場合:
- アプリ本体: `/Users/hasegawaatsuki/Applications/Voice Input.app`
- デスクトップショートカット: `/Users/hasegawaatsuki/Desktop/Voice Input.app`
- 更新用: `/Users/hasegawaatsuki/Desktop/update_voice_app.command`

1. [start_mic_button.command](/Users/hasegawaatsuki/Documents/New%20project/voice/start_mic_button.command) をダブルクリック
2. 小さい `MIC` ボタンが出る
3. そのボタンを1回押して話す
4. 無音になると自動で止まって、AI処理して貼り付ける

Typelessっぽく使うなら、この起動方法がいちばん近いです。

最新版へ更新したいとき:
1. `/Users/hasegawaatsuki/Desktop/update_voice_app.command` をダブルクリック
2. 古いアプリが閉じて、最新版が起動する

## Wi-Fiについて

この自作版は、今の実装では `faster-whisper` をローカルで動かしています。  
そのため、文字起こし自体は Wi-Fi が落ちても止まりません。

Typelessでよくある「喋ったのに通信エラーで消える」を避けるために、次も入れています。

- 文字起こし結果をローカル履歴に自動保存
- `Paste Last` で最後の結果を再貼り付け
- `Copy Last` で最後の結果を再コピー
- `Open History` で履歴ファイルを開ける

## まずは5分で動かす

前提: macOS / Python 3.10+（3.11推奨）

```bash
cd '/Users/hasegawaatsuki/Documents/New project/voice'
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --config config.toml --mic-button
```

起動後の基本操作:
1. `MIC` ボタンを押して録音開始
2. もう一度押すか、無音で自動停止
3. 文字起こし結果が貼り付けされる

`config.toml` が無い場合は初回起動時に自動作成されます。

## macアプリ版

今回、SwiftUI ベースの mac ネイティブアプリも追加しました。

- ソース: `/Users/hasegawaatsuki/Documents/New project/voice/VoiceMacApp`
- ビルドスクリプト: [build_mac_app.command](/Users/hasegawaatsuki/Documents/New%20project/voice/build_mac_app.command)
- オフライン補助: [offline_transcribe.py](/Users/hasegawaatsuki/Documents/New%20project/voice/offline_transcribe.py)

アプリでできること:
- 小さいマイクUI
- `そのまま` と `AIで整える` の2種類の録音ボタン
- 録音中の波形表示と状態表示
- 録音開始音 / 終了音
- 日本語メインのUI
- 標準表示 / 小型表示の切り替え
- OpenAI APIキーの保存と接続テスト
- 録音ショートカット変更
- ショートカットで使う録音モードの選択
- モード切替
  Offline / Balanced / Best
- 履歴保存
- Paste Last / Copy Last
- 今月の使用量表示

## 重要: macの権限

初回のみ必須です。
- マイク権限
- アクセシビリティ権限（自動貼り付け/ホットキー）

## 機能1: 無音で自動停止

`config.toml` のこの3つで調整します。

```toml
auto_stop_silence_enabled = true
auto_stop_silence_sec = 1.2
auto_stop_min_record_sec = 0.7
silence_level_threshold = 0.01
```

## 機能2: 学習（修正内容を辞書に反映）

やり方:
1. いつも通り音声入力する
2. 変換ミスを手で直す
3. 直した文章をコピー
4. `Cmd + Shift + L` を押す（学習ホットキー）

すると `user_replacements_ja.toml` に辞書が追加されます。

## 機能3: 小さいマイクボタンUI

```bash
python main.py --config config.toml --mic-button
```

使えるボタン:
- `MIC`: 録音開始 / 停止
- `Paste Last`: 最後の結果をもう一度貼り付け
- `Copy Last`: 最後の結果をコピー
- `Open History`: 履歴を開く
- `Learn`: 修正内容を辞書へ反映

## 機能4: メニューバー常駐（mac）

```bash
python main.py --config config.toml --tray
```

メニューバーに `Voice` が出ます。
- `Voice●` は録音中
- メニューから録音/設定/学習/ログイン起動のON/OFFが可能

## 機能5: ログイン時に自動起動

有効化:
```bash
python main.py --config config.toml --install-login-item
```

有効化すると、ログイン時に `MIC` ボタンUIが自動で立ち上がります。

無効化:
```bash
python main.py --uninstall-login-item
```

## 機能6: 設定GUI

```bash
python main.py --config config.toml --settings
```

GUIで変更できる項目:
- 録音ホットキー
- 学習ホットキー
- モデル
- 無音自動停止
- マイクボタンのタイトル
- 自動貼り付け
- 辞書ON/OFF

## 予算について

今の実装はローカル処理なので、月額課金は基本ありません。  
つまり、あなたのイメージしている「月500円くらいまで」は十分クリアできます。

## よく使う設定

- `app.hotkey`: 録音開始/停止キー
- `app.learn_hotkey`: 学習キー
- `app.model_name`: 精度優先なら `large-v3`
- `app.paste_after_transcribe`: `true` で貼り付け
- `app.use_common_replacements`: 共通辞書の使用
- `app.use_user_replacements`: 学習辞書の使用
- `app.transcript_history_file`: 履歴ファイル

## ファイル構成

- `main.py`: 本体コード
- `VoiceMacApp`: macアプリ本体
- [build_mac_app.command](/Users/hasegawaatsuki/Documents/New%20project/voice/build_mac_app.command): macアプリをビルドして配置
- [offline_transcribe.py](/Users/hasegawaatsuki/Documents/New%20project/voice/offline_transcribe.py): Offlineモード補助
- `config.example.toml`: 設定サンプル
- `common_replacements_ja.toml`: 共通の言い間違い辞書
- `user_replacements_ja.toml`: 学習で自動生成される辞書
- `transcript_history.jsonl`: 履歴ファイル
- [start_mic_button.command](/Users/hasegawaatsuki/Documents/New%20project/voice/start_mic_button.command): ダブルクリック起動
- [open_settings.command](/Users/hasegawaatsuki/Documents/New%20project/voice/open_settings.command): 設定画面を開く
- [backup_to_github.command](/Users/hasegawaatsuki/Documents/New%20project/voice/backup_to_github.command): GitHubへバックアップ
- [BEGINNER_MANUAL_JA.md](/Users/hasegawaatsuki/Documents/New%20project/voice/BEGINNER_MANUAL_JA.md): 初心者向けマニュアル
- `IMPLEMENTATION_MEMO_JA.md`: 実装手順メモ
