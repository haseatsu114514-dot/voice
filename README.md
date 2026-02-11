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
- macメニューバー常駐
- ログイン時の自動起動
- 設定GUI

## まずは5分で動かす

前提: macOS / Python 3.10+（3.11推奨）

```bash
cd '/Users/hasegawaatsuki/Documents/New project/voice'
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.toml config.toml
python main.py --config config.toml
```

起動後の基本操作:
1. `Cmd + Shift + Space` で録音開始
2. もう一度同じキーで録音停止（または無音で自動停止）
3. 文字起こし結果が貼り付けされる

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

## 機能3: メニューバー常駐（mac）

```bash
python main.py --config config.toml --tray
```

メニューバーに `Voice` が出ます。
- `Voice●` は録音中
- メニューから録音/設定/学習/ログイン起動のON/OFFが可能

## 機能4: ログイン時に自動起動

有効化:
```bash
python main.py --config config.toml --install-login-item
```

無効化:
```bash
python main.py --uninstall-login-item
```

## 機能5: 設定GUI

```bash
python main.py --config config.toml --settings
```

GUIで変更できる項目:
- 録音ホットキー
- 学習ホットキー
- モデル
- 無音自動停止
- 自動貼り付け
- 辞書ON/OFF

## よく使う設定

- `app.hotkey`: 録音開始/停止キー
- `app.learn_hotkey`: 学習キー
- `app.model_name`: 精度優先なら `large-v3`
- `app.paste_after_transcribe`: `true` で貼り付け
- `app.use_common_replacements`: 共通辞書の使用
- `app.use_user_replacements`: 学習辞書の使用

## ファイル構成

- `main.py`: 本体コード
- `config.example.toml`: 設定サンプル
- `common_replacements_ja.toml`: 共通の言い間違い辞書
- `user_replacements_ja.toml`: 学習で自動生成される辞書
- `IMPLEMENTATION_MEMO_JA.md`: 実装手順メモ
