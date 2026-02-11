# voice

PC向けの「自作 音声入力ソフト」です。  
目的は、OS標準の音声入力よりも使いやすくすることです。

このアプリでできること:
- ホットキーで録音の開始/停止
- `faster-whisper` で高精度の音声認識
- 言い間違いの自動変換（辞書）
- フィラー（「えーと」「あの」など）除去
- 認識結果を自動で貼り付け（またはコピーのみ）

## macで使える？

使えます。mac向けの動作は実装済みです。
- 貼り付けは `Cmd + V` で自動実行
- デフォルトの録音ホットキーは `Cmd + Shift + Space`
- `pynput` のグローバルホットキーで、どのアプリ上でも反応

注意: アプリ本体（`python main.py ...`）は起動したままにしてください。  
ホットキーで「録音を開始/停止」できます。

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

起動したら:
1. `Cmd + Shift + Space` を押す（録音開始）
2. もう一度同じキーを押す（録音停止）
3. 文字起こし結果が入力欄に貼り付けられます

## よく使う設定（ここだけ覚えればOK）

`config.toml` を開いて編集します。

- `app.hotkey`  
  録音の開始/停止キー
- `app.model_name`  
  精度優先なら `large-v3`（重い）、軽さ優先なら `small` / `medium`
- `app.paste_after_transcribe`  
  `true` = 自動貼り付け, `false` = クリップボードにコピーのみ
- `app.use_common_replacements`  
  `true` で「よくある言い間違い辞書」を自動で使う
- `app.common_replacements_file`  
  共通辞書ファイル（初期値: `common_replacements_ja.toml`）
- `normalization.replacements`  
  あなた専用の誤認識修正（共通辞書より優先）
- `normalization.fillers`  
  消したい口ぐせリスト

## 特定キーを自分用に変える

`config.toml` の `app.hotkey` を変更します。

例:
```toml
hotkey = "<f8>"
```

ほかの例:
```toml
hotkey = "<cmd>+<option>+space"
```

## 変換辞書の例

```toml
[normalization.replacements]
"えーあい" = "AI"
"ちゃっとじーぴーてぃー" = "ChatGPT"
"おーぷんえーあい" = "OpenAI"
```

共通の「よくある言い間違い」は `common_replacements_ja.toml` に最初から入っています。

## フィラー除去の例

```toml
[normalization]
remove_fillers = true
fillers = ["えーと", "えっと", "あの", "その", "うーん", "えー"]
```

## 初回でハマりやすい点

- マイク権限がないと録音できない
- アクセシビリティ権限がないと自動貼り付けできない
- 重いモデルは初回ロードに時間がかかる

## 実装の中身を知りたい場合

`IMPLEMENTATION_MEMO_JA.md` に、初心者向けで実装手順をまとめています。
