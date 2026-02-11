# voice

PC向けの「自作 音声入力ソフト」です。  
目的は、OS標準の音声入力よりも使いやすくすることです。

このアプリでできること:
- ホットキーで録音の開始/停止
- `faster-whisper` で高精度の音声認識
- 言い間違いの自動変換（辞書）
- フィラー（「えーと」「あの」など）除去
- 認識結果を自動で貼り付け（またはコピーのみ）

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
1. `Ctrl + Alt + Space` を押す（録音開始）
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
- `normalization.replacements`  
  誤認識を直す辞書
- `normalization.fillers`  
  消したい口ぐせリスト

## 変換辞書の例

```toml
[normalization.replacements]
"えーあい" = "AI"
"ちゃっとじーぴーてぃー" = "ChatGPT"
"おーぷんえーあい" = "OpenAI"
```

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
