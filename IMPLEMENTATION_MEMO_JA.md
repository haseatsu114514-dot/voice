# 実装手順メモ（初心者向け）

このメモは「どう作ったか」を、なるべくやさしく書いたものです。

## 1. 目標を決める

今回は次の4機能をゴールにしました。
- 精度を上げる（Whisper系モデルを使う）
- キー1つで録音開始/停止
- 言い間違いを自動で直す
- フィラーを自動で消す

## 2. 技術の選定

- 音声認識: `faster-whisper`
- 録音: `sounddevice`
- グローバルホットキー: `pynput`
- 貼り付け: `pyperclip` + キーボード操作
- 設定: `config.toml`
- 共通の言い間違い辞書: `common_replacements_ja.toml`

理由: Pythonで組みやすく、試行錯誤しやすいからです。

## 3. 実装の流れ

### 3-1. 録音機能

`AudioRecorder` クラスでマイク入力をバッファに保存します。  
ホットキーで「開始」と「停止」を切り替える設計です。

### 3-2. 音声認識

`WhisperTranscriber` クラスで音声配列を文字列に変換します。  
モデルは `config.toml` で変更できます。

### 3-3. テキスト整形

`TextNormalizer` クラスで次を実施します。
- 置換辞書で誤認識を修正
- フィラーを削除
- 余分な空白・改行・句読点の連続を整える

辞書は2段構成です。
- 共通辞書: `common_replacements_ja.toml`（よくある修正を最初から収録）
- 個人辞書: `config.toml` の `[normalization.replacements]`

同じキーがある場合は「個人辞書」が優先されます。

### 3-4. 入力欄へ反映

`OutputInserter` クラスで、結果をクリップボードに入れます。  
`paste_after_transcribe = true` なら `Cmd+V` / `Ctrl+V` で自動貼り付けします。

## 4. 精度改善のやり方

### 4-1. モデルを重くする

`app.model_name = "large-v3"` は精度が高いです。  
ただし、PC性能が低いと遅くなります。

### 4-2. 初期プロンプトを使う

専門用語が多い場合は `app.initial_prompt` に単語を入れると有利です。

例:
```toml
initial_prompt = "OpenAI, ChatGPT, API, Python, 音声認識"
```

### 4-3. 置換辞書を育てる

実際に使って出たミスを `normalization.replacements` に追加します。  
ここを増やすと、体感精度がかなり上がります。

## 5. ファイル構成

- `main.py`: 本体コード
- `config.example.toml`: 設定サンプル
- `common_replacements_ja.toml`: よくある言い間違い辞書
- `requirements.txt`: 必要ライブラリ
- `README.md`: 使い方
- `IMPLEMENTATION_MEMO_JA.md`: この実装メモ

## 6. 次の改善アイデア

- Push-to-talk（押してる間だけ録音）
- アプリ別の置換辞書（Slack用、Notion用など）
- 音声コマンド（例: 「改行」「句点」「箇条書き」）
- GUI化（設定画面つき）
