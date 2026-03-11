# はじめて使う人向けマニュアル

このファイルは、AIやプログラムに詳しくない人向けの手順書です。  
目的は次の2つです。

1. この音声入力ソフトを使えるようにする
2. 念のため GitHub にバックアップできるようにする

---

## 1. まず知っておくこと

このソフトは、mac で使う音声入力ソフトです。  
小さい `MIC` ボタンを押して話すと、AI が文字に変換してくれます。

特徴:
- 文字起こしは基本的に自分のPC内で処理する
- Wi-Fi が不安定でも、Typelessより内容を失いにくい
- 最後の結果をもう一度貼る機能がある
- 履歴ファイルも自動で残る

---

## 2. 最初に必要なもの

必要なものはこれだけです。

- mac
- インターネット
  最初のインストール時に使います
- GitHub アカウント
  バックアップするときに使います

---

## 3. いちばん簡単な使い方

### 手順

いまは 2 通りあります。

### A. 一番おすすめ: macアプリ版

1. デスクトップの `Voice Input.app` をダブルクリック
2. アプリが開いたら `そのまま` か `AIで整える` を押す
3. 初回だけマイク権限とアクセシビリティ権限を許可する
4. 録音中は波形と `音声入力中` の表示が出る
5. 録音の開始時と終了時に小さな効果音が鳴る
6. 右上の `設定` を押すと、APIキーやショートカットを設定できる
7. 邪魔なときは `小さくする` で小型表示に切り替える
8. 新しいバージョンにしたいときは、デスクトップの `update_voice_app.command` をダブルクリックする

### B. これまでのPython版

1. Finderでこのフォルダを開く  
   `/Users/hasegawaatsuki/Documents/New project/voice`

2. [start_mic_button.command](/Users/hasegawaatsuki/Documents/New%20project/voice/start_mic_button.command) をダブルクリックする

3. 初回は少し待つ  
   理由: 必要な部品を自動で入れるためです

4. 小さい `MIC` ボタンが出たら準備完了

5. `MIC` を1回押して話す

6. 話し終わったら:
- もう1回 `MIC` を押す
- そのまま黙る  
  無音を検知すると自動で止まります

7. 文字起こし結果が入力欄に貼り付けられる

---

## 4. 初回だけ必要な許可

最初は mac から許可を求められることがあります。

必要な許可:
- マイク
- アクセシビリティ

もし許可画面が出たら、`許可` を選んでください。  
これがないと、録音や自動貼り付けが動きません。

---

## 5. うまく動かないときの見方

### 1. `MIC` ボタンが出ない

次を確認してください。

- `start_mic_button.command` をダブルクリックしたか
- 初回インストール中で待っているだけではないか
- mac のセキュリティ警告が出て止まっていないか

### 2. 話しても文字が出ない

次を確認してください。

- マイク権限が許可されているか
- 入力先アプリが文字を貼り付けられる状態か
- アクセシビリティ権限が許可されているか

### 3. 貼り付けに失敗した

このソフトは、最後の結果を残しています。  
そのため、話し直さなくても復旧できます。

使うボタン:
- `前回を貼る`
  最後の結果をもう一度貼る
- `前回をコピー`
  最後の結果をコピーする
- `履歴を開く`
  履歴ファイルを開く

---

## 6. 設定を変えたいとき

設定画面を開くには、macアプリ版ならアプリ内の `設定` を押します。
Python版なら、[open_settings.command](/Users/hasegawaatsuki/Documents/New%20project/voice/open_settings.command) をダブルクリックします。

変更しやすい項目:
- 録音ホットキー
- ショートカットで使う録音モード
- 開始音と終了音
- 表示サイズ
- モード
- 無音で止まるまでの秒数
- OpenAI APIキー
- 今のショートカット確認

難しい設定は、最初は触らなくて大丈夫です。

---

## 7. いちばん大事なファイル

よく使うファイルはこの3つです。

- [start_mic_button.command](/Users/hasegawaatsuki/Documents/New%20project/voice/start_mic_button.command)
  起動用
- [open_settings.command](/Users/hasegawaatsuki/Documents/New%20project/voice/open_settings.command)
  設定変更用
- [README.md](/Users/hasegawaatsuki/Documents/New%20project/voice/README.md)
  機能の全体説明

---

## 8. バックアップとは何か

バックアップとは、今の作業内容を GitHub に保存しておくことです。  
これをしておくと、あとでPCが壊れても戻しやすくなります。

今回のバックアップ先は、このリポジトリです。

- GitHub: [voice](https://github.com/haseatsu114514-dot/voice)

---

## 9. いちばん簡単なバックアップ方法

### 方法A: 補助スクリプトを使う

1. [backup_to_github.command](/Users/hasegawaatsuki/Documents/New%20project/voice/backup_to_github.command) をダブルクリック
2. 何か一言メモを入れる  
   例: `マイクボタン調整`
3. Enterを押す
4. バックアップ完了まで待つ

これがいちばん簡単です。

---

## 10. バックアップを手でやる方法

もし手でやる場合は、ターミナルでこの順番です。

```bash
cd '/Users/hasegawaatsuki/Documents/New project/voice'
git status
git add .
git commit -m "作業メモを書く"
git push origin main
```

`git commit -m "作業メモを書く"` の部分は、今何を変えたかを短く書けばOKです。

例:

```bash
git commit -m "READMEを更新"
```

---

## 11. バックアップでエラーが出たとき

### `nothing to commit` と出た

変更がない状態です。  
そのままで問題ありません。

### `push` で止まった

GitHub側に別の更新が入っている可能性があります。  
その場合は、次を順番に試します。

```bash
cd '/Users/hasegawaatsuki/Documents/New project/voice'
git pull --rebase origin main
git push origin main
```

---

## 12. 普段のおすすめ運用

普段はこの流れだけで十分です。

1. `start_mic_button.command` をダブルクリック
2. `MIC` を押して使う
3. 何か変更したら `backup_to_github.command` をダブルクリック

---

## 13. 迷ったらここを見る

まずはこの順です。

1. このファイル
2. [README.md](/Users/hasegawaatsuki/Documents/New%20project/voice/README.md)
3. それでも不明なら、今の状態をそのままAIに見せて質問する
