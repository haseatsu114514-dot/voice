# Voice Input Mac App 設計書

## 1. 目的

Typeless の手軽さと AquaVoice の常駐感を両立した、mac 向け音声入力アプリを作る。

重視する点:
- 1クリックで使える
- Typeless より通信失敗に強い
- 月 1000 円以内に収まりやすい
- API キーやショートカットをアプリ内で設定できる

## 2. 結論

このアプリは `SwiftUI の macOS ネイティブアプリ` として作る。

音声認識方式は 3 モードにする。

1. `Offline`
   ローカル音声認識。最安。通信失敗に最も強い。
2. `Balanced`
   OpenAI `gpt-4o-mini-transcribe` を使う。速度と価格のバランスが良い。
3. `Best`
   OpenAI `gpt-4o-transcribe` を使う。月 1000 円以内で高品質を狙う本命。

初期設定のおすすめは `Best`。

理由:
- 現在の利用量なら月 1000 円以内に収まりやすい
- Typeless より高い即時性は保証できないが、十分実用速度を狙える
- 通信エラー時は履歴と再貼り付けで復旧できる

## 3. 想定コスト

前提:
- 16 日で 8 時間 8 分利用
- 月換算で約 915 分から 946 分

概算:
- `Offline`: ほぼ 0 円
- `Balanced`: 約 430 円から 450 円
- `Best`: 約 860 円から 900 円

方針:
- デフォルトは `Best`
- 月額見積もりが 900 円を超えそうなら警告
- 1000 円を超えそうなら `Balanced` への切り替えを提案

## 4. アプリの完成イメージ

### 常駐UI

画面は 2 つ。

1. 小さいマイクボタンウィンドウ
2. 設定ウィンドウ

### 小さいマイクボタンウィンドウ

必要な要素:
- `MIC` ボタン
- 現在状態の表示
  Ready / Listening / Processing / Error
- 最後の文字起こしプレビュー
- `Paste Last`
- `Copy Last`
- `Open History`
- `Settings`

操作:
- ボタンを 1 回押すと録音開始
- もう 1 回押すか、無音で自動停止
- 結果は現在の入力欄に貼り付け

### 設定ウィンドウ

必要な設定項目:
- 音声認識モード
  Offline / Balanced / Best
- OpenAI API Key
- 録音ショートカット
- 学習ショートカット
- 無音停止秒数
- 自動貼り付け ON/OFF
- フィラー除去 ON/OFF
- 起動時に常駐するか
- ログイン時に起動するか

## 5. 技術方針

### UI

- `SwiftUI`
- `MenuBarExtra`
- `Settings` Scene
- 小さい独立ウィンドウ

### 音声録音

- `AVFoundation`
- 16kHz mono PCM
- RMS で無音検知

### グローバルショートカット

- macOS のグローバルホットキー機能を実装
- UI からキーを再設定可能にする
- 保存先は `UserDefaults`

### API キー保存

- API キーは `Keychain` に保存
- 設定画面で入力
- 平文ファイルには保存しない

### 履歴保存

- `Application Support` に履歴を保存
- 形式は JSON Lines
- 保存内容:
  timestamp / rawText / normalizedText / provider / mode / success

## 6. API 接続の考え方

### 接続先

最初は `OpenAI` だけ対応する。

理由:
- 月 1000 円以内で品質が良い
- 音声認識系モデルが揃っている
- 今回の用途で最も単純に組める

### 設定方法

設定画面に以下を置く。

- `Provider`
  OpenAI
- `API Key`
  入力欄
- `Test Connection`
  接続確認ボタン

### 接続テスト

`Test Connection` を押したら:
- API キー形式を簡易確認
- 軽いリクエストを投げる
- 成功したら `Connected`
- 失敗したら理由を表示

## 7. 音声認識パイプライン

### Offline

1. 録音
2. ローカル音声認識
3. 正規化
4. 貼り付け
5. 履歴保存

### Balanced / Best

1. 録音
2. OpenAI へ送信
3. 認識結果を受信
4. 正規化
5. 貼り付け
6. 履歴保存

### 失敗時の動作

最重要方針:
- 喋った内容を失わない

対応:
- 録音データを一時保存
- エラー時は `Retry`
- 最後の結果は `Paste Last`
- 結果は必ず履歴へ保存
- API モード失敗時は Offline 再試行を選べる

## 8. Typeless / AquaVoice との差別化

Typeless より強くしたい点:
- 通信失敗時に結果を失いにくい
- 履歴から復旧できる
- API あり / なしを自分で切り替えられる
- 月額コストを自分で制御できる

AquaVoice に寄せる点:
- 常駐
- すぐ押せるマイクボタン
- ショートカットで即録音
- 入力欄への即貼り付け

## 9. 実装フェーズ

### Phase 1

- SwiftUI アプリ骨格
- 小さいマイクボタンUI
- 設定画面
- UserDefaults 保存
- Keychain 保存

### Phase 2

- OpenAI API 接続
- モード切替
- 接続テスト
- 月額コスト概算表示

### Phase 3

- グローバルショートカット変更UI
- 履歴ビュー
- 再貼り付け / 再コピー
- エラー時の再試行導線

### Phase 4

- ローカル認識の組み込み
- Offline フォールバック
- 仕上げ

## 10. 最終判断

このアプリは `Typeless より安く`, `Typeless より事故に強い` 形で成立する可能性が高い。

条件:
- クラウド依存を最小化する
- OpenAI は `Best` と `Balanced` の 2 モードで使い分ける
- 履歴保存と再貼り付けを標準機能にする

## 11. 次に実装するもの

次の実装対象はこれ。

1. SwiftUI ベースの mac アプリ本体
2. 設定画面
3. API キー入力と接続テスト
4. ショートカット変更
5. 小さいマイクボタンUI
