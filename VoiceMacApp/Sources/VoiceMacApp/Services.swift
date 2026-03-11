import AppKit
import AVFoundation
import Carbon
import Combine
import Foundation
import Security

final class SettingsStore: ObservableObject {
    @Published var mode: AppMode { didSet { save() } }
    @Published var recordShortcut: Shortcut { didSet { save() } }
    @Published var defaultCaptureMode: CaptureMode { didSet { save() } }
    @Published var interfaceMode: InterfaceMode { didSet { save() } }
    @Published var autoPaste: Bool { didSet { save() } }
    @Published var soundCuesEnabled: Bool { didSet { save() } }
    @Published var autoStopEnabled: Bool { didSet { save() } }
    @Published var autoStopSeconds: Double { didSet { save() } }
    @Published var fillerRemoval: Bool { didSet { save() } }
    @Published var alwaysOnTop: Bool { didSet { save() } }
    @Published var muteSystemAudioWhileRecording: Bool { didSet { save() } }

    private let defaults = UserDefaults.standard

    init() {
        if let modeRaw = defaults.string(forKey: "settings.mode"),
           let savedMode = AppMode(rawValue: modeRaw) {
            self.mode = savedMode
        } else {
            self.mode = .best
        }

        if let shortcutData = defaults.data(forKey: "settings.recordShortcut"),
           let savedShortcut = try? JSONDecoder().decode(Shortcut.self, from: shortcutData) {
            self.recordShortcut = savedShortcut
        } else {
            self.recordShortcut = .defaultRecord
        }

        if let captureModeRaw = defaults.string(forKey: "settings.defaultCaptureMode"),
           let savedCaptureMode = CaptureMode(rawValue: captureModeRaw) {
            self.defaultCaptureMode = savedCaptureMode
        } else {
            self.defaultCaptureMode = .fastRaw
        }

        if let interfaceModeRaw = defaults.string(forKey: "settings.interfaceMode"),
           let savedInterfaceMode = InterfaceMode(rawValue: interfaceModeRaw) {
            self.interfaceMode = savedInterfaceMode
        } else {
            self.interfaceMode = .standard
        }

        self.autoPaste = defaults.object(forKey: "settings.autoPaste") as? Bool ?? true
        self.soundCuesEnabled = defaults.object(forKey: "settings.soundCuesEnabled") as? Bool ?? true
        self.autoStopEnabled = defaults.object(forKey: "settings.autoStopEnabled") as? Bool ?? true
        self.autoStopSeconds = defaults.object(forKey: "settings.autoStopSeconds") as? Double ?? 1.2
        self.fillerRemoval = defaults.object(forKey: "settings.fillerRemoval") as? Bool ?? true
        self.alwaysOnTop = defaults.object(forKey: "settings.alwaysOnTop") as? Bool ?? true
        self.muteSystemAudioWhileRecording = defaults.object(forKey: "settings.muteSystemAudioWhileRecording") as? Bool ?? true
    }

    private func save() {
        defaults.set(mode.rawValue, forKey: "settings.mode")
        defaults.set(defaultCaptureMode.rawValue, forKey: "settings.defaultCaptureMode")
        defaults.set(interfaceMode.rawValue, forKey: "settings.interfaceMode")
        defaults.set(autoPaste, forKey: "settings.autoPaste")
        defaults.set(soundCuesEnabled, forKey: "settings.soundCuesEnabled")
        defaults.set(autoStopEnabled, forKey: "settings.autoStopEnabled")
        defaults.set(autoStopSeconds, forKey: "settings.autoStopSeconds")
        defaults.set(fillerRemoval, forKey: "settings.fillerRemoval")
        defaults.set(alwaysOnTop, forKey: "settings.alwaysOnTop")
        defaults.set(muteSystemAudioWhileRecording, forKey: "settings.muteSystemAudioWhileRecording")
        if let encoded = try? JSONEncoder().encode(recordShortcut) {
            defaults.set(encoded, forKey: "settings.recordShortcut")
        }
    }
}

final class KeychainService {
    private let service = "com.haseatsu.voiceinput"
    private let account = "openai.api.key"

    func save(_ value: String) throws {
        let data = Data(value.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]

        SecItemDelete(query as CFDictionary)

        var newItem = query
        newItem[kSecValueData as String] = data
        let status = SecItemAdd(newItem as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw NSError(domain: "KeychainService", code: Int(status), userInfo: [
                NSLocalizedDescriptionKey: "APIキーの保存に失敗しました。"
            ])
        }
    }

    func load() -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess,
              let data = item as? Data,
              let string = String(data: data, encoding: .utf8) else {
            return ""
        }
        return string
    }
}

struct HistoryEntry: Codable {
    let timestamp: Date
    let mode: String
    let captureMode: String?
    let provider: String
    let durationSeconds: Double
    let text: String
    let success: Bool
    let errorMessage: String?
    let estimatedUSD: Double?
}

final class HistoryStore {
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder
    let fileURL: URL

    init() {
        self.decoder = JSONDecoder()
        self.decoder.dateDecodingStrategy = .iso8601
        self.encoder = JSONEncoder()
        self.encoder.dateEncodingStrategy = .iso8601

        let supportURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("VoiceInputMacApp", isDirectory: true)
        try? FileManager.default.createDirectory(at: supportURL, withIntermediateDirectories: true)
        self.fileURL = supportURL.appendingPathComponent("transcript_history.jsonl")
    }

    func append(_ entry: HistoryEntry) throws {
        let data = try encoder.encode(entry)
        var line = data
        line.append(0x0A)
        if !FileManager.default.fileExists(atPath: fileURL.path) {
            FileManager.default.createFile(atPath: fileURL.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: fileURL)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: line)
    }

    func currentMonthDurationSeconds() -> Double {
        currentMonthStats().totalDurationSeconds
    }

    func currentMonthStats(referenceDate: Date = Date()) -> MonthlyUsageStats {
        let calendar = Calendar.current
        let entries = readEntries()
            .filter { calendar.isDate($0.timestamp, equalTo: referenceDate, toGranularity: .month) }

        let totalDurationSeconds = entries.reduce(0) { $0 + $1.durationSeconds }
        let successfulSessions = entries.filter(\.success).count
        let failedSessions = entries.filter { !$0.success }.count
        let totalCharacters = entries
            .filter(\.success)
            .reduce(0) { $0 + $1.text.count }
        let estimatedUSD = entries.reduce(0) { partialResult, entry in
            if let stored = entry.estimatedUSD {
                return partialResult + stored
            }
            guard entry.success,
                  let mode = AppMode(rawValue: entry.mode) else {
                return partialResult
            }
            return partialResult + (entry.durationSeconds / 60 * mode.usdCostPerMinute)
        }

        return MonthlyUsageStats(
            totalDurationSeconds: totalDurationSeconds,
            successfulSessions: successfulSessions,
            failedSessions: failedSessions,
            totalCharacters: totalCharacters,
            estimatedUSD: estimatedUSD
        )
    }

    private func readEntries() -> [HistoryEntry] {
        guard let content = try? String(contentsOf: fileURL, encoding: .utf8) else {
            return []
        }
        return content
            .split(separator: "\n")
            .compactMap { line -> HistoryEntry? in
                guard let data = line.data(using: .utf8) else { return nil }
                return try? decoder.decode(HistoryEntry.self, from: data)
            }
    }

    func openInFinder() {
        NSWorkspace.shared.open(fileURL)
    }
}

final class RecorderService: NSObject, AVAudioRecorderDelegate {
    private var recorder: AVAudioRecorder?
    private var meterTimer: Timer?
    private var recordStart: Date?
    private var lastVoiceAt: Date?
    private var autoStopEnabled = true
    private var autoStopSeconds: Double = 1.2
    var onSilenceDetected: (() -> Void)?
    var onLevelUpdate: ((Float, TimeInterval) -> Void)?

    var isRecording: Bool {
        recorder?.isRecording ?? false
    }

    func requestPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return true
        case .notDetermined:
            return await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    continuation.resume(returning: granted)
                }
            }
        default:
            return false
        }
    }

    func startRecording(autoStopEnabled: Bool, autoStopSeconds: Double) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("VoiceInputRecordings", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let outputURL = directory.appendingPathComponent(UUID().uuidString).appendingPathExtension("wav")

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16_000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false
        ]

        let recorder = try AVAudioRecorder(url: outputURL, settings: settings)
        recorder.delegate = self
        recorder.isMeteringEnabled = true
        recorder.prepareToRecord()
        recorder.record()
        self.recorder = recorder
        self.recordStart = Date()
        self.lastVoiceAt = Date()
        self.autoStopEnabled = autoStopEnabled
        self.autoStopSeconds = autoStopSeconds
        onLevelUpdate?(0.04, 0)

        meterTimer?.invalidate()
        meterTimer = Timer.scheduledTimer(withTimeInterval: 0.08, repeats: true) { [weak self] _ in
            self?.checkMeters()
        }
    }

    func stopRecording() -> (url: URL, duration: TimeInterval)? {
        guard let recorder else { return nil }
        recorder.stop()
        meterTimer?.invalidate()
        meterTimer = nil
        let url = recorder.url
        let duration = Date().timeIntervalSince(recordStart ?? Date())
        self.recorder = nil
        onLevelUpdate?(0, duration)
        return (url, duration)
    }

    private func checkMeters() {
        guard let recorder else { return }
        recorder.updateMeters()
        let power = recorder.averagePower(forChannel: 0)
        let now = Date()
        let normalizedPower = max(0.06, min(1.0, (power + 52) / 52))
        let elapsed = now.timeIntervalSince(recordStart ?? now)
        onLevelUpdate?(normalizedPower, elapsed)
        if power > -45 {
            lastVoiceAt = now
        }

        guard autoStopEnabled,
              let start = recordStart,
              let lastVoiceAt else { return }

        if now.timeIntervalSince(start) >= 0.7,
           now.timeIntervalSince(lastVoiceAt) >= autoStopSeconds {
            onSilenceDetected?()
        }
    }
}

struct OpenAITranscriptionService {
    private struct TranscriptionResponse: Decodable {
        let text: String
    }

    func testConnection(apiKey: String) async throws {
        var request = URLRequest(url: URL(string: "https://api.openai.com/v1/models")!)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        let (_, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              200..<300 ~= httpResponse.statusCode else {
            throw NSError(domain: "OpenAITranscriptionService", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "OpenAIに接続できませんでした。"
            ])
        }
    }

    func transcribeAudio(fileURL: URL, mode: AppMode, apiKey: String) async throws -> String {
        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: URL(string: "https://api.openai.com/v1/audio/transcriptions")!)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = try makeMultipartBody(fileURL: fileURL, model: mode.openAIModelName, boundary: boundary)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw NSError(domain: "OpenAITranscriptionService", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "OpenAIから不正な応答が返りました。"
            ])
        }

        guard 200..<300 ~= httpResponse.statusCode else {
            let body = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "OpenAITranscriptionService", code: httpResponse.statusCode, userInfo: [
                NSLocalizedDescriptionKey: body
            ])
        }

        let decoded = try JSONDecoder().decode(TranscriptionResponse.self, from: data)
        return decoded.text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func makeMultipartBody(fileURL: URL, model: String, boundary: String) throws -> Data {
        let audioData = try Data(contentsOf: fileURL)
        var body = Data()

        func append(_ string: String) {
            body.append(Data(string.utf8))
        }

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"model\"\r\n\r\n")
        append("\(model)\r\n")

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"language\"\r\n\r\n")
        append("ja\r\n")

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(audioData)
        append("\r\n")
        append("--\(boundary)--\r\n")

        return body
    }
}

struct OpenAITextPolishService {
    private struct ChatCompletionResponse: Decodable {
        struct Choice: Decodable {
            struct Message: Decodable {
                let content: String
            }

            let message: Message
        }

        let choices: [Choice]
    }

    func polishJapaneseText(_ text: String, apiKey: String) async throws -> String {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return text
        }

        var request = URLRequest(url: URL(string: "https://api.openai.com/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let payload: [String: Any] = [
            "model": "gpt-4.1-mini",
            "temperature": 0.2,
            "messages": [
                [
                    "role": "system",
                    "content": "あなたは日本語の音声入力補正アシスタントです。元の意味を変えず、情報を足さず、句読点や文のつながりを自然に整えてください。出力は整えた本文だけにしてください。"
                ],
                [
                    "role": "user",
                    "content": text
                ]
            ]
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw NSError(domain: "OpenAITextPolishService", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "AI整形の応答を確認できませんでした。"
            ])
        }

        guard 200..<300 ~= httpResponse.statusCode else {
            let body = String(data: data, encoding: .utf8) ?? "AI整形に失敗しました。"
            throw NSError(domain: "OpenAITextPolishService", code: httpResponse.statusCode, userInfo: [
                NSLocalizedDescriptionKey: body
            ])
        }

        let decoded = try JSONDecoder().decode(ChatCompletionResponse.self, from: data)
        guard let content = decoded.choices.first?.message.content.trimmingCharacters(in: .whitespacesAndNewlines),
              !content.isEmpty else {
            throw NSError(domain: "OpenAITextPolishService", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "AI整形の結果が空でした。"
            ])
        }

        return content
    }

    func estimatedUSD(inputText: String, outputText: String) -> Double {
        let inputTokens = Double(inputText.count) * 1.2
        let outputTokens = Double(outputText.count) * 1.2
        return (inputTokens * 0.40 / 1_000_000) + (outputTokens * 1.60 / 1_000_000)
    }
}

final class SoundCuePlayer: NSObject, AVAudioPlayerDelegate {
    static let shared = SoundCuePlayer()

    private var activePlayers: [AVAudioPlayer] = []

    func playStartCue() {
        playCue(frequencies: [880, 1174], segmentDuration: 0.045)
    }

    func playStopCue() {
        playCue(frequencies: [988, 740], segmentDuration: 0.05)
    }

    private func playCue(frequencies: [Double], segmentDuration: Double) {
        guard let data = makeToneWAVData(frequencies: frequencies, segmentDuration: segmentDuration) else {
            NSSound.beep()
            return
        }

        do {
            let player = try AVAudioPlayer(data: data)
            player.delegate = self
            player.prepareToPlay()
            activePlayers.append(player)
            player.play()
        } catch {
            NSSound.beep()
        }
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        activePlayers.removeAll { $0 === player }
    }

    private func makeToneWAVData(frequencies: [Double], segmentDuration: Double) -> Data? {
        let sampleRate = 44_100
        let amplitude = 0.24
        let samplesPerSegment = Int(Double(sampleRate) * segmentDuration)
        var pcm = Data(capacity: frequencies.count * samplesPerSegment * 2)

        for frequency in frequencies {
            for sampleIndex in 0..<samplesPerSegment {
                let progress = Double(sampleIndex) / Double(samplesPerSegment)
                let envelope = min(progress / 0.08, 1.0) * min((1.0 - progress) / 0.12, 1.0)
                let theta = 2.0 * Double.pi * frequency * Double(sampleIndex) / Double(sampleRate)
                let value = sin(theta) * amplitude * max(0, envelope)
                var int16 = Int16(max(-1, min(1, value)) * Double(Int16.max))
                pcm.append(Data(bytes: &int16, count: MemoryLayout<Int16>.size))
            }
        }

        let byteRate = sampleRate * 2
        let blockAlign: UInt16 = 2
        let bitsPerSample: UInt16 = 16
        let dataSize = UInt32(pcm.count)
        let riffSize = UInt32(36) + dataSize

        var wav = Data()
        wav.append("RIFF".data(using: .ascii)!)
        wav.append(littleEndianBytes(riffSize))
        wav.append("WAVE".data(using: .ascii)!)
        wav.append("fmt ".data(using: .ascii)!)
        wav.append(littleEndianBytes(UInt32(16)))
        wav.append(littleEndianBytes(UInt16(1)))
        wav.append(littleEndianBytes(UInt16(1)))
        wav.append(littleEndianBytes(UInt32(sampleRate)))
        wav.append(littleEndianBytes(UInt32(byteRate)))
        wav.append(littleEndianBytes(blockAlign))
        wav.append(littleEndianBytes(bitsPerSample))
        wav.append("data".data(using: .ascii)!)
        wav.append(littleEndianBytes(dataSize))
        wav.append(pcm)
        return wav
    }

    private func littleEndianBytes<T: FixedWidthInteger>(_ value: T) -> Data {
        var littleEndianValue = value.littleEndian
        return Data(bytes: &littleEndianValue, count: MemoryLayout<T>.size)
    }
}

final class SystemAudioMuteService {
    private var previousMutedState: Bool?

    func muteSystemAudioForRecording() throws {
        guard previousMutedState == nil else { return }
        guard let currentMuted = readCurrentMutedState() else {
            throw NSError(domain: "SystemAudioMuteService", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "録音中ミュートの状態を確認できませんでした。"
            ])
        }
        previousMutedState = currentMuted
        guard !currentMuted else { return }
        guard runAppleScript("set volume output muted true") != nil else {
            previousMutedState = nil
            throw NSError(domain: "SystemAudioMuteService", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "録音中ミュートを有効にできませんでした。"
            ])
        }
    }

    func restoreSystemAudioAfterRecording() {
        guard let previousMutedState else { return }
        _ = runAppleScript("set volume output muted \(previousMutedState ? "true" : "false")")
        self.previousMutedState = nil
    }

    private func readCurrentMutedState() -> Bool? {
        guard let output = runAppleScript("output muted of (get volume settings)") else {
            return nil
        }
        let normalized = output.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized == "true" {
            return true
        }
        if normalized == "false" {
            return false
        }
        return nil
    }

    @discardableResult
    private func runAppleScript(_ source: String) -> String? {
        let process = Process()
        let outputPipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-e", source]
        process.standardOutput = outputPipe
        process.standardError = outputPipe

        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return nil
        }

        guard process.terminationStatus == 0 else {
            return nil
        }

        let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)
    }
}

struct UserFacingErrorTranslator {
    static func message(for error: Error) -> String {
        let nsError = error as NSError
        let rawMessage = nsError.localizedDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawMessage.isEmpty else {
            return "不明なエラーです。原因: 詳細を取得できませんでした。"
        }

        if containsJapanese(rawMessage) {
            return rawMessage
        }

        if nsError.domain == NSURLErrorDomain {
            return networkMessage(for: nsError)
        }

        let lowered = rawMessage.lowercased()

        if lowered.contains("invalid_api_key") || lowered.contains("incorrect api key") || lowered.contains("unauthorized") || nsError.code == 401 {
            return "OpenAI APIキーが正しくありません。原因: APIキーの入力ミス、無効化、または別アカウントのキーです。"
        }

        if lowered.contains("insufficient_quota") || lowered.contains("quota") || lowered.contains("billing") {
            return "OpenAI APIの利用上限に達しました。原因: クレジット不足、無料枠の終了、または使用量上限です。"
        }

        if lowered.contains("rate limit") || lowered.contains("too many requests") {
            return "OpenAI APIへのアクセスが多すぎます。原因: 短時間にリクエストが集中しています。少し待ってから再試行してください。"
        }

        if lowered.contains("model") && lowered.contains("not found") {
            return "AIモデルが見つかりません。原因: モデル名の変更、利用権限不足、または一時的な提供停止です。"
        }

        if lowered.contains("timeout") || lowered.contains("timed out") {
            return "通信がタイムアウトしました。原因: 回線が遅いか、OpenAI側の応答が遅れています。"
        }

        if lowered.contains("ssl") || lowered.contains("secure connection") || lowered.contains("certificate") {
            return "安全な通信に失敗しました。原因: ネットワーク設定または証明書の問題です。"
        }

        if lowered.contains("json") || lowered.contains("unexpected response") || lowered.contains("invalid response") {
            return "AIサービスの応答を正しく読めませんでした。原因: サービス側から想定外の形式で返ってきました。"
        }

        if lowered.contains("offline") || lowered.contains("network connection was lost") || lowered.contains("could not connect") {
            return "インターネットに接続できませんでした。原因: Wi-Fi切断、回線不調、またはOpenAI側への接続失敗です。"
        }

        return "エラーが発生しました。原因: サービス側で想定外の応答が返りました。"
    }

    private static func networkMessage(for error: NSError) -> String {
        switch error.code {
        case NSURLErrorNotConnectedToInternet, NSURLErrorInternationalRoamingOff, NSURLErrorDataNotAllowed:
            return "インターネットに接続できません。原因: Wi-Fiやモバイル回線が切れています。"
        case NSURLErrorTimedOut:
            return "通信がタイムアウトしました。原因: 回線が遅いか、OpenAI側の応答が遅れています。"
        case NSURLErrorCannotFindHost, NSURLErrorCannotConnectToHost, NSURLErrorDNSLookupFailed:
            return "接続先サーバーが見つかりません。原因: DNSやネットワーク設定の問題です。"
        case NSURLErrorSecureConnectionFailed, NSURLErrorServerCertificateHasBadDate, NSURLErrorServerCertificateUntrusted:
            return "安全な通信に失敗しました。原因: 証明書またはネットワークの問題です。"
        case NSURLErrorNetworkConnectionLost:
            return "通信が途中で切れました。原因: Wi-Fiや回線が不安定です。"
        default:
            return "通信エラーが発生しました。原因: ネットワークまたはOpenAI側の一時的な問題です。"
        }
    }

    private static func containsJapanese(_ text: String) -> Bool {
        text.range(of: "[ぁ-んァ-ン一-龥]", options: .regularExpression) != nil
    }
}

struct OfflineTranscriptionService {
    private struct OfflineResponse: Decodable {
        let text: String
    }

    func transcribeAudio(fileURL: URL) throws -> String {
        let pythonURL = URL(fileURLWithPath: BuildPaths.offlinePythonPath)
        let scriptURL = URL(fileURLWithPath: BuildPaths.offlineScriptPath)

        guard FileManager.default.fileExists(atPath: pythonURL.path) else {
            throw NSError(domain: "OfflineTranscriptionService", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "オフライン用のPython環境が見つかりません。"
            ])
        }

        guard FileManager.default.fileExists(atPath: scriptURL.path) else {
            throw NSError(domain: "OfflineTranscriptionService", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "offline_transcribe.py が見つかりません。"
            ])
        }

        let process = Process()
        let outputPipe = Pipe()
        process.executableURL = pythonURL
        process.arguments = [scriptURL.path, fileURL.path]
        process.standardOutput = outputPipe
        process.standardError = outputPipe
        try process.run()
        process.waitUntilExit()

        let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
        if process.terminationStatus != 0 {
            let body = String(data: data, encoding: .utf8) ?? "オフライン文字起こしに失敗しました。"
            throw NSError(domain: "OfflineTranscriptionService", code: Int(process.terminationStatus), userInfo: [
                NSLocalizedDescriptionKey: body
            ])
        }

        let response = try JSONDecoder().decode(OfflineResponse.self, from: data)
        return response.text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private func hotKeyHandler(
    _ nextHandler: EventHandlerCallRef?,
    _ event: EventRef?,
    _ userData: UnsafeMutableRawPointer?
) -> OSStatus {
    guard let userData else { return noErr }
    let center = Unmanaged<HotKeyCenter>.fromOpaque(userData).takeUnretainedValue()
    center.onHotKeyPressed?()
    return noErr
}

final class HotKeyCenter {
    static let shared = HotKeyCenter()

    var onHotKeyPressed: (() -> Void)?
    private var hotKeyRef: EventHotKeyRef?
    private var eventHandler: EventHandlerRef?

    private init() {
        var eventSpec = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: UInt32(kEventHotKeyPressed)
        )

        InstallEventHandler(
            GetApplicationEventTarget(),
            hotKeyHandler,
            1,
            &eventSpec,
            Unmanaged.passUnretained(self).toOpaque(),
            &eventHandler
        )
    }

    func register(shortcut: Shortcut) {
        unregister()

        let hotKeyID = EventHotKeyID(signature: OSType(0x564F4943), id: 1)
        RegisterEventHotKey(
            UInt32(shortcut.keyCode),
            shortcut.carbonModifiers,
            hotKeyID,
            GetApplicationEventTarget(),
            0,
            &hotKeyRef
        )
    }

    func unregister() {
        if let hotKeyRef {
            UnregisterEventHotKey(hotKeyRef)
            self.hotKeyRef = nil
        }
    }
}
