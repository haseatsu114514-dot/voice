import AppKit
import Combine
import Foundation
import SwiftUI

@MainActor
final class VoiceInputAppController: ObservableObject {
    @Published var status: RecorderStatus = .idle
    @Published var lastTranscript: String = ""
    @Published var errorMessage: String = ""
    @Published var showingSettings: Bool = false
    @Published var apiKeyDraft: String = ""
    @Published var apiConnectionMessage: String = ""
    @Published var audioLevels: [Double] = Array(repeating: 0.08, count: 14)
    @Published var recordingElapsedSeconds: TimeInterval = 0
    @Published var monthlyStats: MonthlyUsageStats = .empty
    @Published var activeCaptureMode: CaptureMode? = nil

    var settings: SettingsStore

    private let keychain = KeychainService()
    private let historyStore = HistoryStore()
    private let recorder = RecorderService()
    private let openAI = OpenAITranscriptionService()
    private let polisher = OpenAITextPolishService()
    private let offline = OfflineTranscriptionService()
    private let soundCuePlayer = SoundCuePlayer.shared
    private var cancellables: Set<AnyCancellable> = []

    init(settings: SettingsStore = SettingsStore()) {
        self.settings = settings
        self.apiKeyDraft = keychain.load()
        self.monthlyStats = historyStore.currentMonthStats()

        settings.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)

        settings.$recordShortcut
            .sink { shortcut in
                HotKeyCenter.shared.register(shortcut: shortcut)
            }
            .store(in: &cancellables)

        HotKeyCenter.shared.onHotKeyPressed = { [weak self] in
            DispatchQueue.main.async {
                self?.toggleRecording()
            }
        }

        recorder.onSilenceDetected = { [weak self] in
            DispatchQueue.main.async {
                self?.stopRecording(trigger: "silence")
            }
        }

        recorder.onLevelUpdate = { [weak self] level, elapsed in
            DispatchQueue.main.async {
                self?.recordingElapsedSeconds = elapsed
                self?.pushAudioLevel(Double(level))
            }
        }
    }

    var statusColor: Color {
        switch status {
        case .idle:
            return .blue
        case .listening:
            return .red
        case .processing:
            return .orange
        case .error:
            return .purple
        }
    }

    var monthlyEstimateText: String {
        guard monthlyStats.totalSessions > 0 else {
            return "今月はまだ使用していません"
        }
        return String(
            format: "今月 %.1f分 / %d回 / 約$%.2f",
            monthlyStats.totalMinutes,
            monthlyStats.totalSessions,
            monthlyStats.estimatedUSD
        )
    }

    var currentShortcutText: String {
        "録音キー: \(settings.recordShortcut.displayString) / \(settings.defaultCaptureMode.title)"
    }

    var recordingElapsedText: String {
        let minutes = Int(recordingElapsedSeconds) / 60
        let seconds = Int(recordingElapsedSeconds) % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }

    var selectedCaptureMode: CaptureMode {
        activeCaptureMode ?? settings.defaultCaptureMode
    }

    var statusDetailText: String {
        switch status {
        case .idle:
            return "\(selectedCaptureMode.title) で待機中です"
        case .listening:
            if selectedCaptureMode == .aiPolish {
                return "録音後にAIで読みやすく整えます"
            }
            return "AIを使わず、そのまま高速に文字へ変換します"
        case .processing:
            if selectedCaptureMode == .aiPolish {
                return "文字起こしとAI整形を進めています"
            }
            return "そのままの文章として文字起こししています"
        case .error(let message):
            return message
        }
    }

    var monthUsageDetails: [UsageMetric] {
        [
            UsageMetric(title: "録音時間", value: monthlyStats.durationText),
            UsageMetric(title: "成功回数", value: "\(monthlyStats.successfulSessions)回"),
            UsageMetric(title: "文字数", value: "\(monthlyStats.totalCharacters)字"),
            UsageMetric(title: "概算費用", value: String(format: "$%.2f", monthlyStats.estimatedUSD))
        ]
    }

    func toggleRecording() {
        if recorder.isRecording {
            stopRecording(trigger: "manual")
            return
        }
        startRecording(captureMode: settings.defaultCaptureMode)
    }

    func handleCaptureButton(_ captureMode: CaptureMode) {
        if recorder.isRecording {
            stopRecording(trigger: "manual")
            return
        }
        startRecording(captureMode: captureMode)
    }

    func isCaptureModeSelected(_ captureMode: CaptureMode) -> Bool {
        selectedCaptureMode == captureMode
    }

    func saveAPIKey() {
        do {
            try keychain.save(apiKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines))
            apiConnectionMessage = "APIキーを保存しました。"
        } catch {
            apiConnectionMessage = error.localizedDescription
        }
    }

    func testConnection() {
        let key = apiKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else {
            apiConnectionMessage = "先にOpenAI APIキーを入力してください。"
            return
        }

        Task {
            do {
                try await openAI.testConnection(apiKey: key)
                apiConnectionMessage = "OpenAIへの接続を確認しました。"
            } catch {
                apiConnectionMessage = error.localizedDescription
            }
        }
    }

    func updateRecordShortcut(_ shortcut: Shortcut) {
        settings.recordShortcut = shortcut
    }

    func toggleInterfaceMode() {
        settings.interfaceMode = settings.interfaceMode == .standard ? .compact : .standard
    }

    func copyLastTranscript() {
        guard !lastTranscript.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(lastTranscript, forType: .string)
    }

    func pasteLastTranscript() {
        guard !lastTranscript.isEmpty else { return }
        paste(text: lastTranscript)
    }

    func openHistory() {
        historyStore.openInFinder()
    }

    private func startRecording(captureMode: CaptureMode) {
        settings.defaultCaptureMode = captureMode
        activeCaptureMode = captureMode

        Task {
            let granted = await recorder.requestPermission()
            guard granted else {
                status = .error("マイク権限が必要です。")
                errorMessage = "マイク権限が必要です。"
                activeCaptureMode = nil
                return
            }

            do {
                try recorder.startRecording(
                    autoStopEnabled: settings.autoStopEnabled,
                    autoStopSeconds: settings.autoStopSeconds
                )
                status = .listening
                errorMessage = ""
                recordingElapsedSeconds = 0
                audioLevels = Array(repeating: 0.08, count: 14)
                if settings.soundCuesEnabled {
                    soundCuePlayer.playStartCue()
                }
            } catch {
                status = .error(error.localizedDescription)
                errorMessage = error.localizedDescription
                activeCaptureMode = nil
            }
        }
    }

    func stopRecording(trigger: String) {
        guard recorder.isRecording,
              let result = recorder.stopRecording() else {
            return
        }

        let captureMode = activeCaptureMode ?? settings.defaultCaptureMode
        status = .processing
        audioLevels = Array(repeating: 0.12, count: 14)
        if settings.soundCuesEnabled {
            soundCuePlayer.playStopCue()
        }

        Task {
            await transcribe(fileURL: result.url, duration: result.duration, trigger: trigger, captureMode: captureMode)
        }
    }

    private func transcribe(fileURL: URL, duration: TimeInterval, trigger: String, captureMode: CaptureMode) async {
        defer { try? FileManager.default.removeItem(at: fileURL) }

        do {
            let transcriptionMode = captureMode == .fastRaw ? AppMode.offline : settings.mode
            let baseText = try await transcribeBaseText(fileURL: fileURL, transcriptionMode: transcriptionMode)
            let normalized = normalize(baseText)
            let finalText: String
            var estimatedUSD = duration / 60 * transcriptionMode.usdCostPerMinute

            switch captureMode {
            case .fastRaw:
                finalText = normalized
            case .aiPolish:
                let key = apiKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !key.isEmpty else {
                    throw NSError(domain: "VoiceInputMacApp", code: 2, userInfo: [
                        NSLocalizedDescriptionKey: "AI整形にはOpenAI APIキーが必要です。"
                    ])
                }
                let polished = try await polisher.polishJapaneseText(normalized, apiKey: key)
                finalText = normalize(polished)
                estimatedUSD += polisher.estimatedUSD(inputText: normalized, outputText: finalText)
            }

            lastTranscript = finalText
            status = .idle
            errorMessage = ""
            audioLevels = Array(repeating: 0.08, count: 14)
            activeCaptureMode = nil

            try? historyStore.append(
                HistoryEntry(
                    timestamp: Date(),
                    mode: transcriptionMode.rawValue,
                    captureMode: captureMode.rawValue,
                    provider: transcriptionMode.providerLabel,
                    durationSeconds: duration,
                    text: finalText,
                    success: true,
                    errorMessage: nil,
                    estimatedUSD: estimatedUSD
                )
            )
            refreshMonthlyStats()

            if settings.autoPaste {
                paste(text: finalText)
            }
        } catch {
            status = .error(error.localizedDescription)
            errorMessage = error.localizedDescription
            audioLevels = Array(repeating: 0.08, count: 14)
            activeCaptureMode = nil
            try? historyStore.append(
                HistoryEntry(
                    timestamp: Date(),
                    mode: settings.mode.rawValue,
                    captureMode: captureMode.rawValue,
                    provider: settings.mode.providerLabel,
                    durationSeconds: duration,
                    text: lastTranscript,
                    success: false,
                    errorMessage: error.localizedDescription,
                    estimatedUSD: nil
                )
            )
            refreshMonthlyStats()
        }
    }

    private func transcribeBaseText(fileURL: URL, transcriptionMode: AppMode) async throws -> String {
        switch transcriptionMode {
        case .offline:
            return try offline.transcribeAudio(fileURL: fileURL)
        case .balanced, .best:
            let key = apiKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty else {
                throw NSError(domain: "VoiceInputMacApp", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "OpenAI APIキーが設定されていません。"
                ])
            }
            return try await openAI.transcribeAudio(fileURL: fileURL, mode: transcriptionMode, apiKey: key)
        }
    }

    private func normalize(_ text: String) -> String {
        var output = text
        if settings.fillerRemoval {
            ["えーと", "えっと", "あの", "その", "うーん", "えー"].forEach {
                output = output.replacingOccurrences(of: $0, with: "")
            }
        }
        output = output.replacingOccurrences(of: "\n\n\n", with: "\n\n")
        output = output.replacingOccurrences(of: "  ", with: " ")
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func paste(text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)

        let source = CGEventSource(stateID: .hidSystemState)
        let commandDown = CGEvent(keyboardEventSource: source, virtualKey: 0x37, keyDown: true)
        commandDown?.flags = .maskCommand
        let vDown = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: true)
        vDown?.flags = .maskCommand
        let vUp = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: false)
        vUp?.flags = .maskCommand
        let commandUp = CGEvent(keyboardEventSource: source, virtualKey: 0x37, keyDown: false)

        commandDown?.post(tap: .cghidEventTap)
        vDown?.post(tap: .cghidEventTap)
        vUp?.post(tap: .cghidEventTap)
        commandUp?.post(tap: .cghidEventTap)
    }

    private func pushAudioLevel(_ level: Double) {
        let normalized = max(0.03, min(1.0, level))
        if audioLevels.count >= 14 {
            audioLevels.removeFirst()
        }
        audioLevels.append(normalized)
    }

    private func refreshMonthlyStats() {
        monthlyStats = historyStore.currentMonthStats()
    }
}

struct UsageMetric: Identifiable {
    let id = UUID()
    let title: String
    let value: String
}
