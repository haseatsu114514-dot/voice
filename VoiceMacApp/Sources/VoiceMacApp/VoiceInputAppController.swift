import AppKit
import ApplicationServices
import Combine
import Foundation
import SwiftUI

enum APIConnectionState: Equatable {
    case missing
    case saved
    case testing
    case verified
    case failed(String)
}

@MainActor
final class VoiceInputAppController: ObservableObject {
    @Published var status: RecorderStatus = .idle
    @Published var lastTranscript: String = ""
    @Published var errorMessage: String = ""
    @Published var showingSettings: Bool = false
    @Published var apiKeyDraft: String = ""
    @Published var apiConnectionMessage: String = ""
    @Published var apiConnectionState: APIConnectionState = .missing
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
    private let systemAudioMuteService = SystemAudioMuteService()
    private let typingBenchmark = TypingBenchmark.sushiDaAverage
    private var cancellables: Set<AnyCancellable> = []
    private var lastExternalApplication: NSRunningApplication?

    init(settings: SettingsStore = SettingsStore()) {
        self.settings = settings
        self.settings.defaultCaptureMode = .aiPolish
        self.apiKeyDraft = keychain.load()
        self.apiConnectionState = apiKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? .missing : .saved
        self.monthlyStats = historyStore.currentMonthStats()

        settings.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)

        NSWorkspace.shared.notificationCenter
            .publisher(for: NSWorkspace.didActivateApplicationNotification)
            .compactMap { $0.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication }
            .sink { [weak self] application in
                self?.rememberExternalApplicationIfNeeded(application)
            }
            .store(in: &cancellables)

        if let frontmostApplication = NSWorkspace.shared.frontmostApplication {
            rememberExternalApplicationIfNeeded(frontmostApplication)
        }

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
            format: "今月 %.1f分 / %d回 / 約%d円",
            monthlyStats.totalMinutes,
            monthlyStats.totalSessions,
            estimatedMonthlyJPY
        )
    }

    var monthlyCostJPYText: String {
        "今月 約\(estimatedMonthlyJPY)円"
    }

    var estimatedMonthlyJPY: Int {
        Int((monthlyStats.estimatedUSD * 160).rounded())
    }

    var estimatedManualTypingSeconds: Double {
        typingBenchmark.estimatedTypingSeconds(forCharacterCount: monthlyStats.totalCharacters)
    }

    var savedTimeSeconds: Double {
        max(0, estimatedManualTypingSeconds - monthlyStats.successfulDurationSeconds)
    }

    var savedTimeText: String {
        formatDuration(savedTimeSeconds)
    }

    var elapsedMonthDays: Int {
        max(1, Calendar.current.component(.day, from: Date()))
    }

    var dailySavedTimeSeconds: Double {
        savedTimeSeconds / Double(elapsedMonthDays)
    }

    var dailySavedTimeText: String {
        formatDuration(dailySavedTimeSeconds)
    }

    var manualTypingTimeText: String {
        formatDuration(estimatedManualTypingSeconds)
    }

    var savingsSummaryText: String {
        "今月 \(savedTimeText) 短縮"
    }

    var dailySavingsSummaryText: String {
        "1日あたり \(dailySavedTimeText)"
    }

    var typingBenchmarkText: String {
        "手打ち 125字/分換算"
    }

    var currentShortcutText: String {
        "録音キー: \(settings.recordShortcut.displayString)"
    }

    var hasSavedAPIKey: Bool {
        !apiKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var apiSetupStatusText: String {
        switch apiConnectionState {
        case .missing:
            return "API未設定"
        case .saved:
            return "API保存済み"
        case .testing:
            return "接続確認中"
        case .verified:
            return "接続OK"
        case .failed:
            return "接続エラー"
        }
    }

    var apiStatusTint: Color {
        switch apiConnectionState {
        case .missing:
            return .orange
        case .saved:
            return .blue
        case .testing:
            return .orange
        case .verified:
            return .green
        case .failed:
            return .red
        }
    }

    var apiConnectionSummaryText: String {
        switch apiConnectionState {
        case .missing:
            return "APIキーがまだありません。保存してから接続テストを押します。"
        case .saved:
            return "APIキーは保存済みです。まだ接続テストはしていません。"
        case .testing:
            return "OpenAIに接続して確認しています..."
        case .verified:
            return "接続テストが成功しました。AIで整えるを使えます。"
        case .failed:
            return "接続テストに失敗しました。必要なときだけ詳細を開いて確認できます。"
        }
    }

    var isTestingAPIConnection: Bool {
        if case .testing = apiConnectionState {
            return true
        }
        return false
    }

    var shouldShowAPIDetailDisclosure: Bool {
        if case .failed = apiConnectionState {
            return true
        }
        return false
    }

    var recordingElapsedText: String {
        let minutes = Int(recordingElapsedSeconds) / 60
        let seconds = Int(recordingElapsedSeconds) % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }

    var selectedCaptureMode: CaptureMode {
        activeCaptureMode ?? .aiPolish
    }

    var statusDetailText: String {
        switch status {
        case .idle:
            return "\(selectedCaptureMode.title) で待機中です"
        case .listening:
            if selectedCaptureMode == .aiPolish {
                return "録音後にAIで読みやすく整えます"
            }
            return "AIを使わず、通常入力として高速に文字へ変換します"
        case .processing:
            if selectedCaptureMode == .aiPolish {
                return "文字起こしとAI整形を進めています"
            }
            return "通常入力として文字起こししています"
        case .error(let message):
            return message
        }
    }

    var inlineErrorText: String {
        errorMessage.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var monthUsageDetails: [UsageMetric] {
        [
            UsageMetric(title: "録音時間", value: monthlyStats.durationText),
            UsageMetric(title: "手打ち換算", value: manualTypingTimeText),
            UsageMetric(title: "節約時間", value: savedTimeText),
            UsageMetric(title: "1日あたり", value: dailySavedTimeText),
            UsageMetric(title: "成功回数", value: "\(monthlyStats.successfulSessions)回"),
            UsageMetric(title: "文字数", value: "\(monthlyStats.totalCharacters)字"),
            UsageMetric(title: "概算費用", value: "\(estimatedMonthlyJPY)円")
        ]
    }

    func toggleRecording() {
        if recorder.isRecording {
            stopRecording(trigger: "manual")
            return
        }
        guard ensureCaptureModeIsReady(.aiPolish) else { return }
        startRecording(captureMode: .aiPolish)
    }

    func handleCaptureButton(_ captureMode: CaptureMode) {
        if recorder.isRecording {
            stopRecording(trigger: "manual")
            return
        }
        guard ensureCaptureModeIsReady(captureMode) else { return }
        startRecording(captureMode: captureMode)
    }

    func isCaptureModeSelected(_ captureMode: CaptureMode) -> Bool {
        selectedCaptureMode == captureMode
    }

    func saveAPIKey() {
        let trimmedKey = apiKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        do {
            try keychain.save(trimmedKey)
            apiKeyDraft = trimmedKey
            apiConnectionState = trimmedKey.isEmpty ? .missing : .saved
            apiConnectionMessage = ""
        } catch {
            let localizedError = localized(error)
            apiConnectionState = .failed(localizedError)
            apiConnectionMessage = localizedError
        }
    }

    func testConnection() {
        let key = apiKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else {
            apiConnectionState = .missing
            apiConnectionMessage = ""
            return
        }

        apiConnectionState = .testing
        apiConnectionMessage = ""
        Task {
            do {
                try await openAI.testConnection(apiKey: key)
                apiConnectionState = .verified
                apiConnectionMessage = ""
            } catch {
                let localizedError = localized(error)
                apiConnectionState = .failed(localizedError)
                apiConnectionMessage = localizedError
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
        settings.defaultCaptureMode = .aiPolish
        activeCaptureMode = captureMode
        if let frontmostApplication = NSWorkspace.shared.frontmostApplication {
            rememberExternalApplicationIfNeeded(frontmostApplication)
        }

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
                if settings.muteSystemAudioWhileRecording {
                    do {
                        try systemAudioMuteService.muteSystemAudioForRecording()
                    } catch {
                        errorMessage = localized(error)
                    }
                }
                if settings.soundCuesEnabled {
                    soundCuePlayer.playStartCue()
                }
            } catch {
                let localizedError = localized(error)
                status = .error(localizedError)
                errorMessage = localizedError
                activeCaptureMode = nil
                systemAudioMuteService.restoreSystemAudioAfterRecording()
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
        systemAudioMuteService.restoreSystemAudioAfterRecording()
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
                let polished = try await polisher.polishJapaneseText(
                    normalized,
                    tone: settings.polishTone,
                    apiKey: key
                )
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
            let localizedError = localized(error)
            status = .error(localizedError)
            errorMessage = localizedError
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
                    errorMessage: localizedError,
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

        guard AXIsProcessTrusted() else {
            status = .error("自動貼り付けにはアクセシビリティ権限が必要です。")
            errorMessage = "システム設定 > プライバシーとセキュリティ > アクセシビリティ で Voice Input.app を許可してください。"
            return
        }

        let targetApplication = resolvedPasteTargetApplication()
        targetApplication?.activate(options: [.activateIgnoringOtherApps])

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.18) { [weak self] in
            self?.postCommandV()
        }
    }

    private func postCommandV() {
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

    private func resolvedPasteTargetApplication() -> NSRunningApplication? {
        if let lastExternalApplication, !lastExternalApplication.isTerminated {
            return lastExternalApplication
        }
        if let frontmostApplication = NSWorkspace.shared.frontmostApplication {
            rememberExternalApplicationIfNeeded(frontmostApplication)
        }
        return lastExternalApplication
    }

    private func rememberExternalApplicationIfNeeded(_ application: NSRunningApplication) {
        guard application.bundleIdentifier != Bundle.main.bundleIdentifier else { return }
        lastExternalApplication = application
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

    private func ensureCaptureModeIsReady(_ captureMode: CaptureMode) -> Bool {
        guard captureMode == .aiPolish else { return true }
        guard hasSavedAPIKey else {
            showingSettings = true
            apiConnectionState = .missing
            apiConnectionMessage = ""
            status = .error("AIで整えるにはAPI設定が必要です。")
            errorMessage = "AIで整えるにはAPI設定が必要です。"
            return false
        }
        return true
    }

    private func localized(_ error: Error) -> String {
        UserFacingErrorTranslator.message(for: error)
    }

    private func formatDuration(_ seconds: Double) -> String {
        if seconds >= 3600 {
            return String(format: "%.1f時間", seconds / 3600)
        }
        if seconds >= 60 {
            return String(format: "%.1f分", seconds / 60)
        }
        return String(format: "%.0f秒", seconds)
    }
}

struct UsageMetric: Identifiable {
    let id = UUID()
    let title: String
    let value: String
}
