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

    var settings: SettingsStore

    private let keychain = KeychainService()
    private let historyStore = HistoryStore()
    private let recorder = RecorderService()
    private let openAI = OpenAITranscriptionService()
    private let offline = OfflineTranscriptionService()
    private var cancellables: Set<AnyCancellable> = []

    init(settings: SettingsStore = SettingsStore()) {
        self.settings = settings
        self.apiKeyDraft = keychain.load()

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

    var micButtonTitle: String {
        switch status {
        case .listening:
            return "STOP"
        case .processing:
            return "WAIT"
        default:
            return "MIC"
        }
    }

    var monthlyEstimateText: String {
        let minutes = historyStore.currentMonthDurationSeconds() / 60
        let estimate = minutes * settings.mode.usdCostPerMinute
        return String(format: "This month: %.1f min / about $%.2f", minutes, estimate)
    }

    func toggleRecording() {
        if recorder.isRecording {
            stopRecording(trigger: "manual")
            return
        }

        Task {
            let granted = await recorder.requestPermission()
            guard granted else {
                status = .error("Microphone permission is required.")
                errorMessage = "Microphone permission is required."
                return
            }

            do {
                try recorder.startRecording(
                    autoStopEnabled: settings.autoStopEnabled,
                    autoStopSeconds: settings.autoStopSeconds
                )
                status = .listening
                errorMessage = ""
            } catch {
                status = .error(error.localizedDescription)
                errorMessage = error.localizedDescription
            }
        }
    }

    func stopRecording(trigger: String) {
        guard recorder.isRecording,
              let result = recorder.stopRecording() else {
            return
        }

        status = .processing

        Task {
            await transcribe(fileURL: result.url, duration: result.duration, trigger: trigger)
        }
    }

    func saveAPIKey() {
        do {
            try keychain.save(apiKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines))
            apiConnectionMessage = "API key saved."
        } catch {
            apiConnectionMessage = error.localizedDescription
        }
    }

    func testConnection() {
        let key = apiKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else {
            apiConnectionMessage = "Enter your OpenAI API key first."
            return
        }

        Task {
            do {
                try await openAI.testConnection(apiKey: key)
                apiConnectionMessage = "Connected to OpenAI."
            } catch {
                apiConnectionMessage = error.localizedDescription
            }
        }
    }

    func updateRecordShortcut(_ shortcut: Shortcut) {
        settings.recordShortcut = shortcut
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

    private func transcribe(fileURL: URL, duration: TimeInterval, trigger: String) async {
        defer { try? FileManager.default.removeItem(at: fileURL) }

        do {
            let text: String
            switch settings.mode {
            case .offline:
                text = try offline.transcribeAudio(fileURL: fileURL)
            case .balanced, .best:
                let key = apiKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !key.isEmpty else {
                    throw NSError(domain: "VoiceInputMacApp", code: 1, userInfo: [
                        NSLocalizedDescriptionKey: "OpenAI API key is not set."
                    ])
                }
                text = try await openAI.transcribeAudio(fileURL: fileURL, mode: settings.mode, apiKey: key)
            }

            let normalized = normalize(text)
            lastTranscript = normalized
            status = .idle
            errorMessage = ""

            try? historyStore.append(
                HistoryEntry(
                    timestamp: Date(),
                    mode: settings.mode.rawValue,
                    provider: settings.mode.providerLabel,
                    durationSeconds: duration,
                    text: normalized,
                    success: true,
                    errorMessage: nil
                )
            )

            if settings.autoPaste {
                paste(text: normalized)
            }
        } catch {
            status = .error(error.localizedDescription)
            errorMessage = error.localizedDescription
            try? historyStore.append(
                HistoryEntry(
                    timestamp: Date(),
                    mode: settings.mode.rawValue,
                    provider: settings.mode.providerLabel,
                    durationSeconds: duration,
                    text: lastTranscript,
                    success: false,
                    errorMessage: error.localizedDescription
                )
            )
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
}
