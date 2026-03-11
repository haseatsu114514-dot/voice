import AppKit
import AVFoundation
import Carbon
import Combine
import Foundation
import Security

final class SettingsStore: ObservableObject {
    @Published var mode: AppMode { didSet { save() } }
    @Published var recordShortcut: Shortcut { didSet { save() } }
    @Published var autoPaste: Bool { didSet { save() } }
    @Published var autoStopEnabled: Bool { didSet { save() } }
    @Published var autoStopSeconds: Double { didSet { save() } }
    @Published var fillerRemoval: Bool { didSet { save() } }
    @Published var alwaysOnTop: Bool { didSet { save() } }

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

        self.autoPaste = defaults.object(forKey: "settings.autoPaste") as? Bool ?? true
        self.autoStopEnabled = defaults.object(forKey: "settings.autoStopEnabled") as? Bool ?? true
        self.autoStopSeconds = defaults.object(forKey: "settings.autoStopSeconds") as? Double ?? 1.2
        self.fillerRemoval = defaults.object(forKey: "settings.fillerRemoval") as? Bool ?? true
        self.alwaysOnTop = defaults.object(forKey: "settings.alwaysOnTop") as? Bool ?? true
    }

    private func save() {
        defaults.set(mode.rawValue, forKey: "settings.mode")
        defaults.set(autoPaste, forKey: "settings.autoPaste")
        defaults.set(autoStopEnabled, forKey: "settings.autoStopEnabled")
        defaults.set(autoStopSeconds, forKey: "settings.autoStopSeconds")
        defaults.set(fillerRemoval, forKey: "settings.fillerRemoval")
        defaults.set(alwaysOnTop, forKey: "settings.alwaysOnTop")
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
                NSLocalizedDescriptionKey: "Failed to save API key."
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
    let provider: String
    let durationSeconds: Double
    let text: String
    let success: Bool
    let errorMessage: String?
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
        guard let content = try? String(contentsOf: fileURL, encoding: .utf8) else {
            return 0
        }
        let calendar = Calendar.current
        return content
            .split(separator: "\n")
            .compactMap { line -> HistoryEntry? in
                guard let data = line.data(using: .utf8) else { return nil }
                return try? decoder.decode(HistoryEntry.self, from: data)
            }
            .filter { calendar.isDate($0.timestamp, equalTo: Date(), toGranularity: .month) }
            .reduce(0) { $0 + $1.durationSeconds }
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

        meterTimer?.invalidate()
        meterTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
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
        return (url, duration)
    }

    private func checkMeters() {
        guard let recorder else { return }
        recorder.updateMeters()
        let power = recorder.averagePower(forChannel: 0)
        let now = Date()
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
                NSLocalizedDescriptionKey: "Failed to connect to OpenAI."
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
                NSLocalizedDescriptionKey: "Invalid response from OpenAI."
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

struct OfflineTranscriptionService {
    private struct OfflineResponse: Decodable {
        let text: String
    }

    func transcribeAudio(fileURL: URL) throws -> String {
        let pythonURL = URL(fileURLWithPath: BuildPaths.offlinePythonPath)
        let scriptURL = URL(fileURLWithPath: BuildPaths.offlineScriptPath)

        guard FileManager.default.fileExists(atPath: pythonURL.path) else {
            throw NSError(domain: "OfflineTranscriptionService", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Offline Python environment was not found."
            ])
        }

        guard FileManager.default.fileExists(atPath: scriptURL.path) else {
            throw NSError(domain: "OfflineTranscriptionService", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "offline_transcribe.py was not found."
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
            let body = String(data: data, encoding: .utf8) ?? "Offline transcription failed."
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
