import AppKit
import SwiftUI

struct MainMicView: View {
    @ObservedObject var controller: VoiceInputAppController

    var body: some View {
        VStack(spacing: 16) {
            VStack(spacing: 6) {
                Text("Voice Input")
                    .font(.system(size: 24, weight: .bold))
                Text(controller.status.title)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
                Text(controller.settings.mode.title)
                    .font(.system(size: 11, weight: .semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(Color.secondary.opacity(0.12)))
            }

            Button(action: controller.toggleRecording) {
                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [controller.statusColor.opacity(0.95), controller.statusColor.opacity(0.6)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 120, height: 120)
                    Text(controller.micButtonTitle)
                        .font(.system(size: 28, weight: .heavy))
                        .foregroundStyle(.white)
                }
            }
            .buttonStyle(.plain)

            Text(controller.lastTranscript.isEmpty ? "ここに最後の文字起こし結果が表示されます" : controller.lastTranscript)
                .font(.system(size: 12))
                .frame(maxWidth: .infinity, minHeight: 96, alignment: .topLeading)
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color.black.opacity(0.06))
                )

            VStack(spacing: 10) {
                HStack {
                    ActionButton(title: "Paste Last", action: controller.pasteLastTranscript)
                    ActionButton(title: "Copy Last", action: controller.copyLastTranscript)
                }

                HStack {
                    ActionButton(title: "Open History", action: controller.openHistory)
                    ActionButton(title: "Settings") {
                        controller.showingSettings = true
                    }
                }
            }

            Text(controller.monthlyEstimateText)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
        }
        .padding(20)
        .frame(width: 340, height: 430)
        .background(WindowAccessor(alwaysOnTop: controller.settings.alwaysOnTop))
        .sheet(isPresented: $controller.showingSettings) {
            SettingsView(controller: controller)
        }
    }
}

struct ActionButton: View {
    let title: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 12, weight: .semibold))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
        }
        .buttonStyle(.borderedProminent)
    }
}

struct SettingsView: View {
    @ObservedObject var controller: VoiceInputAppController
    @State private var capturingShortcut = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Settings")
                .font(.system(size: 24, weight: .bold))

            GroupBox("Transcription Mode") {
                Picker("Mode", selection: $controller.settings.mode) {
                    ForEach(AppMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
            }

            GroupBox("OpenAI") {
                VStack(alignment: .leading, spacing: 10) {
                    SecureField("sk-...", text: $controller.apiKeyDraft)
                    HStack {
                        Button("Save API Key") {
                            controller.saveAPIKey()
                        }
                        Button("Test Connection") {
                            controller.testConnection()
                        }
                    }
                    if !controller.apiConnectionMessage.isEmpty {
                        Text(controller.apiConnectionMessage)
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                    }
                }
            }

            GroupBox("Shortcut") {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Record Shortcut: \(controller.settings.recordShortcut.displayString)")
                    Button(capturingShortcut ? "Press your shortcut..." : "Change Shortcut") {
                        capturingShortcut = true
                    }
                    if capturingShortcut {
                        ShortcutCaptureRepresentable { shortcut in
                            controller.updateRecordShortcut(shortcut)
                            capturingShortcut = false
                        }
                        .frame(height: 44)
                        .background(RoundedRectangle(cornerRadius: 10).fill(Color.black.opacity(0.08)))
                    }
                }
            }

            GroupBox("Behavior") {
                VStack(alignment: .leading, spacing: 10) {
                    Toggle("Auto paste after transcription", isOn: $controller.settings.autoPaste)
                    Toggle("Remove fillers", isOn: $controller.settings.fillerRemoval)
                    Toggle("Always on top", isOn: $controller.settings.alwaysOnTop)
                    Toggle("Auto stop on silence", isOn: $controller.settings.autoStopEnabled)
                    HStack {
                        Text("Silence stop seconds")
                        Slider(value: $controller.settings.autoStopSeconds, in: 0.8...3.0, step: 0.1)
                        Text(String(format: "%.1fs", controller.settings.autoStopSeconds))
                            .frame(width: 48)
                    }
                }
            }

            Text(controller.monthlyEstimateText)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)

            HStack {
                Spacer()
                Button("Close") {
                    controller.showingSettings = false
                }
            }
        }
        .padding(24)
        .frame(width: 520)
    }
}

struct ShortcutCaptureRepresentable: NSViewRepresentable {
    let onCapture: (Shortcut) -> Void

    func makeNSView(context: Context) -> ShortcutCaptureNSView {
        let view = ShortcutCaptureNSView()
        view.onCapture = onCapture
        DispatchQueue.main.async {
            view.window?.makeFirstResponder(view)
        }
        return view
    }

    func updateNSView(_ nsView: ShortcutCaptureNSView, context: Context) {
        nsView.onCapture = onCapture
        DispatchQueue.main.async {
            nsView.window?.makeFirstResponder(nsView)
        }
    }
}

final class ShortcutCaptureNSView: NSView {
    var onCapture: ((Shortcut) -> Void)?

    override var acceptsFirstResponder: Bool { true }

    override func draw(_ dirtyRect: NSRect) {
        NSColor.clear.setFill()
        dirtyRect.fill()
        let message = "Press shortcut now"
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 14, weight: .medium),
            .foregroundColor: NSColor.secondaryLabelColor
        ]
        let attributed = NSAttributedString(string: message, attributes: attrs)
        attributed.draw(at: NSPoint(x: 12, y: 12))
    }

    override func keyDown(with event: NSEvent) {
        let flags = event.modifierFlags.intersection([.command, .option, .control, .shift])
        guard !flags.isEmpty else { return }
        onCapture?(Shortcut(keyCode: UInt32(event.keyCode), modifiers: flags.rawValue))
    }
}

struct WindowAccessor: NSViewRepresentable {
    let alwaysOnTop: Bool

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            configureWindow(for: view)
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        DispatchQueue.main.async {
            configureWindow(for: nsView)
        }
    }

    private func configureWindow(for view: NSView) {
        guard let window = view.window else { return }
        window.level = alwaysOnTop ? .floating : .normal
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
    }
}
