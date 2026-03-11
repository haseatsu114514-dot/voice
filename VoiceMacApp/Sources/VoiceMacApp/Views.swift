import AppKit
import SwiftUI

struct MainMicView: View {
    @ObservedObject var controller: VoiceInputAppController

    var body: some View {
        let windowSize = controller.settings.interfaceMode.windowSize

        Group {
            if controller.settings.interfaceMode == .standard {
                StandardMicView(controller: controller)
            } else {
                CompactMicView(controller: controller)
            }
        }
        .frame(width: windowSize.width, height: windowSize.height)
        .background(
            WindowAccessor(
                alwaysOnTop: controller.settings.alwaysOnTop,
                windowSize: windowSize
            )
        )
        .sheet(isPresented: $controller.showingSettings) {
            SettingsView(controller: controller)
        }
    }
}

struct StandardMicView: View {
    @ObservedObject var controller: VoiceInputAppController

    var body: some View {
        VStack(spacing: 14) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("音声入力")
                        .font(.system(size: 24, weight: .bold))
                    Text("ボタンかショートカットで、すぐに話し始められます")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                    ShortcutChip(text: controller.currentShortcutText)
                }

                Spacer()

                HStack(spacing: 8) {
                    SmallIconButton(systemImage: "rectangle.compress.vertical") {
                        controller.toggleInterfaceMode()
                    }
                    .help("小型モードに切り替え")

                    SmallIconButton(systemImage: "gearshape.fill") {
                        controller.showingSettings = true
                    }
                    .help("設定を開く")
                }
            }

            StatusCard(controller: controller)

            HStack(spacing: 10) {
                ForEach(CaptureMode.allCases) { captureMode in
                    CaptureModeButton(
                        controller: controller,
                        captureMode: captureMode,
                        compact: false
                    )
                }
            }

            GlassPanel {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text(controller.status.isListening ? "録音の波形" : "入力レベル")
                            .font(.system(size: 13, weight: .semibold))
                        Spacer()
                        if controller.status.isListening {
                            Text(controller.recordingElapsedText)
                                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }
                    }

                    WaveformView(
                        levels: controller.audioLevels,
                        tint: controller.statusColor,
                        isActive: controller.status.isListening
                    )
                    .frame(height: 42)

                    Text(controller.statusDetailText)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
            }

            GlassPanel {
                VStack(alignment: .leading, spacing: 8) {
                    Text("直前の文字起こし")
                        .font(.system(size: 13, weight: .semibold))
                    Text(controller.lastTranscript.isEmpty ? "ここに最後の文字起こし結果が表示されます" : controller.lastTranscript)
                        .font(.system(size: 12))
                        .frame(maxWidth: .infinity, minHeight: 84, alignment: .topLeading)
                        .padding(12)
                        .background(
                            RoundedRectangle(cornerRadius: 14)
                                .fill(Color.black.opacity(0.05))
                        )
                }
            }

            UsagePanel(metrics: controller.monthUsageDetails, summary: controller.monthlyEstimateText)

            VStack(spacing: 10) {
                HStack(spacing: 10) {
                    ActionButton(title: "前回を貼る", systemImage: "arrowshape.turn.up.right.fill", action: controller.pasteLastTranscript)
                    ActionButton(title: "前回をコピー", systemImage: "doc.on.doc.fill", action: controller.copyLastTranscript)
                }

                HStack(spacing: 10) {
                    ActionButton(title: "履歴を開く", systemImage: "folder.fill", action: controller.openHistory)
                    ActionButton(title: "設定", systemImage: "slider.horizontal.3") {
                        controller.showingSettings = true
                    }
                }
            }
        }
        .padding(.top, 44)
        .padding(.horizontal, 18)
        .padding(.bottom, 18)
        .background(WindowBackground())
    }
}

struct CompactMicView: View {
    @ObservedObject var controller: VoiceInputAppController

    var body: some View {
        VStack(spacing: 10) {
            HStack(spacing: 8) {
                StatusPill(title: controller.status.title, color: controller.statusColor, compact: true)
                Spacer()
                SmallIconButton(systemImage: "gearshape.fill") {
                    controller.showingSettings = true
                }
                .help("設定")
                SmallIconButton(systemImage: "rectangle.expand.vertical") {
                    controller.toggleInterfaceMode()
                }
                .help("標準サイズに戻す")
            }

            HStack(spacing: 8) {
                ForEach(CaptureMode.allCases) { captureMode in
                    CaptureModeButton(
                        controller: controller,
                        captureMode: captureMode,
                        compact: true
                    )
                }
            }

            WaveformView(
                levels: controller.audioLevels,
                tint: controller.statusColor,
                isActive: controller.status.isListening
            )
            .frame(height: 24)

            VStack(spacing: 4) {
                Text(controller.currentShortcutText)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.secondary)
                Text(controller.monthlyStats.shortSummaryText)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.secondary)
            }
            .multilineTextAlignment(.center)

            HStack(spacing: 8) {
                SmallIconButton(systemImage: "arrowshape.turn.up.right.fill", action: controller.pasteLastTranscript)
                    .help("前回を貼る")
                SmallIconButton(systemImage: "doc.on.doc.fill", action: controller.copyLastTranscript)
                    .help("前回をコピー")
                SmallIconButton(systemImage: "folder.fill", action: controller.openHistory)
                    .help("履歴を開く")
            }
        }
        .padding(.top, 38)
        .padding(.horizontal, 12)
        .padding(.bottom, 12)
        .background(WindowBackground())
    }
}

struct CaptureModeButton: View {
    @ObservedObject var controller: VoiceInputAppController
    let captureMode: CaptureMode
    let compact: Bool

    var body: some View {
        let isSelected = controller.isCaptureModeSelected(captureMode)
        let isRecordingThisMode = controller.activeCaptureMode == captureMode
        let buttonColor = captureMode == .aiPolish ? Color.orange : Color.blue

        Button(action: {
            controller.handleCaptureButton(captureMode)
        }) {
            ZStack {
                RoundedRectangle(cornerRadius: compact ? 16 : 22)
                    .fill(
                        LinearGradient(
                            colors: [
                                buttonColor.opacity(isSelected ? 0.95 : 0.72),
                                buttonColor.opacity(isSelected ? 0.72 : 0.52)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(height: compact ? 54 : 116)

                if isRecordingThisMode {
                    RoundedRectangle(cornerRadius: compact ? 16 : 22)
                        .stroke(Color.white.opacity(0.42), lineWidth: 3)
                        .padding(4)
                }

                if controller.status.isProcessing && isSelected {
                    ProgressView()
                        .progressViewStyle(.circular)
                        .tint(.white)
                        .scaleEffect(compact ? 0.9 : 1.1)
                } else {
                    VStack(spacing: 4) {
                        Image(systemName: captureMode.systemImage)
                            .font(.system(size: compact ? 14 : 22, weight: .bold))
                        Text(captureMode.title)
                            .font(.system(size: compact ? 11 : 18, weight: .heavy))
                        if !compact {
                            Text(captureMode.subtitle)
                                .font(.system(size: 11, weight: .medium))
                                .multilineTextAlignment(.center)
                        }
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8)
                }
            }
        }
        .buttonStyle(.plain)
        .disabled(controller.status.isProcessing)
    }
}

struct StatusCard: View {
    @ObservedObject var controller: VoiceInputAppController

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    StatusPill(title: controller.status.title, color: controller.statusColor, compact: false)
                    ModeChip(title: controller.settings.mode.title)
                    ModeChip(title: controller.selectedCaptureMode.title)
                    Spacer()
                }

                Text(controller.statusDetailText)
                    .font(.system(size: 12))
                    .foregroundStyle(controller.status.isError ? .red : .secondary)
            }
        }
    }
}

struct UsagePanel: View {
    let metrics: [UsageMetric]
    let summary: String

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 12) {
                Text("今月の使用量")
                    .font(.system(size: 13, weight: .semibold))

                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                    ForEach(metrics) { metric in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(metric.title)
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(.secondary)
                            Text(metric.value)
                                .font(.system(size: 16, weight: .bold))
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.black.opacity(0.05))
                        )
                    }
                }

                Text(summary)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
        }
    }
}

struct ActionButton: View {
    let title: String
    let systemImage: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: systemImage)
                    .font(.system(size: 12, weight: .bold))
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
        }
        .buttonStyle(.borderedProminent)
    }
}

struct SmallIconButton: View {
    let systemImage: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 12, weight: .bold))
                .frame(width: 28, height: 28)
                .background(
                    Circle()
                        .fill(Color.white.opacity(0.7))
                )
        }
        .buttonStyle(.plain)
    }
}

struct ShortcutChip: View {
    let text: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "keyboard")
                .font(.system(size: 11, weight: .bold))
            Text(text)
                .font(.system(size: 11, weight: .semibold))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Capsule().fill(Color.black.opacity(0.06)))
    }
}

struct StatusPill: View {
    let title: String
    let color: Color
    let compact: Bool

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: compact ? 8 : 10, height: compact ? 8 : 10)
            Text(title)
                .font(.system(size: compact ? 10 : 11, weight: .semibold))
        }
        .padding(.horizontal, compact ? 8 : 10)
        .padding(.vertical, compact ? 5 : 6)
        .background(Capsule().fill(color.opacity(0.14)))
    }
}

struct ModeChip: View {
    let title: String

    var body: some View {
        Text(title)
            .font(.system(size: 11, weight: .semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Capsule().fill(Color.black.opacity(0.06)))
    }
}

struct WaveformView: View {
    let levels: [Double]
    let tint: Color
    let isActive: Bool

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 14)
                .fill(tint.opacity(isActive ? 0.10 : 0.05))

            HStack(alignment: .center, spacing: 4) {
                ForEach(Array(levels.enumerated()), id: \.offset) { _, level in
                    Capsule()
                        .fill(
                            LinearGradient(
                                colors: [tint.opacity(isActive ? 0.98 : 0.55), tint.opacity(isActive ? 0.62 : 0.30)],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                        .frame(width: 7, height: max(10, CGFloat(level) * 52))
                        .shadow(color: tint.opacity(isActive ? 0.28 : 0.10), radius: 5, y: 1)
                        .animation(.easeOut(duration: 0.1), value: level)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
        }
        .frame(maxWidth: .infinity, alignment: .center)
    }
}

struct GlassPanel<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 18)
                    .fill(Color.white.opacity(0.72))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18)
                    .stroke(Color.white.opacity(0.65), lineWidth: 1)
            )
    }
}

struct WindowBackground: View {
    var body: some View {
        LinearGradient(
            colors: [
                Color(red: 0.93, green: 0.96, blue: 1.0),
                Color(red: 0.98, green: 0.98, blue: 0.95)
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
    }
}

struct SettingsView: View {
    @ObservedObject var controller: VoiceInputAppController
    @State private var capturingShortcut = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("設定")
                        .font(.system(size: 24, weight: .bold))
                    Text(controller.currentShortcutText)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("閉じる") {
                    controller.showingSettings = false
                }
            }

            GroupBox("文字起こしモード") {
                VStack(alignment: .leading, spacing: 10) {
                    Picker("モード", selection: $controller.settings.mode) {
                        ForEach(AppMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)

                    Text("オフラインはPC内で処理、標準と高精度はOpenAI APIを使います。")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
            }

            GroupBox("録音ボタン") {
                VStack(alignment: .leading, spacing: 10) {
                    Picker("通常の録音ボタン", selection: $controller.settings.defaultCaptureMode) {
                        ForEach(CaptureMode.allCases) { captureMode in
                            Text(captureMode.title).tag(captureMode)
                        }
                    }
                    .pickerStyle(.segmented)
                    Text("ショートカットで開始するときは、このモードが使われます。")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
            }

            GroupBox("OpenAI API") {
                VStack(alignment: .leading, spacing: 10) {
                    SecureField("sk-...", text: $controller.apiKeyDraft)
                    HStack {
                        Button("APIキーを保存") {
                            controller.saveAPIKey()
                        }
                        Button("接続テスト") {
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

            GroupBox("録音ショートカット") {
                VStack(alignment: .leading, spacing: 10) {
                    Text("現在: \(controller.settings.recordShortcut.displayString)")
                    Text("このキーで録音開始と停止を切り替えます。")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                    Button(capturingShortcut ? "今、キーを押してください..." : "ショートカットを変更") {
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

            GroupBox("表示") {
                VStack(alignment: .leading, spacing: 10) {
                    Picker("表示サイズ", selection: $controller.settings.interfaceMode) {
                        ForEach(InterfaceMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    Toggle("常に手前に表示", isOn: $controller.settings.alwaysOnTop)
                }
            }

            GroupBox("動作") {
                VStack(alignment: .leading, spacing: 10) {
                    Toggle("文字起こし後に自動で貼り付ける", isOn: $controller.settings.autoPaste)
                    Toggle("開始音と終了音を鳴らす", isOn: $controller.settings.soundCuesEnabled)
                    Toggle("フィラーを自動で減らす", isOn: $controller.settings.fillerRemoval)
                    Toggle("無音で自動停止する", isOn: $controller.settings.autoStopEnabled)
                    HStack {
                        Text("無音で止めるまで")
                        Slider(value: $controller.settings.autoStopSeconds, in: 0.8...3.0, step: 0.1)
                        Text(String(format: "%.1f秒", controller.settings.autoStopSeconds))
                            .frame(width: 52)
                    }
                }
            }

            UsagePanel(metrics: controller.monthUsageDetails, summary: controller.monthlyEstimateText)
        }
        .padding(24)
        .frame(width: 560)
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
        let message = "ショートカットを押してください"
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
    let windowSize: CGSize

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
        window.isMovableByWindowBackground = true
        window.setContentSize(windowSize)
    }
}
