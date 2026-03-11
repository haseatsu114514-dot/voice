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
        .frame(
            minWidth: windowSize.width,
            idealWidth: windowSize.width,
            minHeight: windowSize.height,
            idealHeight: windowSize.height
        )
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
    @State private var showingUtilities = false
    @State private var showingErrorDetails = false

    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                HStack(alignment: .center, spacing: 12) {
                    HStack(alignment: .center, spacing: 12) {
                        AppHeroBadge()
                        VStack(alignment: .leading, spacing: 6) {
                            Text("日本語音声入力")
                                .font(.system(size: 20, weight: .bold))
                            Text("録音して、AIで読みやすく整えて貼り付けます")
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(.secondary)
                        }
                    }

                    Spacer()

                    VStack(alignment: .trailing, spacing: 8) {
                        HeaderSettingsButton {
                            controller.showingSettings = true
                        }
                    }
                }

                HStack(alignment: .center) {
                    VStack(alignment: .leading, spacing: 4) {
                        ShortcutChip(text: controller.currentShortcutText)
                        HStack(spacing: 8) {
                            StatusPill(title: controller.status.title, color: controller.statusColor, compact: false)
                            StatusBadge(
                                title: controller.apiSetupStatusText,
                                tint: controller.apiStatusTint
                            )
                            CostChip(text: controller.monthlyCostJPYText)
                        }
                        Text("\(controller.typingBenchmarkText) / \(controller.savingsSummaryText)")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(.secondary)
                        Text(controller.dailySavingsSummaryText)
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    SmallIconButton(systemImage: "rectangle.compress.vertical") {
                        controller.toggleInterfaceMode()
                    }
                    .help("小型モードに切り替え")
                }

                HStack(spacing: 10) {
                    CaptureModeButton(
                        controller: controller,
                        captureMode: .aiPolish,
                        compact: false
                    )
                    ShortcutActionCard(
                        shortcut: controller.settings.recordShortcut.displayString,
                        tone: controller.settings.polishTone.title
                    )
                }

                Picker("AI文体", selection: $controller.settings.polishTone) {
                    ForEach(PolishTone.allCases) { tone in
                        Text(tone.title).tag(tone)
                    }
                }
                .pickerStyle(.segmented)

                Text("通常は自然な文章、会話は友達向けのタメ口です")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)

                if !controller.inlineErrorText.isEmpty {
                    OneLineErrorBanner(text: controller.inlineErrorText)
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
                        .frame(height: 36)

                        if controller.status.isError {
                            Text("問題が起きました。必要なときだけ詳細を見られます。")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                            DisclosureGroup("詳細を表示", isExpanded: $showingErrorDetails) {
                                Text(controller.statusDetailText)
                                    .font(.system(size: 11))
                                    .foregroundStyle(.secondary)
                                    .padding(.top, 4)
                            }
                            .font(.system(size: 11, weight: .semibold))
                            .tint(.secondary)
                        } else {
                            Text(controller.statusDetailText)
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                if !controller.lastTranscript.isEmpty {
                    GlassPanel {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("直前の文字起こし")
                                .font(.system(size: 13, weight: .semibold))

                            Text(controller.lastTranscript)
                                .font(.system(size: 12))
                                .frame(maxWidth: .infinity, minHeight: 72, alignment: .topLeading)
                                .padding(12)
                                .background(
                                    RoundedRectangle(cornerRadius: 14)
                                        .fill(Color.black.opacity(0.05))
                                )
                        }
                    }
                }

                DisclosureGroup("補助メニュー", isExpanded: $showingUtilities) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("\(controller.monthlyStats.shortSummaryText) / \(controller.monthlyCostJPYText)")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                        Text(controller.savingsSummaryText)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                        Text(controller.dailySavingsSummaryText)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)

                        if !controller.hasSavedAPIKey {
                            Text("この音声入力を使う前に、設定で APIキー を保存します。")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(.orange)
                        }

                        HStack(spacing: 8) {
                            UtilityButton(title: "履歴", systemImage: "folder.fill", action: controller.openHistory)
                            UtilityButton(title: "設定", systemImage: "slider.horizontal.3") {
                                controller.showingSettings = true
                            }
                        }
                    }
                    .padding(.top, 8)
                }
                .font(.system(size: 12, weight: .semibold))
                .tint(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 44)
            .padding(.horizontal, 18)
            .padding(.bottom, 18)
        }
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

            CaptureModeButton(
                controller: controller,
                captureMode: .aiPolish,
                compact: true
            )

            Text(controller.settings.recordShortcut.displayString + " / " + controller.settings.polishTone.title)
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(.secondary)

            Picker("AI文体", selection: $controller.settings.polishTone) {
                ForEach(PolishTone.allCases) { tone in
                    Text(tone.title).tag(tone)
                }
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            .font(.system(size: 9, weight: .semibold))

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
                Text(controller.apiSetupStatusText)
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(controller.apiStatusTint)
                Text(controller.monthlyCostJPYText)
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(controller.savingsSummaryText)
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(controller.dailySavingsSummaryText)
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(controller.monthlyStats.shortSummaryText)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.secondary)
                if !controller.inlineErrorText.isEmpty {
                    Text(controller.inlineErrorText)
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(.red)
                        .lineLimit(2)
                }
            }
            .multilineTextAlignment(.center)

            HStack(spacing: 8) {
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

struct AppHeroBadge: View {
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18)
                .fill(
                    LinearGradient(
                        colors: [
                            Color(red: 0.21, green: 0.54, blue: 1.0),
                            Color(red: 0.97, green: 0.73, blue: 0.38)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(width: 56, height: 56)

            VStack(spacing: 2) {
                Image(systemName: "mic.fill")
                    .font(.system(size: 18, weight: .bold))
                HStack(spacing: 2) {
                    Capsule().frame(width: 4, height: 10)
                    Capsule().frame(width: 4, height: 14)
                    Capsule().frame(width: 4, height: 10)
                }
                .foregroundStyle(.white.opacity(0.9))
            }
            .foregroundStyle(.white)
        }
    }
}

struct HeaderSettingsButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: "gearshape.fill")
                Text("設定")
            }
            .font(.system(size: 12, weight: .semibold))
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .buttonStyle(.borderedProminent)
    }
}

struct StatusBadge: View {
    let title: String
    let tint: Color

    var body: some View {
        Text(title)
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(tint)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Capsule().fill(tint.opacity(0.13)))
    }
}

struct CostChip: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Capsule().fill(Color.black.opacity(0.06)))
    }
}

struct ShortcutActionCard: View {
    let shortcut: String
    let tone: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("録音キー", systemImage: "keyboard")
                .font(.system(size: 11, weight: .semibold))
            Text(shortcut)
                .font(.system(size: 15, weight: .bold, design: .rounded))
            Text(tone)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .frame(width: 102, height: 98, alignment: .topLeading)
        .padding(12)
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

struct OneLineErrorBanner: View {
    let text: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 11, weight: .bold))
            Text(text)
                .font(.system(size: 11, weight: .medium))
                .lineLimit(2)
        }
        .foregroundStyle(.red)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.red.opacity(0.08))
        )
    }
}

struct SetupGuidePanel: View {
    @ObservedObject var controller: VoiceInputAppController

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 8) {
                Text("最初にやること")
                    .font(.system(size: 13, weight: .semibold))

                Text("1. 右上の「設定」を押す")
                    .font(.system(size: 11))
                Text("2. 「OpenAI API」で APIキーを保存する")
                    .font(.system(size: 11))
                Text("3. 「録音ショートカット」で好きなキーに変える")
                    .font(.system(size: 11))
                Text("4. 戻って「AIで整える」を押す")
                    .font(.system(size: 11))

                if !controller.hasSavedAPIKey {
                    Text("今は API が未設定なので、「AIで整える」を押すと設定画面が開きます。")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.orange)
                }
            }
        }
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
                    .frame(height: compact ? 54 : 98)

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
                            .font(.system(size: compact ? 14 : 20, weight: .bold))
                        Text(captureMode.title)
                            .font(.system(size: compact ? 11 : 17, weight: .heavy))
                        if !compact {
                            Text(captureMode.subtitle)
                                .font(.system(size: 10, weight: .medium))
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
    let benchmark: String

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
                Text(benchmark)
                    .font(.system(size: 10))
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

struct UtilityButton: View {
    let title: String
    let systemImage: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: systemImage)
                    .font(.system(size: 11, weight: .bold))
                Text(title)
                    .font(.system(size: 11, weight: .semibold))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
        }
        .buttonStyle(.bordered)
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
    @State private var showingAdvancedSettings = false
    @State private var showingUsageDetails = false
    @State private var showingAPIDetails = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("設定")
                            .font(.system(size: 22, weight: .bold))
                        Text("普段は上の項目だけ触れば十分です")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("閉じる") {
                        controller.showingSettings = false
                    }
                }

                GlassPanel {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(spacing: 8) {
                            ShortcutChip(text: controller.currentShortcutText)
                            StatusBadge(
                                title: controller.apiSetupStatusText,
                                tint: controller.apiStatusTint
                            )
                        }
                        Text("録音キーと API だけ設定すれば、すぐ使えます。")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
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

                GroupBox("OpenAI API") {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("現在: \(controller.apiSetupStatusText)")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(controller.apiStatusTint)
                        SecureField("sk-...", text: $controller.apiKeyDraft)
                        HStack {
                            Button("APIキーを保存") {
                                controller.saveAPIKey()
                            }
                            Button(controller.isTestingAPIConnection ? "接続確認中..." : "接続テスト") {
                                controller.testConnection()
                            }
                            .disabled(controller.isTestingAPIConnection)
                        }
                        Text(controller.apiConnectionSummaryText)
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                        if controller.shouldShowAPIDetailDisclosure {
                            DisclosureGroup("詳細を表示", isExpanded: $showingAPIDetails) {
                                Text(controller.apiConnectionMessage)
                                    .font(.system(size: 11))
                                    .foregroundStyle(.secondary)
                                    .padding(.top, 4)
                            }
                            .font(.system(size: 11, weight: .semibold))
                            .tint(.secondary)
                        }
                    }
                }

                GroupBox("よく使う設定") {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("録音キーでも AIで整える が起動します。")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                        Picker("AI文体", selection: $controller.settings.polishTone) {
                            ForEach(PolishTone.allCases) { tone in
                                Text(tone.title).tag(tone)
                            }
                        }
                        .pickerStyle(.segmented)
                        Picker("表示サイズ", selection: $controller.settings.interfaceMode) {
                            ForEach(InterfaceMode.allCases) { mode in
                                Text(mode.title).tag(mode)
                            }
                        }
                        .pickerStyle(.segmented)
                        Toggle("常に手前に表示", isOn: $controller.settings.alwaysOnTop)
                        Toggle("文字起こし後に自動で貼り付ける", isOn: $controller.settings.autoPaste)
                    }
                }

                DisclosureGroup("詳細設定", isExpanded: $showingAdvancedSettings) {
                    VStack(alignment: .leading, spacing: 14) {
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

                        GroupBox("録音の細かい動作") {
                            VStack(alignment: .leading, spacing: 10) {
                                Toggle("録音中はMacの再生音をミュートする", isOn: $controller.settings.muteSystemAudioWhileRecording)
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

                        DisclosureGroup("今月の使用量", isExpanded: $showingUsageDetails) {
                            UsagePanel(
                                metrics: controller.monthUsageDetails,
                                summary: controller.monthlyEstimateText,
                                benchmark: "\(controller.typingBenchmarkText) / \(controller.savingsSummaryText) / \(controller.dailySavingsSummaryText)"
                            )
                                .padding(.top, 8)
                        }
                        .font(.system(size: 12, weight: .semibold))
                        .tint(.secondary)
                    }
                    .padding(.top, 8)
                }
                .font(.system(size: 12, weight: .semibold))
                .tint(.secondary)

                if controller.status.isError {
                    DisclosureGroup("現在のエラー詳細") {
                        Text(controller.errorMessage)
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                            .padding(.top, 4)
                    }
                    .font(.system(size: 12, weight: .semibold))
                    .tint(.secondary)
                }
            }
            .padding(24)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(width: 520, height: 620)
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

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            configureWindow(for: view, coordinator: context.coordinator, forceInitialSize: true)
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        DispatchQueue.main.async {
            configureWindow(for: nsView, coordinator: context.coordinator, forceInitialSize: false)
        }
    }

    private func configureWindow(for view: NSView, coordinator: Coordinator, forceInitialSize: Bool) {
        guard let window = view.window else { return }
        window.level = alwaysOnTop ? .floating : .normal
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.isMovableByWindowBackground = true
        window.minSize = windowSize
        window.maxSize = CGSize(width: windowSize.width + 80, height: 900)

        if coordinator.lastAppliedWindowSize != windowSize {
            window.setContentSize(windowSize)
            coordinator.lastAppliedWindowSize = windowSize
            coordinator.didApplyInitialSize = true
            return
        }

        if forceInitialSize && !coordinator.didApplyInitialSize {
            let shouldResetWideWindow = window.frame.size.width > windowSize.width + 80
            let shouldResetShortWindow = window.frame.size.width < windowSize.width || window.frame.size.height < windowSize.height
            if shouldResetWideWindow || shouldResetShortWindow {
                window.setContentSize(windowSize)
            }
            coordinator.didApplyInitialSize = true
        } else if window.frame.size.width < windowSize.width || window.frame.size.height < windowSize.height {
            window.setContentSize(windowSize)
        }
    }

    final class Coordinator {
        var didApplyInitialSize = false
        var lastAppliedWindowSize: CGSize?
    }
}
