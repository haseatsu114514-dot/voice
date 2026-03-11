import AppKit
import Carbon
import Foundation

enum AppMode: String, CaseIterable, Identifiable, Codable {
    case offline
    case balanced
    case best

    var id: String { rawValue }

    var title: String {
        switch self {
        case .offline:
            return "Offline"
        case .balanced:
            return "Balanced"
        case .best:
            return "Best"
        }
    }

    var providerLabel: String {
        switch self {
        case .offline:
            return "Local"
        case .balanced, .best:
            return "OpenAI"
        }
    }

    var openAIModelName: String {
        switch self {
        case .offline:
            return ""
        case .balanced:
            return "gpt-4o-mini-transcribe"
        case .best:
            return "gpt-4o-transcribe"
        }
    }

    var usdCostPerMinute: Double {
        switch self {
        case .offline:
            return 0
        case .balanced:
            return 0.003
        case .best:
            return 0.006
        }
    }
}

enum RecorderStatus: Equatable {
    case idle
    case listening
    case processing
    case error(String)

    var title: String {
        switch self {
        case .idle:
            return "Ready"
        case .listening:
            return "Listening"
        case .processing:
            return "Processing"
        case .error:
            return "Error"
        }
    }
}

struct Shortcut: Codable, Equatable, Hashable {
    var keyCode: UInt32
    var modifiers: UInt

    static let defaultRecord = Shortcut(
        keyCode: 49,
        modifiers: NSEvent.ModifierFlags.command.union(.shift).rawValue
    )

    var modifierFlags: NSEvent.ModifierFlags {
        NSEvent.ModifierFlags(rawValue: modifiers)
    }

    var carbonModifiers: UInt32 {
        var result: UInt32 = 0
        let flags = modifierFlags
        if flags.contains(.command) { result |= UInt32(cmdKey) }
        if flags.contains(.option) { result |= UInt32(optionKey) }
        if flags.contains(.control) { result |= UInt32(controlKey) }
        if flags.contains(.shift) { result |= UInt32(shiftKey) }
        return result
    }

    var displayString: String {
        var parts: [String] = []
        let flags = modifierFlags
        if flags.contains(.command) { parts.append("Cmd") }
        if flags.contains(.option) { parts.append("Option") }
        if flags.contains(.control) { parts.append("Ctrl") }
        if flags.contains(.shift) { parts.append("Shift") }
        parts.append(Self.keyName(for: keyCode))
        return parts.joined(separator: "+")
    }

    private static func keyName(for keyCode: UInt32) -> String {
        switch keyCode {
        case 0: return "A"
        case 1: return "S"
        case 2: return "D"
        case 3: return "F"
        case 4: return "H"
        case 5: return "G"
        case 6: return "Z"
        case 7: return "X"
        case 8: return "C"
        case 9: return "V"
        case 11: return "B"
        case 12: return "Q"
        case 13: return "W"
        case 14: return "E"
        case 15: return "R"
        case 16: return "Y"
        case 17: return "T"
        case 18: return "1"
        case 19: return "2"
        case 20: return "3"
        case 21: return "4"
        case 22: return "6"
        case 23: return "5"
        case 24: return "="
        case 25: return "9"
        case 26: return "7"
        case 27: return "-"
        case 28: return "8"
        case 29: return "0"
        case 30: return "]"
        case 31: return "O"
        case 32: return "U"
        case 33: return "["
        case 34: return "I"
        case 35: return "P"
        case 36: return "Return"
        case 37: return "L"
        case 38: return "J"
        case 39: return "'"
        case 40: return "K"
        case 41: return ";"
        case 42: return "\\"
        case 43: return ","
        case 44: return "/"
        case 45: return "N"
        case 46: return "M"
        case 47: return "."
        case 49: return "Space"
        case 51: return "Delete"
        case 53: return "Esc"
        case 122: return "F1"
        case 120: return "F2"
        case 99: return "F3"
        case 118: return "F4"
        case 96: return "F5"
        case 97: return "F6"
        case 98: return "F7"
        case 100: return "F8"
        case 101: return "F9"
        case 109: return "F10"
        case 103: return "F11"
        case 111: return "F12"
        default:
            return "Key \(keyCode)"
        }
    }
}

enum BuildPaths {
    static let repoRoot = "/Users/hasegawaatsuki/Documents/New project/voice"
    static let offlinePythonPath = "\(repoRoot)/.venv/bin/python"
    static let offlineScriptPath = "\(repoRoot)/offline_transcribe.py"
}
