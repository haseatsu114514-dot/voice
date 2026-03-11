import SwiftUI

@main
struct VoiceInputMacApp: App {
    @StateObject private var controller = VoiceInputAppController()

    var body: some Scene {
        WindowGroup {
            MainMicView(controller: controller)
        }
        .defaultSize(width: 360, height: 560)
        .windowResizability(.automatic)
    }
}
