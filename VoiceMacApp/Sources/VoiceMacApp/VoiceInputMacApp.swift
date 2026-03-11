import SwiftUI

@main
struct VoiceInputMacApp: App {
    @StateObject private var controller = VoiceInputAppController()

    var body: some Scene {
        WindowGroup {
            MainMicView(controller: controller)
        }
        .windowResizability(.automatic)
    }
}
