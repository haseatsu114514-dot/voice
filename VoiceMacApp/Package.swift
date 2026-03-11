// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "VoiceMacApp",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "VoiceMacApp", targets: ["VoiceMacApp"])
    ],
    targets: [
        .executableTarget(
            name: "VoiceMacApp"
        )
    ]
)
