// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "EfficientNoiseReduction",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "EfficientNoiseReduction", targets: ["EfficientNoiseReduction"]),
        .executable(name: "noise-reduction", targets: ["NoiseReductionCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.33.0"),
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "EfficientNoiseReduction",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ComplexModule", package: "swift-numerics"),
            ],
            resources: [.copy("Resources/DeepFilterNet3.safetensors")]
        ),
        .executableTarget(
            name: "NoiseReductionCLI",
            dependencies: [
                "EfficientNoiseReduction",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "EfficientNoiseReductionTests",
            dependencies: ["EfficientNoiseReduction"]
        ),
    ]
)
