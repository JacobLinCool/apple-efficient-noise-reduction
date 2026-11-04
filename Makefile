# MLX's Metal shaders only build through Xcode's build system, so everything goes
# through xcodebuild rather than `swift build` / `swift test`.
DERIVED := .build/xcode
XCB := xcodebuild -destination 'platform=macOS,arch=arm64' -derivedDataPath $(DERIVED)
CLI := $(DERIVED)/Build/Products/Release/noise-reduction

.PHONY: build test bench golden

build:
	$(XCB) -scheme noise-reduction -configuration Release build -quiet
	@echo "built $(CLI)"

test:
	$(XCB) -scheme EfficientNoiseReduction-Package test -quiet

bench: build
	$(CLI) bench --sample-rate 16000
	$(CLI) bench --sample-rate 48000

# Golden test data from NVIDIA's and DeepFilterNet's reference code (needs uv; the
# RE-USE weights are downloaded under NVIDIA's noncommercial license). Not committed.
golden:
	uv run reference/golden.py
	uv run reference/golden_dfn3.py
