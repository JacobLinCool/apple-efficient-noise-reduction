# MLX's Metal shaders only build through Xcode's build system, so everything goes
# through xcodebuild rather than `swift build` / `swift test`.
DERIVED := .build/xcode
XCB := xcodebuild -destination 'platform=macOS,arch=arm64' -derivedDataPath $(DERIVED)
CLI := $(DERIVED)/Build/Products/Release/noise-reduction

.PHONY: build test bench golden app dist

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

# Menu bar app with the virtual microphone driver (needs XcodeGen). Signs with Developer ID
# by default; `make app SIGN=-` signs ad hoc for local use.
APP_DERIVED := .build/app
APP := $(APP_DERIVED)/Build/Products/Release/Noise Reduction.app
SIGN ?= Developer ID Application

app:
	cd App && xcodegen generate --quiet
	xcodebuild -project App/NoiseReduction.xcodeproj -scheme NoiseReduction -configuration Release \
		-destination 'platform=macOS,arch=arm64' -derivedDataPath $(APP_DERIVED) \
		SIGN_IDENTITY="$(SIGN)" build -quiet
	@echo "built $(APP)"

# Notarized zip for a GitHub release; needs `xcrun notarytool store-credentials $(NOTARY_PROFILE)`.
NOTARY_PROFILE ?= notary
DIST := .build/NoiseReduction.zip

dist: app
	ditto -c -k --keepParent "$(APP)" $(DIST)
	xcrun notarytool submit $(DIST) --keychain-profile $(NOTARY_PROFILE) --wait
	xcrun stapler staple "$(APP)"
	ditto -c -k --keepParent "$(APP)" $(DIST)
	@echo "notarized $(DIST)"
