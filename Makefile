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
# RE-USE weights are downloaded under their licenses). Not committed.
golden:
	uv run reference/golden.py
	uv run reference/golden_dfn3.py
	uv run reference/golden_nemotron.py

# Menu bar app with the virtual microphone driver (needs XcodeGen). Signs with Developer ID
# by default; `make app SIGN=-` signs ad hoc for local use.
APP_DERIVED := .build/app
APP := $(APP_DERIVED)/Build/Products/Release/Noise Reduction.app
SIGN ?= Developer ID Application
# Release builds pass the tag and a build number; the app reinstalls the driver when
# the build number changes.
VERSION_FLAGS := $(if $(VERSION),MARKETING_VERSION=$(VERSION)) $(if $(BUILD),CURRENT_PROJECT_VERSION=$(BUILD))

app:
	cd App && xcodegen generate --quiet
	xcodebuild -project App/NoiseReduction.xcodeproj -scheme NoiseReduction -configuration Release \
		-destination 'platform=macOS,arch=arm64' -derivedDataPath $(APP_DERIVED) \
		SIGN_IDENTITY="$(SIGN)" $(VERSION_FLAGS) build -quiet
	@echo "built $(APP)"

# Notarized zip for a GitHub release. Authenticates with the keychain profile made by
# `xcrun notarytool store-credentials notary`; CI passes an API key in NOTARY_AUTH instead.
NOTARY_AUTH ?= --keychain-profile notary
DIST := .build/NoiseReduction.zip

dist: app
	ditto -c -k --keepParent "$(APP)" $(DIST)
	xcrun notarytool submit $(DIST) $(NOTARY_AUTH) --wait
	xcrun stapler staple "$(APP)"
	spctl --assess --type execute --verbose=2 "$(APP)"
	ditto -c -k --keepParent "$(APP)" $(DIST)
	@echo "notarized $(DIST)"
