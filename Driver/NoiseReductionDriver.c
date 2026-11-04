// Virtual microphone for the Noise Reduction app: an Audio Server plug-in with two devices
// on one clock.
//
// - "Noise Reduction Microphone" (visible, input only) is what meeting apps select.
// - "Noise Reduction Sink" (hidden, output only) is where the app plays the enhanced audio.
//
// Both devices share one timeline, so a sample the app writes at output sample time t is
// read by the microphone at input sample time t, with no drift between them. Samples the
// app has not written (app stopped, or not yet started) read as silence.

#include <CoreAudio/AudioServerPlugIn.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>

#define kSampleRate 48000.0
#define kChannels 1
#define kZeroTimeStampPeriod 16384u
// Must be a power of two; holds about 1.4 s.
#define kRingFrames 65536

enum {
    kObjectPlugIn = kAudioObjectPlugInObject,
    kObjectMic = 2,
    kObjectMicStream = 3,
    kObjectSink = 4,
    kObjectSinkStream = 5,
};

#define kManufacturer CFSTR("Efficient Noise Reduction")
#define kMicName CFSTR("Noise Reduction Microphone")
#define kMicUID CFSTR("EfficientNoiseReduction_Microphone")
#define kSinkName CFSTR("Noise Reduction Sink")
#define kSinkUID CFSTR("EfficientNoiseReduction_Sink")
#define kModelUID CFSTR("EfficientNoiseReduction_Model")

static AudioServerPlugInHostRef gHost = NULL;
static pthread_mutex_t gStateMutex = PTHREAD_MUTEX_INITIALIZER;
static UInt32 gRefCount = 0;

// IO clients per device, and the timeline shared by both devices while either runs.
static UInt32 gRunning[2] = {0, 0};  // mic, sink
static UInt64 gAnchorHostTime = 0;
static Float64 gHostTicksPerFrame = 0;

static float gRing[kRingFrames];
// One past the last sample time the sink has written; samples in
// [gWrittenEnd - kRingFrames, gWrittenEnd) are valid.
static _Atomic SInt64 gWrittenEnd = 0;

static Boolean IsDevice(AudioObjectID id) { return id == kObjectMic || id == kObjectSink; }
static Boolean IsStream(AudioObjectID id) { return id == kObjectMicStream || id == kObjectSinkStream; }
static int DeviceIndex(AudioObjectID id) { return id == kObjectMic ? 0 : 1; }

static AudioStreamBasicDescription Format(void) {
    AudioStreamBasicDescription f = {0};
    f.mSampleRate = kSampleRate;
    f.mFormatID = kAudioFormatLinearPCM;
    f.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagsNativeEndian | kAudioFormatFlagIsPacked;
    f.mBytesPerPacket = 4 * kChannels;
    f.mFramesPerPacket = 1;
    f.mBytesPerFrame = 4 * kChannels;
    f.mChannelsPerFrame = kChannels;
    f.mBitsPerChannel = 32;
    return f;
}

#pragma mark - COM

static HRESULT QueryInterface(void *inDriver, REFIID inUUID, LPVOID *outInterface);
static ULONG AddRef(void *inDriver);
static ULONG Release(void *inDriver);
static OSStatus Initialize(AudioServerPlugInDriverRef inDriver, AudioServerPlugInHostRef inHost);
static OSStatus CreateDevice(AudioServerPlugInDriverRef, CFDictionaryRef, const AudioServerPlugInClientInfo *, AudioObjectID *);
static OSStatus DestroyDevice(AudioServerPlugInDriverRef, AudioObjectID);
static OSStatus AddDeviceClient(AudioServerPlugInDriverRef, AudioObjectID, const AudioServerPlugInClientInfo *);
static OSStatus RemoveDeviceClient(AudioServerPlugInDriverRef, AudioObjectID, const AudioServerPlugInClientInfo *);
static OSStatus PerformDeviceConfigurationChange(AudioServerPlugInDriverRef, AudioObjectID, UInt64, void *);
static OSStatus AbortDeviceConfigurationChange(AudioServerPlugInDriverRef, AudioObjectID, UInt64, void *);
static Boolean HasProperty(AudioServerPlugInDriverRef, AudioObjectID, pid_t, const AudioObjectPropertyAddress *);
static OSStatus IsPropertySettable(AudioServerPlugInDriverRef, AudioObjectID, pid_t, const AudioObjectPropertyAddress *, Boolean *);
static OSStatus GetPropertyDataSize(AudioServerPlugInDriverRef, AudioObjectID, pid_t, const AudioObjectPropertyAddress *, UInt32, const void *, UInt32 *);
static OSStatus GetPropertyData(AudioServerPlugInDriverRef, AudioObjectID, pid_t, const AudioObjectPropertyAddress *, UInt32, const void *, UInt32, UInt32 *, void *);
static OSStatus SetPropertyData(AudioServerPlugInDriverRef, AudioObjectID, pid_t, const AudioObjectPropertyAddress *, UInt32, const void *, UInt32, const void *);
static OSStatus StartIO(AudioServerPlugInDriverRef, AudioObjectID, UInt32);
static OSStatus StopIO(AudioServerPlugInDriverRef, AudioObjectID, UInt32);
static OSStatus GetZeroTimeStamp(AudioServerPlugInDriverRef, AudioObjectID, UInt32, Float64 *, UInt64 *, UInt64 *);
static OSStatus WillDoIOOperation(AudioServerPlugInDriverRef, AudioObjectID, UInt32, UInt32, Boolean *, Boolean *);
static OSStatus BeginIOOperation(AudioServerPlugInDriverRef, AudioObjectID, UInt32, UInt32, UInt32, const AudioServerPlugInIOCycleInfo *);
static OSStatus DoIOOperation(AudioServerPlugInDriverRef, AudioObjectID, AudioObjectID, UInt32, UInt32, UInt32, const AudioServerPlugInIOCycleInfo *, void *, void *);
static OSStatus EndIOOperation(AudioServerPlugInDriverRef, AudioObjectID, UInt32, UInt32, UInt32, const AudioServerPlugInIOCycleInfo *);

static AudioServerPlugInDriverInterface gInterface = {
    NULL, QueryInterface, AddRef, Release, Initialize, CreateDevice, DestroyDevice, AddDeviceClient,
    RemoveDeviceClient, PerformDeviceConfigurationChange, AbortDeviceConfigurationChange, HasProperty,
    IsPropertySettable, GetPropertyDataSize, GetPropertyData, SetPropertyData, StartIO, StopIO,
    GetZeroTimeStamp, WillDoIOOperation, BeginIOOperation, DoIOOperation, EndIOOperation,
};
static AudioServerPlugInDriverInterface *gInterfacePtr = &gInterface;
static AudioServerPlugInDriverRef gDriver = &gInterfacePtr;

void *NoiseReductionDriverCreate(CFAllocatorRef allocator, CFUUIDRef requestedTypeUUID) {
    (void)allocator;
    return CFEqual(requestedTypeUUID, kAudioServerPlugInTypeUUID) ? gDriver : NULL;
}

static HRESULT QueryInterface(void *inDriver, REFIID inUUID, LPVOID *outInterface) {
    if (inDriver != gDriver || outInterface == NULL) return kAudioHardwareBadObjectError;
    CFUUIDRef requested = CFUUIDCreateFromUUIDBytes(NULL, inUUID);
    Boolean match = CFEqual(requested, IUnknownUUID) || CFEqual(requested, kAudioServerPlugInDriverInterfaceUUID);
    CFRelease(requested);
    if (!match) return E_NOINTERFACE;
    pthread_mutex_lock(&gStateMutex);
    gRefCount++;
    pthread_mutex_unlock(&gStateMutex);
    *outInterface = gDriver;
    return S_OK;
}

static ULONG AddRef(void *inDriver) {
    if (inDriver != gDriver) return 0;
    pthread_mutex_lock(&gStateMutex);
    ULONG count = ++gRefCount;
    pthread_mutex_unlock(&gStateMutex);
    return count;
}

static ULONG Release(void *inDriver) {
    if (inDriver != gDriver) return 0;
    pthread_mutex_lock(&gStateMutex);
    if (gRefCount > 0) gRefCount--;
    ULONG count = gRefCount;
    pthread_mutex_unlock(&gStateMutex);
    return count;
}

#pragma mark - Basic operations

static OSStatus Initialize(AudioServerPlugInDriverRef inDriver, AudioServerPlugInHostRef inHost) {
    if (inDriver != gDriver) return kAudioHardwareBadObjectError;
    gHost = inHost;
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    Float64 ticksPerSecond = 1e9 * (Float64)timebase.denom / (Float64)timebase.numer;
    gHostTicksPerFrame = ticksPerSecond / kSampleRate;
    return 0;
}

static OSStatus CreateDevice(AudioServerPlugInDriverRef d, CFDictionaryRef desc, const AudioServerPlugInClientInfo *c, AudioObjectID *out) {
    (void)d; (void)desc; (void)c; (void)out;
    return kAudioHardwareUnsupportedOperationError;
}

static OSStatus DestroyDevice(AudioServerPlugInDriverRef d, AudioObjectID id) {
    (void)d; (void)id;
    return kAudioHardwareUnsupportedOperationError;
}

static OSStatus AddDeviceClient(AudioServerPlugInDriverRef d, AudioObjectID id, const AudioServerPlugInClientInfo *c) {
    (void)c;
    return d == gDriver && IsDevice(id) ? 0 : kAudioHardwareBadObjectError;
}

static OSStatus RemoveDeviceClient(AudioServerPlugInDriverRef d, AudioObjectID id, const AudioServerPlugInClientInfo *c) {
    (void)c;
    return d == gDriver && IsDevice(id) ? 0 : kAudioHardwareBadObjectError;
}

static OSStatus PerformDeviceConfigurationChange(AudioServerPlugInDriverRef d, AudioObjectID id, UInt64 a, void *i) {
    (void)a; (void)i;
    return d == gDriver && IsDevice(id) ? 0 : kAudioHardwareBadObjectError;
}

static OSStatus AbortDeviceConfigurationChange(AudioServerPlugInDriverRef d, AudioObjectID id, UInt64 a, void *i) {
    (void)a; (void)i;
    return d == gDriver && IsDevice(id) ? 0 : kAudioHardwareBadObjectError;
}

#pragma mark - Properties

// Writes one value of type T into outData, checking the caller's buffer.
#define RETURN_VALUE(T, value)                                       \
    do {                                                             \
        if (inDataSize < sizeof(T)) return kAudioHardwareBadPropertySizeError; \
        *(T *)outData = (value);                                     \
        *outDataSize = sizeof(T);                                    \
        return 0;                                                    \
    } while (0)

// Writes as many of `count` object IDs as fit.
static OSStatus ReturnIDs(const AudioObjectID *ids, UInt32 count, UInt32 inDataSize, UInt32 *outDataSize, void *outData) {
    UInt32 n = inDataSize / sizeof(AudioObjectID);
    if (n > count) n = count;
    memcpy(outData, ids, n * sizeof(AudioObjectID));
    *outDataSize = n * sizeof(AudioObjectID);
    return 0;
}

// The streams of a device visible in `scope`.
static UInt32 DeviceStreams(AudioObjectID device, AudioObjectPropertyScope scope, AudioObjectID *out) {
    Boolean isMic = device == kObjectMic;
    Boolean visible = scope == kAudioObjectPropertyScopeGlobal || (isMic && scope == kAudioObjectPropertyScopeInput) ||
                      (!isMic && scope == kAudioObjectPropertyScopeOutput);
    if (!visible) return 0;
    out[0] = isMic ? kObjectMicStream : kObjectSinkStream;
    return 1;
}

static Boolean HasProperty(AudioServerPlugInDriverRef inDriver, AudioObjectID id, pid_t pid, const AudioObjectPropertyAddress *a) {
    (void)pid;
    if (inDriver != gDriver || a == NULL) return false;
    switch (a->mSelector) {
    case kAudioObjectPropertyBaseClass:
    case kAudioObjectPropertyClass:
    case kAudioObjectPropertyOwner:
    case kAudioObjectPropertyOwnedObjects:
        return id == kObjectPlugIn || IsDevice(id) || IsStream(id);
    case kAudioObjectPropertyManufacturer:
        return id == kObjectPlugIn || IsDevice(id);
    case kAudioDevicePropertyLatency:  // same selector as kAudioStreamPropertyLatency
        return IsDevice(id) || IsStream(id);
    case kAudioPlugInPropertyDeviceList:
    case kAudioPlugInPropertyTranslateUIDToDevice:
    case kAudioPlugInPropertyResourceBundle:
        return id == kObjectPlugIn;
    case kAudioObjectPropertyName:
    case kAudioDevicePropertyDeviceUID:
    case kAudioDevicePropertyModelUID:
    case kAudioDevicePropertyTransportType:
    case kAudioDevicePropertyRelatedDevices:
    case kAudioDevicePropertyClockDomain:
    case kAudioDevicePropertyDeviceIsAlive:
    case kAudioDevicePropertyDeviceIsRunning:
    case kAudioDevicePropertyDeviceCanBeDefaultDevice:
    case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
    case kAudioDevicePropertyStreams:
    case kAudioObjectPropertyControlList:
    case kAudioDevicePropertySafetyOffset:
    case kAudioDevicePropertyNominalSampleRate:
    case kAudioDevicePropertyAvailableNominalSampleRates:
    case kAudioDevicePropertyIsHidden:
    case kAudioDevicePropertyZeroTimeStampPeriod:
    case kAudioDevicePropertyClockIsStable:
        return IsDevice(id);
    case kAudioStreamPropertyIsActive:
    case kAudioStreamPropertyDirection:
    case kAudioStreamPropertyTerminalType:
    case kAudioStreamPropertyStartingChannel:
    case kAudioStreamPropertyVirtualFormat:
    case kAudioStreamPropertyPhysicalFormat:
    case kAudioStreamPropertyAvailableVirtualFormats:
    case kAudioStreamPropertyAvailablePhysicalFormats:
        return IsStream(id);
    default:
        return false;
    }
}

static OSStatus IsPropertySettable(AudioServerPlugInDriverRef inDriver, AudioObjectID id, pid_t pid, const AudioObjectPropertyAddress *a, Boolean *out) {
    if (!HasProperty(inDriver, id, pid, a)) return kAudioHardwareUnknownPropertyError;
    *out = false;
    return 0;
}

static OSStatus GetPropertyDataSize(AudioServerPlugInDriverRef inDriver, AudioObjectID id, pid_t pid, const AudioObjectPropertyAddress *a,
                                    UInt32 qSize, const void *q, UInt32 *out) {
    (void)qSize; (void)q;
    if (!HasProperty(inDriver, id, pid, a)) return kAudioHardwareUnknownPropertyError;
    AudioObjectID ids[2];
    switch (a->mSelector) {
    case kAudioObjectPropertyBaseClass:
    case kAudioObjectPropertyClass:
        *out = sizeof(AudioClassID);
        return 0;
    case kAudioObjectPropertyOwner:
    case kAudioPlugInPropertyTranslateUIDToDevice:
        *out = sizeof(AudioObjectID);
        return 0;
    case kAudioObjectPropertyManufacturer:
    case kAudioPlugInPropertyResourceBundle:
    case kAudioObjectPropertyName:
    case kAudioDevicePropertyDeviceUID:
    case kAudioDevicePropertyModelUID:
        *out = sizeof(CFStringRef);
        return 0;
    case kAudioObjectPropertyOwnedObjects:
        if (id == kObjectPlugIn) *out = 2 * sizeof(AudioObjectID);
        else if (IsDevice(id)) *out = DeviceStreams(id, a->mScope, ids) * sizeof(AudioObjectID);
        else *out = 0;
        return 0;
    case kAudioPlugInPropertyDeviceList:
        *out = 2 * sizeof(AudioObjectID);
        return 0;
    case kAudioDevicePropertyStreams:
        *out = DeviceStreams(id, a->mScope, ids) * sizeof(AudioObjectID);
        return 0;
    case kAudioDevicePropertyRelatedDevices:
        *out = sizeof(AudioObjectID);
        return 0;
    case kAudioObjectPropertyControlList:
        *out = 0;
        return 0;
    case kAudioDevicePropertyNominalSampleRate:
        *out = sizeof(Float64);
        return 0;
    case kAudioDevicePropertyAvailableNominalSampleRates:
        *out = sizeof(AudioValueRange);
        return 0;
    case kAudioStreamPropertyVirtualFormat:
    case kAudioStreamPropertyPhysicalFormat:
        *out = sizeof(AudioStreamBasicDescription);
        return 0;
    case kAudioStreamPropertyAvailableVirtualFormats:
    case kAudioStreamPropertyAvailablePhysicalFormats:
        *out = sizeof(AudioStreamRangedDescription);
        return 0;
    default:  // all remaining properties are UInt32
        *out = sizeof(UInt32);
        return 0;
    }
}

static OSStatus GetPropertyData(AudioServerPlugInDriverRef inDriver, AudioObjectID id, pid_t pid, const AudioObjectPropertyAddress *a,
                                UInt32 qSize, const void *q, UInt32 inDataSize, UInt32 *outDataSize, void *outData) {
    if (!HasProperty(inDriver, id, pid, a)) return kAudioHardwareUnknownPropertyError;
    Boolean isMic = id == kObjectMic || id == kObjectMicStream;
    AudioObjectID ids[2];
    switch (a->mSelector) {
    case kAudioObjectPropertyBaseClass:
        RETURN_VALUE(AudioClassID, kAudioObjectClassID);
    case kAudioObjectPropertyClass:
        RETURN_VALUE(AudioClassID, id == kObjectPlugIn ? kAudioPlugInClassID : IsDevice(id) ? kAudioDeviceClassID : kAudioStreamClassID);
    case kAudioObjectPropertyOwner:
        RETURN_VALUE(AudioObjectID, id == kObjectPlugIn ? kAudioObjectUnknown
                                    : IsDevice(id)      ? kObjectPlugIn
                                    : isMic             ? kObjectMic
                                                        : kObjectSink);
    case kAudioObjectPropertyManufacturer:
        RETURN_VALUE(CFStringRef, kManufacturer);
    case kAudioObjectPropertyOwnedObjects:
    case kAudioPlugInPropertyDeviceList:
        if (id == kObjectPlugIn) {
            ids[0] = kObjectMic;
            ids[1] = kObjectSink;
            return ReturnIDs(ids, 2, inDataSize, outDataSize, outData);
        }
        if (IsDevice(id)) return ReturnIDs(ids, DeviceStreams(id, a->mScope, ids), inDataSize, outDataSize, outData);
        *outDataSize = 0;
        return 0;
    case kAudioPlugInPropertyTranslateUIDToDevice: {
        if (qSize != sizeof(CFStringRef) || q == NULL) return kAudioHardwareBadPropertySizeError;
        CFStringRef uid = *(const CFStringRef *)q;
        AudioObjectID device = kAudioObjectUnknown;
        if (uid && CFEqual(uid, kMicUID)) device = kObjectMic;
        else if (uid && CFEqual(uid, kSinkUID)) device = kObjectSink;
        RETURN_VALUE(AudioObjectID, device);
    }
    case kAudioPlugInPropertyResourceBundle:
        RETURN_VALUE(CFStringRef, CFSTR(""));
    case kAudioObjectPropertyName:
        RETURN_VALUE(CFStringRef, isMic ? kMicName : kSinkName);
    case kAudioDevicePropertyDeviceUID:
        RETURN_VALUE(CFStringRef, isMic ? kMicUID : kSinkUID);
    case kAudioDevicePropertyModelUID:
        RETURN_VALUE(CFStringRef, kModelUID);
    case kAudioDevicePropertyTransportType:
        RETURN_VALUE(UInt32, kAudioDeviceTransportTypeVirtual);
    case kAudioDevicePropertyRelatedDevices:
        ids[0] = id;
        return ReturnIDs(ids, 1, inDataSize, outDataSize, outData);
    case kAudioDevicePropertyClockDomain:
        RETURN_VALUE(UInt32, 0);
    case kAudioDevicePropertyDeviceIsAlive:
    case kAudioDevicePropertyClockIsStable:
        RETURN_VALUE(UInt32, 1);
    case kAudioDevicePropertyDeviceIsRunning: {
        pthread_mutex_lock(&gStateMutex);
        UInt32 running = gRunning[DeviceIndex(id)] > 0;
        pthread_mutex_unlock(&gStateMutex);
        RETURN_VALUE(UInt32, running);
    }
    case kAudioDevicePropertyDeviceCanBeDefaultDevice:
    case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
        // Only the microphone, and only as an input.
        RETURN_VALUE(UInt32, isMic && a->mScope != kAudioObjectPropertyScopeOutput);
    case kAudioDevicePropertyLatency:
    case kAudioDevicePropertySafetyOffset:
        RETURN_VALUE(UInt32, 0);
    case kAudioDevicePropertyStreams:
        return ReturnIDs(ids, DeviceStreams(id, a->mScope, ids), inDataSize, outDataSize, outData);
    case kAudioObjectPropertyControlList:
        *outDataSize = 0;
        return 0;
    case kAudioDevicePropertyNominalSampleRate:
        RETURN_VALUE(Float64, kSampleRate);
    case kAudioDevicePropertyAvailableNominalSampleRates: {
        AudioValueRange range = {kSampleRate, kSampleRate};
        RETURN_VALUE(AudioValueRange, range);
    }
    case kAudioDevicePropertyIsHidden:
        RETURN_VALUE(UInt32, !isMic);
    case kAudioDevicePropertyZeroTimeStampPeriod:
        RETURN_VALUE(UInt32, kZeroTimeStampPeriod);
    case kAudioStreamPropertyIsActive:
        RETURN_VALUE(UInt32, 1);
    case kAudioStreamPropertyDirection:
        RETURN_VALUE(UInt32, isMic ? 1 : 0);
    case kAudioStreamPropertyTerminalType:
        RETURN_VALUE(UInt32, isMic ? kAudioStreamTerminalTypeMicrophone : kAudioStreamTerminalTypeSpeaker);
    case kAudioStreamPropertyStartingChannel:
        RETURN_VALUE(UInt32, 1);
    case kAudioStreamPropertyVirtualFormat:
    case kAudioStreamPropertyPhysicalFormat:
        RETURN_VALUE(AudioStreamBasicDescription, Format());
    case kAudioStreamPropertyAvailableVirtualFormats:
    case kAudioStreamPropertyAvailablePhysicalFormats: {
        AudioStreamRangedDescription ranged = {Format(), {kSampleRate, kSampleRate}};
        RETURN_VALUE(AudioStreamRangedDescription, ranged);
    }
    default:
        return kAudioHardwareUnknownPropertyError;
    }
}

static OSStatus SetPropertyData(AudioServerPlugInDriverRef inDriver, AudioObjectID id, pid_t pid, const AudioObjectPropertyAddress *a,
                                UInt32 qSize, const void *q, UInt32 inDataSize, const void *inData) {
    (void)qSize; (void)q; (void)inDataSize; (void)inData;
    if (!HasProperty(inDriver, id, pid, a)) return kAudioHardwareUnknownPropertyError;
    return kAudioHardwareUnsupportedOperationError;
}

#pragma mark - IO

static OSStatus StartIO(AudioServerPlugInDriverRef inDriver, AudioObjectID id, UInt32 client) {
    (void)client;
    if (inDriver != gDriver || !IsDevice(id)) return kAudioHardwareBadObjectError;
    pthread_mutex_lock(&gStateMutex);
    if (gRunning[0] + gRunning[1] == 0) {
        // A fresh timeline: forget everything written on the previous one.
        gAnchorHostTime = mach_absolute_time();
        atomic_store(&gWrittenEnd, 0);
    }
    gRunning[DeviceIndex(id)]++;
    pthread_mutex_unlock(&gStateMutex);
    return 0;
}

static OSStatus StopIO(AudioServerPlugInDriverRef inDriver, AudioObjectID id, UInt32 client) {
    (void)client;
    if (inDriver != gDriver || !IsDevice(id)) return kAudioHardwareBadObjectError;
    pthread_mutex_lock(&gStateMutex);
    if (gRunning[DeviceIndex(id)] > 0) gRunning[DeviceIndex(id)]--;
    pthread_mutex_unlock(&gStateMutex);
    return 0;
}

static OSStatus GetZeroTimeStamp(AudioServerPlugInDriverRef inDriver, AudioObjectID id, UInt32 client, Float64 *outSampleTime,
                                 UInt64 *outHostTime, UInt64 *outSeed) {
    (void)client;
    if (inDriver != gDriver || !IsDevice(id)) return kAudioHardwareBadObjectError;
    pthread_mutex_lock(&gStateMutex);
    UInt64 anchor = gAnchorHostTime;
    pthread_mutex_unlock(&gStateMutex);
    // The most recent period boundary of the shared timeline.
    Float64 ticksPerPeriod = gHostTicksPerFrame * kZeroTimeStampPeriod;
    UInt64 periods = (UInt64)((Float64)(mach_absolute_time() - anchor) / ticksPerPeriod);
    *outSampleTime = (Float64)periods * kZeroTimeStampPeriod;
    *outHostTime = anchor + (UInt64)((Float64)periods * ticksPerPeriod);
    *outSeed = 1;
    return 0;
}

static OSStatus WillDoIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID id, UInt32 client, UInt32 op, Boolean *outWillDo,
                                  Boolean *outInPlace) {
    (void)client;
    if (inDriver != gDriver || !IsDevice(id)) return kAudioHardwareBadObjectError;
    *outWillDo = (id == kObjectMic && op == kAudioServerPlugInIOOperationReadInput) ||
                 (id == kObjectSink && op == kAudioServerPlugInIOOperationWriteMix);
    *outInPlace = true;
    return 0;
}

static OSStatus BeginIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID id, UInt32 c, UInt32 op, UInt32 n,
                                 const AudioServerPlugInIOCycleInfo *info) {
    (void)c; (void)op; (void)n; (void)info;
    return inDriver == gDriver && IsDevice(id) ? 0 : kAudioHardwareBadObjectError;
}

static OSStatus EndIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID id, UInt32 c, UInt32 op, UInt32 n,
                               const AudioServerPlugInIOCycleInfo *info) {
    (void)c; (void)op; (void)n; (void)info;
    return inDriver == gDriver && IsDevice(id) ? 0 : kAudioHardwareBadObjectError;
}

static void WriteSink(SInt64 start, const float *samples, UInt32 frames) {
    SInt64 end = atomic_load(&gWrittenEnd);
    // Silence any gap since the last write so stale samples never become valid.
    if (start > end) {
        SInt64 gap = start - end;
        if (gap > kRingFrames) gap = kRingFrames;
        for (SInt64 t = start - gap; t < start; t++) gRing[t & (kRingFrames - 1)] = 0;
    }
    for (UInt32 i = 0; i < frames; i++) gRing[(start + i) & (kRingFrames - 1)] = samples[i * kChannels];
    atomic_store(&gWrittenEnd, start + frames);
}

static void ReadMic(SInt64 start, float *samples, UInt32 frames) {
    SInt64 end = atomic_load(&gWrittenEnd);
    for (UInt32 i = 0; i < frames; i++) {
        SInt64 t = start + i;
        float v = t < end && t >= end - kRingFrames ? gRing[t & (kRingFrames - 1)] : 0;
        for (UInt32 c = 0; c < kChannels; c++) samples[i * kChannels + c] = v;
    }
}

static OSStatus DoIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID id, AudioObjectID stream, UInt32 client, UInt32 op,
                              UInt32 frames, const AudioServerPlugInIOCycleInfo *info, void *main, void *secondary) {
    (void)stream; (void)client; (void)secondary;
    if (inDriver != gDriver || !IsDevice(id)) return kAudioHardwareBadObjectError;
    if (id == kObjectSink && op == kAudioServerPlugInIOOperationWriteMix)
        WriteSink((SInt64)info->mOutputTime.mSampleTime, (const float *)main, frames);
    else if (id == kObjectMic && op == kAudioServerPlugInIOOperationReadInput)
        ReadMic((SInt64)info->mInputTime.mSampleTime, (float *)main, frames);
    return 0;
}
