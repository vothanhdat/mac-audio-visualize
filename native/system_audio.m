// Native Core Audio process-tap bridge. No installed HAL driver is needed.
#import <Foundation/Foundation.h>
#import <CoreAudio/CoreAudio.h>
#import <CoreAudio/AudioHardwareTapping.h>
#import <CoreAudio/CATapDescription.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*VASamples)(void *context, const float *left, const float *right,
                          uint32_t frames, uint32_t stride);
// Reports the sample rate of the samples that follow. Called on the capture
// queue only while no samples are delivered: before capture first starts and
// while it is stopped for a rebuild.
typedef void (*VASampleRate)(void *context, uint32_t sampleRate);

typedef struct {
    VASamples samples;
    void *context;
} VASink;

// Output changes often arrive as a burst (the device, then its format), so a
// rebuild waits for them to settle. The watchdog catches changes whose
// notification was missed and retries a rebuild that failed.
static const int64_t REBUILD_DELAY_NS = 200 * NSEC_PER_MSEC;
static const int64_t WATCHDOG_INTERVAL_NS = NSEC_PER_SEC;
static const OSStatus UNSUPPORTED_TAP_FORMAT = 'vafm';

static OSStatus read_property(AudioObjectID object, AudioObjectPropertySelector selector,
                               AudioObjectPropertyScope scope, void *value, UInt32 size) {
    AudioObjectPropertyAddress address = {selector, scope, kAudioObjectPropertyElementMain};
    return AudioObjectGetPropertyData(object, &address, 0, NULL, &size, value);
}

static AudioObjectPropertyAddress global_address(AudioObjectPropertySelector selector) {
    return (AudioObjectPropertyAddress){selector, kAudioObjectPropertyScopeGlobal,
                                        kAudioObjectPropertyElementMain};
}

static AudioObjectID default_output_device(void) {
    AudioObjectID device = kAudioObjectUnknown;
    if (read_property(kAudioObjectSystemObject, kAudioHardwarePropertyDefaultOutputDevice,
                      kAudioObjectPropertyScopeGlobal, &device, sizeof(device)) != noErr)
        return kAudioObjectUnknown;
    return device;
}

// Zero when the rate is unavailable.
static Float64 nominal_rate(AudioObjectID device) {
    Float64 rate = 0;
    if (device == kAudioObjectUnknown ||
        read_property(device, kAudioDevicePropertyNominalSampleRate,
                      kAudioObjectPropertyScopeGlobal, &rate, sizeof(rate)) != noErr)
        return 0;
    return rate;
}

// Called on the UI thread at most once per second, never in the audio callback.
// Negative means unavailable; device and stream latency are additive per Core Audio.
double va_output_latency_ms(void) {
    AudioObjectID device = default_output_device();
    if (device == kAudioObjectUnknown) return -1;
    Float64 rate = nominal_rate(device);
    UInt32 deviceLatency = 0, safety = 0, buffer = 0, streamLatency = 0;
    if (!isfinite(rate) || rate <= 0 ||
        read_property(device, kAudioDevicePropertyLatency, kAudioDevicePropertyScopeOutput,
                      &deviceLatency, sizeof(deviceLatency)) != noErr) return -1;
    read_property(device, kAudioDevicePropertySafetyOffset, kAudioDevicePropertyScopeOutput,
                  &safety, sizeof(safety));
    read_property(device, kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeGlobal,
                  &buffer, sizeof(buffer));
    AudioObjectPropertyAddress address = {kAudioDevicePropertyStreams,
        kAudioDevicePropertyScopeOutput, kAudioObjectPropertyElementMain};
    UInt32 size = 0;
    if (AudioObjectGetPropertyDataSize(device, &address, 0, NULL, &size) == noErr && size > 0) {
        AudioStreamID *streams = malloc(size);
        if (streams) {
            UInt32 capacity = size;
            if (AudioObjectGetPropertyData(device, &address, 0, NULL, &size, streams) == noErr &&
                size <= capacity) {
                for (UInt32 index = 0; index < size / sizeof(AudioStreamID); ++index) {
                    UInt32 latency = 0;
                    read_property(streams[index], kAudioStreamPropertyLatency,
                                  kAudioObjectPropertyScopeGlobal, &latency, sizeof(latency));
                    if (latency > streamLatency) streamLatency = latency;
                }
            }
            free(streams);
        }
    }
    return 1000.0 * ((double)deviceLatency + streamLatency + safety + buffer) / rate;
}

static OSStatus receive_audio(AudioObjectID device, const AudioTimeStamp *now,
                              const AudioBufferList *input, const AudioTimeStamp *inputTime,
                              AudioBufferList *output, const AudioTimeStamp *outputTime,
                              void *context) {
    (void)device; (void)now; (void)inputTime; (void)output; (void)outputTime;
    const VASink *sink = context;
    if (!input || input->mNumberBuffers == 0) return noErr;
    const AudioBuffer *first = &input->mBuffers[0];
    if (!first->mData || first->mNumberChannels == 0) return noErr;
    uint32_t stride = first->mNumberChannels;
    uint32_t frames = first->mDataByteSize / sizeof(float) / stride;
    const float *left = first->mData;
    const float *right = left;
    if (stride >= 2) {
        right = left + 1;
    } else if (input->mNumberBuffers >= 2) {
        const AudioBuffer *second = &input->mBuffers[1];
        if (!second->mData || second->mNumberChannels != 1) return noErr;
        right = second->mData;
        uint32_t rightFrames = second->mDataByteSize / sizeof(float);
        if (rightFrames < frames) frames = rightFrames;
    }
    sink->samples(sink->context, left, right, frames, stride);
    return noErr;
}

// Owns the process tap, its private aggregate device and the IOProc, and
// rebuilds them when the default output device or the tap's format changes.
// All capture state is used only on `_queue`, so a rebuild never overlaps
// startup, another rebuild or shutdown, and samples always come from at most
// one IOProc.
API_AVAILABLE(macos(14.2))
@interface VACapture : NSObject
- (instancetype)initWithSink:(VASink)sink sampleRate:(VASampleRate)sampleRate;
- (OSStatus)open:(const char **)stage;
- (void)restart;
- (void)close;
@end

@implementation VACapture {
    VASink _sink;
    VASampleRate _sampleRate;
    dispatch_queue_t _queue;
    dispatch_source_t _watchdog;
    AudioObjectPropertyListenerBlock _listener;
    BOOL _listening;
    // Used only on _queue.
    AudioObjectID _tap;
    AudioObjectID _aggregate;
    AudioDeviceIOProcID _io;
    AudioObjectID _output;
    Float64 _outputRate;
    Float64 _tapRate;
    AudioObjectID _watchedTap;
    AudioObjectID _watchedOutput;
    uint64_t _rebuildRequest;
    BOOL _rebuildPending;
    BOOL _forceRebuild;
    BOOL _failing;
    BOOL _closed;
}

- (instancetype)initWithSink:(VASink)sink sampleRate:(VASampleRate)sampleRate {
    if ((self = [super init])) {
        _sink = sink;
        _sampleRate = sampleRate;
        dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_autorelease_frequency(
            DISPATCH_QUEUE_SERIAL, DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM);
        attributes = dispatch_queue_attr_make_with_qos_class(attributes,
                                                             QOS_CLASS_USER_INITIATED, 0);
        _queue = dispatch_queue_create("VisualAudio system capture", attributes);
        // Blocks hold the capture weakly: queued work that outlives close does nothing.
        __weak VACapture *weakSelf = self;
        _listener = ^(UInt32 count, const AudioObjectPropertyAddress *addresses) {
            BOOL outputChanged = NO;
            for (UInt32 index = 0; index < count; ++index) {
                outputChanged |=
                    addresses[index].mSelector == kAudioHardwarePropertyDefaultOutputDevice;
            }
            [weakSelf scheduleRebuild:outputChanged];
        };
    }
    return self;
}

// Returns ownership of a running capture only after every startup stage succeeds.
- (OSStatus)open:(const char **)stage {
    __block OSStatus status = noErr;
    dispatch_sync(_queue, ^{
        status = [self startCapture:stage];
    });
    if (status != noErr) return status;
    AudioObjectPropertyAddress address = global_address(kAudioHardwarePropertyDefaultOutputDevice);
    _listening = AudioObjectAddPropertyListenerBlock(kAudioObjectSystemObject, &address, _queue,
                                                     _listener) == noErr;
    _watchdog = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, _queue);
    dispatch_source_set_timer(_watchdog, dispatch_time(DISPATCH_TIME_NOW, WATCHDOG_INTERVAL_NS),
                              WATCHDOG_INTERVAL_NS, WATCHDOG_INTERVAL_NS / 10);
    __weak VACapture *weakSelf = self;
    dispatch_source_set_event_handler(_watchdog, ^{
        [weakSelf rebuildIfNeeded:NO];
    });
    dispatch_resume(_watchdog);
    return noErr;
}

- (void)restart {
    __weak VACapture *weakSelf = self;
    dispatch_async(_queue, ^{
        [weakSelf scheduleRebuild:YES];
    });
}

// When this returns, no capture callback is running or will run again, so the
// Rust callback context can be dropped. Startup failures close the same way.
- (void)close {
    if (_listening) {
        AudioObjectPropertyAddress address =
            global_address(kAudioHardwarePropertyDefaultOutputDevice);
        AudioObjectRemovePropertyListenerBlock(kAudioObjectSystemObject, &address, _queue,
                                               _listener);
        _listening = NO;
    }
    dispatch_sync(_queue, ^{
        self->_closed = YES;
        if (self->_watchdog) dispatch_source_cancel(self->_watchdog);
        [self stopCapture];
    });
}

// On _queue, with no capture running. A failed start leaves partial state for
// stopCapture.
- (OSStatus)startCapture:(const char **)stage {
    _output = default_output_device();
    _outputRate = nominal_rate(_output);

    *stage = "create process tap";
    CATapDescription *description = [[CATapDescription alloc]
        initStereoGlobalTapButExcludeProcesses:@[]];
    description.name = @"VisualAudio System Audio";
    description.UUID = [NSUUID UUID];
    description.privateTap = YES;
    // A visualizer only observes audio; speakers continue playing directly.
    description.muteBehavior = CATapUnmuted;
    OSStatus status = AudioHardwareCreateProcessTap(description, &_tap);
    if (status != noErr) return status;

    *stage = "read process tap format";
    AudioStreamBasicDescription format = {0};
    status = read_property(_tap, kAudioTapPropertyFormat, kAudioObjectPropertyScopeGlobal,
                           &format, sizeof(format));
    if (status != noErr) return status;
    if (format.mFormatID != kAudioFormatLinearPCM ||
        !(format.mFormatFlags & kAudioFormatFlagIsFloat) ||
        (format.mFormatFlags & kAudioFormatFlagIsBigEndian) ||
        format.mBitsPerChannel != 32 || format.mChannelsPerFrame != 2 ||
        !isfinite(format.mSampleRate) || format.mSampleRate < 1 ||
        format.mSampleRate > UINT32_MAX) return UNSUPPORTED_TAP_FORMAT;
    _tapRate = format.mSampleRate;

    *stage = "create private capture aggregate";
    // This ephemeral, private tap-only device never becomes the sound output.
    NSDictionary *aggregate = @{
        @kAudioAggregateDeviceNameKey: @"VisualAudio Capture",
        @kAudioAggregateDeviceUIDKey: [NSUUID UUID].UUIDString,
        @kAudioAggregateDeviceIsPrivateKey: @YES,
        @kAudioAggregateDeviceIsStackedKey: @NO,
        @kAudioAggregateDeviceTapAutoStartKey: @YES,
        @kAudioAggregateDeviceTapListKey: @[@{
            @kAudioSubTapUIDKey: description.UUID.UUIDString,
            @kAudioSubTapDriftCompensationKey: @YES
        }]
    };
    status = AudioHardwareCreateAggregateDevice((__bridge CFDictionaryRef)aggregate, &_aggregate);
    if (status != noErr) return status;
    *stage = "register capture callback";
    status = AudioDeviceCreateIOProcID(_aggregate, receive_audio, &_sink, &_io);
    if (status != noErr) return status;
    // No samples flow yet, so the rate applies exactly to the samples that follow.
    _sampleRate(_sink.context, (uint32_t)llround(_tapRate));
    *stage = "start system audio capture";
    status = AudioDeviceStart(_aggregate, _io);
    if (status != noErr) return status;
    _watchedTap = [self watch:_tap selector:kAudioTapPropertyFormat];
    _watchedOutput = [self watch:_output selector:kAudioDevicePropertyNominalSampleRate];
    return noErr;
}

// On _queue. Safe after a partial start.
- (void)stopCapture {
    [self unwatch:&_watchedTap selector:kAudioTapPropertyFormat];
    [self unwatch:&_watchedOutput selector:kAudioDevicePropertyNominalSampleRate];
    if (_io) {
        AudioDeviceStop(_aggregate, _io);
        // Removing the IOProc completes before the Rust callback context is dropped.
        AudioDeviceDestroyIOProcID(_aggregate, _io);
        _io = NULL;
    }
    if (_aggregate) AudioHardwareDestroyAggregateDevice(_aggregate);
    _aggregate = kAudioObjectUnknown;
    if (_tap) AudioHardwareDestroyProcessTap(_tap);
    _tap = kAudioObjectUnknown;
}

- (AudioObjectID)watch:(AudioObjectID)object selector:(AudioObjectPropertySelector)selector {
    if (object == kAudioObjectUnknown) return kAudioObjectUnknown;
    AudioObjectPropertyAddress address = global_address(selector);
    return AudioObjectAddPropertyListenerBlock(object, &address, _queue, _listener) == noErr
        ? object : kAudioObjectUnknown;
}

- (void)unwatch:(AudioObjectID *)object selector:(AudioObjectPropertySelector)selector {
    if (*object == kAudioObjectUnknown) return;
    AudioObjectPropertyAddress address = global_address(selector);
    AudioObjectRemovePropertyListenerBlock(*object, &address, _queue, _listener);
    *object = kAudioObjectUnknown;
}

// On _queue. Each notification postpones the rebuild until the burst settles.
- (void)scheduleRebuild:(BOOL)force {
    if (_closed) return;
    _forceRebuild |= force;
    _rebuildPending = YES;
    uint64_t request = ++_rebuildRequest;
    __weak VACapture *weakSelf = self;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, REBUILD_DELAY_NS), _queue, ^{
        VACapture *capture = weakSelf;
        if (capture && capture->_rebuildRequest == request) [capture rebuildIfNeeded:YES];
    });
}

// On _queue: from a settled notification burst, or once per second from the watchdog.
- (void)rebuildIfNeeded:(BOOL)scheduled {
    if (_closed || (!scheduled && _rebuildPending)) return;
    BOOL force = _forceRebuild;
    _rebuildPending = NO;
    _forceRebuild = NO;
    if (force || [self captureChanged]) [self rebuild];
}

// On _queue. True if capture failed or no longer matches the output it was built for.
- (BOOL)captureChanged {
    if (!_io) return YES;
    AudioObjectID output = default_output_device();
    if (output != _output) return YES;
    Float64 rate = nominal_rate(output);
    if (rate > 0 && rate != _outputRate) return YES;
    AudioStreamBasicDescription format = {0};
    return read_property(_tap, kAudioTapPropertyFormat, kAudioObjectPropertyScopeGlobal,
                         &format, sizeof(format)) == noErr &&
        format.mSampleRate != _tapRate;
}

// On _queue. The old IOProc stops before the new one starts, so the Rust ring
// keeps a single producer, and the new rate is published in between.
- (void)rebuild {
    AudioObjectID previousOutput = _output;
    Float64 previousRate = _io ? _tapRate : 0;
    [self stopCapture];
    const char *stage = "restart system audio capture";
    OSStatus status = [self startCapture:&stage];
    if (status != noErr) {
        [self stopCapture];
        if (!_failing) {
            fprintf(stderr, "system audio capture stopped: could not %s after an output change "
                    "(Core Audio OSStatus %d); retrying every second\n", stage, (int)status);
        }
        _failing = YES;
        return;
    }
    if (_failing || _output != previousOutput || _tapRate != previousRate) {
        fprintf(stderr, "system audio output changed; capturing at %.0f Hz\n", _tapRate);
    }
    _failing = NO;
}

@end

void *va_capture_open(VASamples samples, VASampleRate sampleRate, void *context,
                      uint16_t *channels, char *error, size_t errorSize) {
    @autoreleasepool {
        if (@available(macOS 14.2, *)) {
            VACapture *capture = [[VACapture alloc]
                initWithSink:(VASink){samples, context} sampleRate:sampleRate];
            const char *stage = "start system audio capture";
            OSStatus status = [capture open:&stage];
            if (status == noErr) {
                *channels = 2;
                return (__bridge_retained void *)capture;
            }
            if (status == UNSUPPORTED_TAP_FORMAT) {
                snprintf(error, errorSize, "System audio tap did not provide stereo Float32 PCM");
            } else {
                snprintf(error, errorSize, "Could not %s (Core Audio OSStatus %d). Allow System Audio Recording for VisualAudio or your terminal in System Settings > Privacy & Security, then restart.", stage, (int)status);
            }
            [capture close];
            return NULL;
        } else {
            snprintf(error, errorSize, "System audio capture requires macOS 14.2 or newer; use --device NAME for an input device");
            return NULL;
        }
    }
}

// Rebuilds the tap as if the output had changed. Returns immediately.
void va_capture_restart(void *handle) {
    if (!handle) return;
    if (@available(macOS 14.2, *)) {
        [(__bridge VACapture *)handle restart];
    }
}

void va_capture_close(void *handle) {
    if (!handle) return;
    if (@available(macOS 14.2, *)) {
        @autoreleasepool {
            VACapture *capture = (__bridge_transfer VACapture *)handle;
            [capture close];
        }
    }
}
