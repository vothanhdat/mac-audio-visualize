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

typedef struct {
    AudioObjectID tap;
    AudioObjectID aggregate;
    AudioDeviceIOProcID io;
    VASamples samples;
    void *context;
} VACapture;

static OSStatus read_property(AudioObjectID object, AudioObjectPropertySelector selector,
                               AudioObjectPropertyScope scope, void *value, UInt32 size) {
    AudioObjectPropertyAddress address = {selector, scope, kAudioObjectPropertyElementMain};
    return AudioObjectGetPropertyData(object, &address, 0, NULL, &size, value);
}

// Called on the UI thread at most once per second, never in the audio callback.
// Negative means unavailable; device and stream latency are additive per Core Audio.
double va_output_latency_ms(void) {
    AudioObjectID device = kAudioObjectUnknown;
    if (read_property(kAudioObjectSystemObject, kAudioHardwarePropertyDefaultOutputDevice,
                      kAudioObjectPropertyScopeGlobal, &device, sizeof(device)) != noErr ||
        device == kAudioObjectUnknown) return -1;
    Float64 rate = 0;
    UInt32 deviceLatency = 0, safety = 0, buffer = 0, streamLatency = 0;
    if (read_property(device, kAudioDevicePropertyNominalSampleRate,
                      kAudioObjectPropertyScopeGlobal, &rate, sizeof(rate)) != noErr ||
        !isfinite(rate) || rate <= 0 ||
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

void va_capture_close(VACapture *capture) {
    if (!capture) return;
    if (@available(macOS 14.2, *)) {
        if (capture->io) {
            AudioDeviceStop(capture->aggregate, capture->io);
            // Removing the IOProc completes before the Rust callback context is dropped.
            AudioDeviceDestroyIOProcID(capture->aggregate, capture->io);
        }
        if (capture->aggregate) AudioHardwareDestroyAggregateDevice(capture->aggregate);
        if (capture->tap) AudioHardwareDestroyProcessTap(capture->tap);
    }
    free(capture);
}

static OSStatus receive_audio(AudioObjectID device, const AudioTimeStamp *now,
                              const AudioBufferList *input, const AudioTimeStamp *inputTime,
                              AudioBufferList *output, const AudioTimeStamp *outputTime,
                              void *context) {
    (void)device; (void)now; (void)inputTime; (void)output; (void)outputTime;
    VACapture *capture = context;
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
    capture->samples(capture->context, left, right, frames, stride);
    return noErr;
}

// Returns ownership only after every startup stage succeeds. Partial startup
// follows the same teardown path as normal shutdown.
VACapture *va_capture_open(VASamples samples, void *context, uint32_t *sampleRate,
                           uint16_t *channels, char *error, size_t errorSize) {
    @autoreleasepool {
        if (@available(macOS 14.2, *)) {
            VACapture *capture = calloc(1, sizeof(VACapture));
            if (!capture) {
                snprintf(error, errorSize, "Could not allocate system audio capture");
                return NULL;
            }
            capture->samples = samples;
            capture->context = context;
            NSDictionary *aggregate = nil;
            const char *stage = "create process tap";
            CATapDescription *description = [[CATapDescription alloc]
                initStereoGlobalTapButExcludeProcesses:@[]];
            description.name = @"VisualAudio System Audio";
            description.UUID = [NSUUID UUID];
            description.privateTap = YES;
            // A visualizer only observes audio; speakers continue playing directly.
            description.muteBehavior = CATapUnmuted;
            OSStatus status = AudioHardwareCreateProcessTap(description, &capture->tap);
            if (status != noErr) goto failed;

            stage = "read process tap format";
            AudioStreamBasicDescription format = {0};
            UInt32 formatSize = sizeof(format);
            AudioObjectPropertyAddress address = {
                kAudioTapPropertyFormat, kAudioObjectPropertyScopeGlobal,
                kAudioObjectPropertyElementMain
            };
            status = AudioObjectGetPropertyData(capture->tap, &address, 0, NULL,
                                                &formatSize, &format);
            if (status != noErr) goto failed;
            if (format.mFormatID != kAudioFormatLinearPCM ||
                !(format.mFormatFlags & kAudioFormatFlagIsFloat) ||
                (format.mFormatFlags & kAudioFormatFlagIsBigEndian) ||
                format.mBitsPerChannel != 32 || format.mChannelsPerFrame != 2 ||
                !isfinite(format.mSampleRate) || format.mSampleRate < 1 ||
                format.mSampleRate > UINT32_MAX) {
                snprintf(error, errorSize, "System audio tap did not provide stereo Float32 PCM");
                va_capture_close(capture);
                return NULL;
            }
            *sampleRate = (uint32_t)llround(format.mSampleRate);
            *channels = (uint16_t)format.mChannelsPerFrame;

            stage = "create private capture aggregate";
            // This ephemeral, private tap-only device never becomes the sound output.
            aggregate = @{
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
            status = AudioHardwareCreateAggregateDevice((__bridge CFDictionaryRef)aggregate,
                                                        &capture->aggregate);
            if (status != noErr) goto failed;
            stage = "register capture callback";
            status = AudioDeviceCreateIOProcID(capture->aggregate, receive_audio,
                                               capture, &capture->io);
            if (status != noErr) goto failed;
            stage = "start system audio capture";
            status = AudioDeviceStart(capture->aggregate, capture->io);
            if (status != noErr) goto failed;
            return capture;

        failed:
            snprintf(error, errorSize, "Could not %s (Core Audio OSStatus %d). Allow System Audio Recording for VisualAudio or your terminal in System Settings > Privacy & Security, then restart.", stage, (int)status);
            va_capture_close(capture);
            return NULL;
        } else {
            snprintf(error, errorSize, "System audio capture requires macOS 14.2 or newer; use --device NAME for an input device");
            return NULL;
        }
    }
}
