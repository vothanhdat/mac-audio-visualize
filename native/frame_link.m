#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

// A display link follows the display the view is on and fires in step with its
// refresh, so a limited frame rate stays even. A timer would drift against the
// refresh and show some frames one refresh late.

static atomic_bool tick_pending;

API_AVAILABLE(macos(14.0))
@interface VAFrameLink : NSObject
@property(nonatomic, strong) CADisplayLink *link;
@end

@implementation VAFrameLink
- (void)tick:(CADisplayLink *)__unused link {
    atomic_store(&tick_pending, true);
}
@end

// Returns a paused link for the view, or NULL before macOS 14.
void *va_frame_link_create(void *ns_view) {
    if (@available(macOS 14.0, *)) {
        NSView *view = (__bridge NSView *)ns_view;
        VAFrameLink *target = [VAFrameLink new];
        target.link = [view displayLinkWithTarget:target selector:@selector(tick:)];
        target.link.paused = YES;
        [target.link addToRunLoop:NSRunLoop.mainRunLoop forMode:NSRunLoopCommonModes];
        return (void *)CFBridgingRetain(target);
    }
    return NULL;
}

// Runs the link at this many frames per second, or pauses it for 0.
void va_frame_link_set_rate(void *handle, uint32_t frames_per_second) {
    if (@available(macOS 14.0, *)) {
        VAFrameLink *target = (__bridge VAFrameLink *)handle;
        if (frames_per_second > 0) {
            float rate = (float)frames_per_second;
            target.link.preferredFrameRateRange = CAFrameRateRangeMake(rate, rate, rate);
        }
        target.link.paused = frames_per_second == 0;
    }
}

// Whether the link has fired since the previous call.
bool va_frame_link_take_tick(void) {
    return atomic_exchange(&tick_pending, false);
}
