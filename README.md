# VisualAudio

A low-latency macOS audio visualizer written in Rust. By default it captures
system audio through a native Core Audio process tap on macOS 14.2 or newer,
moves synchronized stereo samples through a lock-free SPSC ring, and
combines CPU FFTs with GPU resonator analysis and rendering through `wgpu`/Metal
and WGSL.
The optional transparent-background mode renders normally onto opaque black,
then a GPU compositor converts RGB light intensity into a clean premultiplied
alpha channel. It is disabled by default; the detached controls stay opaque.
The visualizer uses a hidden, transparent macOS title bar: the traffic-light
buttons remain available and dragging anywhere in the canvas moves the window.
It can be as short as 60 points (320 wide) for a thin spectrum strip.
The visualizer and controls windows reopen where you left them, at the same
size. Their positions are saved half a second after a move or resize and when
quitting, in `window.json` (`window-NAME.json` for `--profile NAME`) next to
the settings. A position on a display that is no longer connected is ignored,
so the window opens on screen. **Reset** does not move windows.
Visualizer controls are restored on startup from
`~/Library/Application Support/VisualAudio/settings.json`; edits are saved
automatically after a short debounce.
New profiles and **Reset** use the saved configuration adopted on October 20,
2026: gain 1.7, hybrid resonator + FFT with adaptive matching, 3 px bars, a bottom
mirror, and a 16 Hz–22.5 kHz frequency range. Existing profiles keep their saved
settings. Automatic output synchronization is enabled by default.

## Install

With Homebrew on an Apple Silicon Mac running macOS 14.2 or newer:

```sh
brew install --cask vothanhdat/tap/visual-audio
```

Open **VisualAudio** from Applications and allow **System Audio Recording**
when macOS asks. Update with `brew upgrade --cask visual-audio`; remove with
`brew uninstall --cask visual-audio` (add `--zap` to also delete settings).
The app is ad-hoc signed, not notarized, so the cask clears the quarantine
flag after installing. Because of that ad-hoc signature, macOS may ask for the
recording permission again after an update.

To publish a release, run `sh scripts/package-release.sh`. It builds the app,
checks that the bundle version matches `Cargo.toml`, and writes
`dist/VisualAudio-VERSION.zip` and its SHA-256 for the cask.

## System audio capture (default)

No BlackHole, virtual audio driver, or Multi-Output Device is required. Keep
your normal speakers/headphones selected as the macOS sound output, then run:

```sh
cargo run --release
```

Allow **System Audio Recording** when macOS asks. If audio stays silent after a
denial, enable VisualAudio (or the terminal hosting `cargo run`) under **System
Settings → Privacy & Security → Screen & System Audio Recording** and restart.
The permission category's exact name varies by macOS version. This mode does
not use the microphone.

To build a local app bundle with a consistent application identity and capture
usage description:

```sh
sh scripts/build-macos-app.sh
open target/VisualAudio.app
```

The script builds and ad-hoc signs the bundle in `target`; it does not install it.
The command-line binary also embeds the same usage descriptions. Rebuilding an
ad-hoc signed app can cause macOS to ask for permission again.

The capture architecture follows the same Apple API approach used by
[iqualize](https://github.com/DariusCorvus/iqualize/blob/main/Sources/iQualizeCapture/main.swift):
`CATapDescription` → `AudioHardwareCreateProcessTap` → a private, tap-only
aggregate device → an `AudioDeviceIOProc` → our existing ring and GPU analysis.
The aggregate is created only for capture and destroyed on normal shutdown; it
is never selected as the system output. This still uses an internal Core Audio
device object, but requires no installed virtual driver or manual audio routing.

iqualize mutes the captured original audio so it can play an equalized version.
VisualAudio sets `CATapUnmuted`: your audio goes directly to the output as usual,
and the visualizer only reads a copy. The callback handles interleaved and planar
Float32 stereo without allocating or locking. The native bridge is Objective-C;
the analyzer and UI remain Rust. Building requires Xcode Command Line Tools with
a macOS 14.2+ SDK.

Capture follows output changes without a restart. When the default output
device, its sample rate, or the tap's format changes, the tap and its aggregate
are rebuilt once the change settles (about 200 ms); a once-per-second check
catches missed notifications and retries a failed rebuild. Capture pauses for a
few milliseconds during a rebuild. The tap's rate is read again each time, and
need not match the output device (one Mac's tap delivered 48 kHz while every
output ran at 44.1 kHz). If the rate changed, the history, analyzers and note
overlay restart at the new rate so frequencies stay calibrated: bars drop and
refill. If capture delivers nothing for more than 250 ms mid-sound, for example
while it restarts or when a player pauses as Bluetooth headphones disconnect,
the missing time (up to 10 s) counts as silence, so bars decay instead of
freezing. Protected audio may be unavailable to system capture.

Bluetooth playback can reach your ears later than the captured samples. Native
system capture now enables **Auto sync to audio output** by default, delaying
analysis by macOS's reported output device latency, stream latency, safety
offset, and buffer duration. The estimate is refreshed once per second to follow
output changes. This queries properties outside the real-time audio callback.

Right-click the visualizer and open the **UI** tab to see the estimate and applied
delay. Use **Sync adjustment** to fine-tune: positive values delay the visuals
further, negative values reduce automatic compensation. Total visual delay is
limited to 0–1,000 ms. Bluetooth firmware/codec buffering may differ from the
reported estimate, so fine-tuning may still be needed. Existing manual delay
settings are retained as an adjustment; switch off automatic sync if you want
to retain an already calibrated manual delay. For `--device` capture, the
automatic estimate is unavailable and only the manual adjustment applies.

## Optional input-device / BlackHole routing

Use `--device NAME` for an existing loopback device or microphone through CPAL.
To set up BlackHole explicitly:

1. Install BlackHole 2ch (`brew install blackhole-2ch`) if it is not installed.
2. Open **Audio MIDI Setup** and create a **Multi-Output Device** containing both
   your speakers/headphones and BlackHole 2ch. Keep your physical output first;
   enable drift correction for BlackHole.
3. Select that Multi-Output Device as macOS sound output.
4. Give the built app/Terminal microphone permission when macOS asks. CoreAudio
   classifies input capture—including a virtual loopback—as microphone access.

Select an input by a case-insensitive name fragment:

```sh
cargo run --release -- --device "BlackHole 2ch"
```

Run independent named profiles to compare two configurations at the same time:

```bash
target/release/visual-audio --profile hybrid
target/release/visual-audio --profile resonator
```

Each profile has its own control window and settings file under
`~/Library/Application Support/VisualAudio/`. Running without `--profile`
continues to use the original `settings.json`.

Use **Up** or **+** to increase visual gain, **Down** or **-** to decrease it,
**H** to hide/show the detached controls window, right-click the visualizer to
show and focus it, and **Esc** to quit.

Controls apply immediately. Analyzer controls follow the selected method; inactive
settings are hidden. New profiles default to hybrid resonator + FFT with adaptive
level matching. DFT, temporal/frequency smoothing, manual hybrid FFT gains,
two-stage resonators, and transient assistance have been removed. Old DFT
profiles migrate to the resonator bank; obsolete fields are ignored when loading
and omitted on the next settings save. Other saved settings are retained.

The **UI → Show note activity at top** toggle displays a C–B ribbon above the
spectrum (enabled by default). **Note ribbon width** adjusts its centered span
from 20% to 100% of the window width; the default is 76%, and the choice is saved.
Its twelve labels sit on a continuous color gradient; note energy controls
their glow, and the dominant note grows brighter
and slightly larger. Matching notes across C1–B7 and the selected audio channels
contribute to one shared ribbon. **Midrange notes only (200 Hz–2 kHz)** is enabled
by default and restricts the ribbon's input without changing the spectrum.
Turn it off to include C1–B7 within the selected analyzer range.
**Fast note response** is enabled by default for quick arpeggios and dance music.
It measures spectral peaks directly from short audio windows (2,048 samples at
48 kHz, about 43 ms), independently of the spectrum's resonator Q, envelope,
display layout, and FFT update divisor. Peaks are interpolated between FFT bins
and folded into pitch classes. The current window chooses the winner, with a
1% lead margin, an immediate highlight, and a 12 ms glow fade time constant.
Short windows trade some accuracy for closely spaced low notes for response
speed. The audio output synchronization still applies.

**Note smoothing** adjusts the glow and label transitions from 0 to 250 ms
(a smoothing time constant). The default, 0 ms, preserves the snappy response;
30–60 ms adds a light crossfade, while larger values give a gentler look.
It changes the visible brightness and label size without adding memory to the
fast detector's winner selection. Changes apply immediately and are saved.

With **Fast note response** off, the ribbon reads the current spectrum, using
an 8 ms attack and 80 ms release, a 20 ms highlight attack and 60 ms fade, and a
5% lead margin. These are smoothing time constants. Activity dims during silence. This
shows measured pitch-class energy, including harmonics, rather than identifying
a song's key or separating melody from accompaniment. It follows the current
frequency range, analyzer resolution, and output synchronization. The overlay
also works with 3D and transparent backgrounds; disabling it releases its GPU
resources.

**FFT update rate** offers **1×, ½×, ⅓×, or ¼× display** in hybrid and standalone
FFT modes. At 60 visual frames per second these produce approximately 60, 30,
20, or 15 FFT updates per second. Audio continues accumulating between updates,
while bars, peaks, and resonators animate every visual frame. The default is 1×;
old fixed-rate settings migrate to that default. Lower rates save FFT work but
can add up to three visual frames of FFT response delay.

## Performance design

- The CoreAudio callback writes synchronized stereo into a preallocated,
  lock-free ring without allocating, locking, analyzing, or calling the GPU.
- Hybrid mode uses multirate GPU resonators below about 0.091 × sample rate,
  then half-rate and full-rate CPU FFTs above that boundary. Default FFT sizes
  are 1,024 and 512 samples with an Asymmetric Kaiser window (β = 8.1).
  Adaptive matching adjusts FFT levels against reference resonators.
- The resonator bank processes newly captured samples. Frequency-dependent Q
  varies smoothly between 50 Hz and 10 kHz. Its output envelope has configurable
  decay and peak/decay ordering. Optional multirate processing, a fifth /16 stage,
  and timestamp alignment remain available.
  Phase-corrected magnitude accounts for the finite filter history after reset,
  then uses the settled response. It preserves the resonator's natural attack
  instead of normalizing it away. Very short, poorly conditioned histories blend
  back toward the settled correction. Stable small-exponent calculations improve
  precision at low frequencies and high Q. This adds one scalar of state per bin
  within the existing auxiliary allocation, with no extra audio buffer or delay.
  The startup history begins at analyzer reset; this is not an onset detector
  or a way to remove the Q-dependent response time.
  Both envelope modes retain the responsive frame-end measurement instead of
  holding a maximum from earlier in the audio block. The /16 stage stays
  selected on empty updates, avoiding changes in how stored state is measured.
  The optional **Position + velocity** measurement estimates amplitude from
  displacement and normalized velocity together: `A ≈ sqrt(x² + (v/ω)²)`.
  It uses an exact adjacent-sample quadrature difference for a tone at the bin
  frequency, evaluated through the resonator recurrence to avoid cancellation.
  On-center gain is calibrated. One latest-sample scalar fits inside the existing
  auxiliary allocation; empty updates retain it. There is no block maximum,
  lookahead, amplitude smoothing, or additional peak hold. Startup and off-center
  response can differ from complex magnitude. The existing magnitude estimator
  already uses quadrature, so this alternative is not guaranteed to respond
  faster than it. Compare under **Resonator tuning → Level measurement**;
  the saved selection and default remain unchanged.
  **Position + velocity, π/2 pair** combines two complete position/velocity
  measurements, one current and one delayed by a quarter-cycle at that bin:
  `A = sqrt((E_now + E_delayed) / 2)`, where `E = x² + (v/ω)²` after calibration.
  Energy history is updated at each audio sample, with linear interpolation for
  fractional delays. Only this mode allocates the history; changing the range,
  multirate stages, or measurement resizes/releases it as needed. The second
  measurement is 5 ms old at 50 Hz and 15.625 ms old at 16 Hz. Fresh measurements
  still contribute immediately, but averaging can soften onset and extend tails.
  It targets phase-related energy ripple rather than guaranteeing less noise
  or faster response on every signal. The FFT portion of hybrid mode is unchanged.
  **Position only, π/2 pair** uses the real resonator displacement at those two
  times instead: `A ∝ sqrt(x_now² + x_delayed²)`, calibrated for on-center gain.
  It does not measure velocity. Fractional delay interpolates displacement before
  squaring; its response can still vary during transients or for off-center tones.
  It shares the on-demand quarter-cycle history and its timing tradeoff.
- Standalone 3-band and 5-band FFT modes and the auditory ERB analyzer remain
  selectable. FFT modes expose band sizes, decimation, crossover frequencies,
  and normalized Hann, Half-Hann, Half-Blackman, Kaiser, or Flanagan windows.
- Only active analysis pipelines and channel analyzers are created. Mono uses
  one analyzer; stereo creates a second on demand and releases it on returning
  to mono. FFT magnitude buffers fit the selected sizes. Inactive resampling
  buffers use minimal binding placeholders.
- 3D, transparency, axis-overlay, and peak-feedback pipelines are created when
  enabled and released when disabled. Full-size render/history/trail textures
  are allocated only for enabled effects; inactive texture bindings use 1×1
  placeholders. Resizing and effect changes recreate the required targets.
- GPU display composition handles bar gravity, peak momentum/hold, and rise
  highlights every visual frame. One instanced draw per layer renders up to
  2,048 bars without CPU-built meshes.
- Static frequency-grid calculations are cached in a full-precision, one-row
  GPU texture and refreshed when settings or window size change. Per-frame grid
  rendering reads that row instead of repeating logarithms and note mapping at
  every pixel. The cache uses four bytes per window pixel of width, rather than
  a full-window history texture. Resolution, bar count, effects, and frame cadence
  remain unchanged. Final 8-bit pixel rounding can differ by one channel step.
- Streaming audio uploads include only the suffix consumed by the analyzer.
  At 48 kHz and 60 updates/second that is typically 800 samples, rather than
  copying all 16,384 samples each frame. Resets upload the full required warm-up
  window, and no-sample updates skip the upload. Analyzer math is unchanged.
- Flat bar lighting is calculated per vertex instead of repeating the same
  color and brightness calculations at every pixel. Body glows below their
  existing brightness threshold and peak glows with exactly zero strength are
  clipped before rasterization. Visible glow arithmetic remains unchanged.
- The note ribbon fetches its font atlas only inside each label's bounds. In
  flat opaque mode it shares the spectrum's render pass, avoiding another
  attachment store/load. Transparent and 3D modes keep the final overlay pass.
- The default frequency range is 16 Hz–22.5 kHz, capped below input Nyquist.
  The smooth floor curve defaults to -51 dB at the low end and -62 dB at the
  high end. Fixed-width bars support a configurable frequency focus lens;
  musical-note layout gives each semitone equal width.
- Frames follow the display's actual refresh rate through FIFO VSync (Metal
  has no Mailbox mode). Each frame first waits for a drawable, then reads the
  newest audio, so samples do not age during that wait, and at most one frame
  is queued (two drawables). The next frame is requested half a refresh
  interval after the previous one, so the drawable wait aligns it with the
  display without missing a refresh. Measured with the default hybrid
  profile: an even 120 fps at 120 Hz (previously about 103 fps from a timer
  that drifted later each frame), and an even 60 fps when Low Power Mode
  limits the display to 60 Hz. At 60 Hz, audio read for a frame is under 1 ms
  old at presentation instead of about 7 ms, and one fewer frame waits in the
  queue. GPU work is briefly deferred during live window movement and resize.
  Axis-label layout runs only within the ten-pixel-high text strip; overlay
  draws are scissored to that strip.
- While the visualizer window is fully hidden, rendering stops (macOS can
  withhold drawables for up to a second). The capture ring is still drained
  every 250 ms, and analyzers restart from the continuous history when the
  window is visible again.
- **Idle in silence.** Once the newest audio is silent and every bar, peak,
  glow and trail has had time to settle, redraws pause and the settled frame
  stays on screen. Audio is still checked every 20 ms, and drawing resumes
  as soon as it returns, with analyzers restarting from the continuous history.
  Silence means samples 30 dB below the lowest dB floor after gain, or no
  samples for 250 ms (system capture delivers none while no app plays). The
  settle time follows the current decay settings and output delay: about
  4.2 s with the defaults, longer for slow peak gravity or 3D history and
  trails. Changing a setting, resizing, or revealing the window draws again
  until things settle. Measured at 60 Hz: about 0.3% CPU while idle, compared
  with about 10% while drawing.

Run CPU and shader-validation checks with `cargo test --offline`. The additional
Metal resource check can be run on a Mac with
`cargo test --offline gpu_analyzers_validate_with_lazy_buffers -- --ignored`.
The note overlay has a separate Metal check for octave folding, stereo input,
frequency focus, note layout, silence, fast audio input, and opaque/transparent rendering:
`cargo test --offline gpu_note_activity_tracks_octaves_stereo_release_and_renders -- --ignored`.
CPU tests also check actual synthesized PCM across common sample rates and
180 BPM sixteenth-note sequences, including quieter notes after louder ones.

The tap rebuild that output changes trigger can be checked against real system
capture, without playing sound or changing any setting. It closes capture
during pending and running rebuilds, and, if an app is playing audio, checks
that samples keep arriving after each rebuild:
`VISUAL_AUDIO_CAPTURE_TEST=1 cargo test --offline --lib system_capture_keeps_delivering_across_rebuilds -- --ignored --nocapture`.

The static-grid regression compares the original and cached shader across 96
configurations, including 1080p, stereo, labels, transparency, focus, and resize.
It allows at most one 8-bit rounding step per channel and reports isolated grid
draw completion times (CPU submission plus GPU completion, excluding analysis,
bars, and cache refresh):
`cargo test --offline gpu_cached_grid -- --ignored --nocapture`.

To compare 1 px and 10 px bars with the saved default hybrid/flat profile, run
`cargo test --offline gpu_bar_width_cost_breakdown -- --ignored --nocapture`.
This read-only diagnostic reports bin counts and separate completed-batch times
for GPU analysis and bar drawing, including CPU encoding overhead. It uses
synthetic audio and equal-height bars and excludes grid, labels, note ribbon,
CPU FFT, and presentation. Ten times fewer bins does not mean ten times less
total GPU time: dispatch costs and most pixel coverage remain, and bin processing
runs in parallel. Fixed-width layouts also have a minimum of 32 bins per channel.

Bar-render optimization has 144 original-versus-optimized pixel comparisons
covering 1/3/10 px widths, mono/stereo, mirrors, fades, silence, and highlights:
`cargo test --offline gpu_bar_render_optimization -- --ignored --nocapture`.
The note ribbon has another 144 comparisons for its atlas sampling and merged
pass: `cargo test --offline gpu_note_ribbon_render_optimization -- --ignored --nocapture`.
Both checks allow at most one 8-bit rounding step per channel; the ribbon
comparisons currently match exactly. Resolution, frame rate, and effects are
unchanged. Timings report isolated completed batches, not whole-app GPU usage.

The numerical magnitude regression test requires Metal access:

```sh
cargo test --offline phase_corrected_magnitude_matches_weighted_tone_fit -- --ignored --nocapture
```

It compares the production WGSL correction with an independent weighted
least-squares fit across 1,440 synthetic cases (phase changes, detuned tones,
mixtures, and bursts), checks bounded output during short histories, and exercises
streaming updates, empty updates (including /16), and resets through the full
resonator shader. A short-burst regression checks that both envelope modes preserve frame-end
measurement and decay evenly over the following silent frames.

The same GPU test validates position/velocity measurement against 1,440 independent
adjacent-position references, including displacement near zero, and checks the
full streaming short-burst release in both measurement modes.

Quarter-cycle pairing is checked through the production Metal shader against an
independent fractional-delay energy reference and subsequent release frames.

## WaveAudio — separate waveform app

WaveAudio is an independent executable and macOS app. Both apps share
`visual_audio::audio`, including native capture and output-device latency
reporting. WaveAudio does not read or change VisualAudio's saved settings.
`cargo run` still starts VisualAudio.

```sh
sh scripts/build-wave-audio-app.sh
open target/WaveAudio.app
# Synthetic visual inputs; no capture or sound playback:
cargo run --release --bin wave-audio -- --demo
cargo run --release --bin wave-audio -- --demo-tone 59.94
cargo run --release --bin wave-audio -- --demo-beats
# A fixed 60 Hz timebase is the Min=Max special case:
cargo run --release --bin wave-audio -- --demo --min-hz 60 --max-hz 60
# Optional microphone/loopback input:
cargo run --release --bin wave-audio -- --device NAME
```

There is **one waveform**, with reference F varying continuously from Min at
screen left to Max at screen right. Defaults are 60–6000 Hz (limited by the input
sample rate), logarithmic spacing, continuous scrolling (phase lock off) and
target 120 FPS.
Set Min=Max to recover the fixed-F oscilloscope timebase.

For `u = x / width` increasing from **left to right**:

```
F(u) = Min * (Max / Min)^u          # Log spacing enabled
F(u) = Min + (Max - Min)*u          # Log spacing disabled
y(u) = audio(t_anchor - (1-u)/F(u))
```

These are direct reads from the same raw audio buffer with linear interpolation.
The changing timebase stretches the full signal horizontally; it does not isolate
frequencies or give individual input tones separate propagation speeds. Frequency
labels beneath the curve show F(u).

- **Lock phase to Min:** use one shared endpoint
  `t_anchor = floor(t * Min) / Min`, with t measured in captured sample time after
  output-delay compensation. A tone exactly at Min (or an integer harmonic)
  stays stationary. All x positions share that endpoint: independently quantizing
  by F(u) would introduce spatial discontinuities. This does not lock every
  frequency simultaneously and is not automatic frequency detection.
- **Continuous scrolling (default):** with phase lock disabled, use the current delayed sample
  time directly. At Min=Max=60 Hz and 120 FPS, two successive frames of a pure
  60 Hz tone have opposite phases unless phase lock is enabled.
- **Target FPS:** defaults to 120, independently of F(u). Actual presentation is
  limited by the display/VSync and rendering work. All sampling positions come
  from audio timestamps, not a per-frame scroll increment.
- **Dense portions:** each pixel uses extrema of its actual raw-sample interval
  when that interval spans more than one sample. The continuous signed curve is
  also drawn; there are no additional frequency traces. Missing history stays
  blank. At high reference frequencies only a few captured samples represent a
  period, so interpolated shapes are limited by the sample rate.
- **Output timing:** native capture follows reported device delay, with a signed
  manual adjustment. Phase lock can hold the endpoint for less than one Min
  period. Left and Right can be selected without mixing channels.

There is no Fourier transform, filter bank, dispersion simulation, envelope or
smoothing in WaveAudio. Four seconds of history cover the minimum 1 Hz reference,
phase hold and up to 1.5 seconds of output compensation. Controls last for the
current launch and are not saved. VisualAudio's spectrum analyzers are independent.
The previous multiple-row controls and `--multiple` / `--frequency` flags have
been removed in favor of `--min-hz` and `--max-hz` on the single curve.
