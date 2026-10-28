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
Visualizer controls are restored on startup from
`~/Library/Application Support/VisualAudio/settings.json`; edits are saved
automatically after a short debounce.
New profiles and **Reset** use the saved configuration adopted on October 20,
2026: gain 1.7, hybrid resonator + FFT with adaptive matching, 3 px bars, a bottom
mirror, and a 16 Hz–22.5 kHz frequency range. Existing profiles keep their saved
settings. Automatic output synchronization is enabled by default.

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

Capture reads the tap's actual sample rate at startup. Restart VisualAudio after
changing output devices or sample rates if capture stops or the spectrum becomes
miscalibrated; automatic tap rebuilding is not implemented. Protected audio may
be unavailable to system capture.

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
- The default frequency range is 16 Hz–22.5 kHz, capped below input Nyquist.
  The smooth floor curve defaults to -51 dB at the low end and -62 dB at the
  high end. Fixed-width bars support a configurable frequency focus lens;
  musical-note layout gives each semitone equal width.
- Mailbox VSync (FIFO fallback) follows the active display's refresh rate. GPU
  work is briefly deferred during live window movement and resize. Axis-label
  layout runs only within the ten-pixel-high text strip; overlay draws are
  scissored to that strip.

Run CPU and shader-validation checks with `cargo test --offline`. The additional
Metal resource check can be run on a Mac with
`cargo test --offline gpu_analyzers_validate_with_lazy_buffers -- --ignored`.

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
