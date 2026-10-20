# VisualAudio

A low-latency macOS audio visualizer written in Rust. By default it captures
system audio through a native Core Audio process tap on macOS 14.2 or newer,
moves synchronized stereo samples through a lock-free SPSC ring, and
runs spectral analysis plus rendering on the GPU through `wgpu`/Metal and WGSL.
The optional transparent-background mode renders normally onto opaque black,
then a GPU compositor converts RGB light intensity into a clean premultiplied
alpha channel. It is disabled by default; the detached controls stay opaque.
The visualizer uses a hidden, transparent macOS title bar: the traffic-light
buttons remain available and dragging anywhere in the canvas moves the window.
Visualizer controls are restored on startup from
`~/Library/Application Support/VisualAudio/settings.json`; edits are saved
automatically after a short debounce.

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
target/release/visual-audio --profile dft
target/release/visual-audio --profile resonator
```

Each profile has its own control window and settings file under
`~/Library/Application Support/VisualAudio/`. Running without `--profile`
continues to use the original `settings.json`.

Use **Up** or **+** to increase visual gain, **Down** or **-** to decrease it,
**H** to hide/show the detached controls window, right-click the visualizer to
show and focus it, and **Esc** to quit.

The in-window controls apply immediately without shader recompilation. They
include visual gain, optional attack and decay smoothing for all five frequency
bands, adjustable bar brightness and peak indicators, a 0–1,000 ms visual delay
for output-device synchronization, global and independent per-band DFT sample
sizes, fixed pixel bar width, adjustable 0–12 px bar spacing with a
resize-responsive bar count, and a reset
button.

## Performance design

- The CoreAudio callback writes synchronized stereo into a preallocated,
  lock-free ring: no locks, allocations, FFTs, logging, or GPU calls occur there.
- The render loop uploads a 16,384-sample history (64 KiB) each frame, reserving
  enough recent audio for every band's largest selectable window.
- GPU FIR preprocessing builds five analysis rates. Their default DFT sizes are
  sub-bass ÷64 (128 samples), bass ÷32 (256), low-mid ÷8 (512), upper-mid ÷2
  (1,024), and high full-rate (2,048); each can independently select half,
  standard, or double size. Adjacent bands overlap and crossfade around 80 Hz,
  320 Hz, 1.28 kHz, and 5.12 kHz, preventing visible crossover steps. A 512-way
  compute pass maps these streams onto a configurable log-spaced display (35
  Hz–22 kHz by default, selectable from 16–200 Hz at the low end and 10–40 kHz
  at the high end). A smooth two-control frequency lens can expand frequencies
  around a logarithmic origin or compact them in favor of bass and treble;
  50% strength is the exact unmodified logarithmic scale.
- An optional `1× + 2× Adaptive` mode uses the short window for fresh attacks,
  then settles onto the sharper double-length spectrum. Its response amount is
  configurable and defaults to 50%. A `0.5× + 1× + 2× Adaptive` mode adds an
  independently tracked half-length leading-edge stage for the quickest
  response while retaining the 2× steady spectrum.
- Each band is evaluated directly at every log-spaced display frequency,
  avoiding the low-bin stair stepping that FFT visualizers need interpolation
  to hide. There is no separate responsive-bass or leakage-cleanup path.
- Temporal smoothing is optional and disabled by default, exposing each
  frame's direct DFT result.
- Multirate FFT analysis is available in independent 3-band and 5-band modes.
  The 5-band default uses crossovers at 80 Hz, 320 Hz, 1.28 kHz, and 5.12 kHz,
  with decimation factors ÷128, ÷64, ÷16, ÷4, and ÷1. Every band exposes its
  FFT size and decimation, and adjacent spectra crossfade smoothly.
- Higher-band decay smoothing is also frequency-dependent: 320 Hz–1.28 kHz
  uses 0.78, 1.28–5.12 kHz uses 0.84, and 5.12–22 kHz uses 0.87.
- All five analysis bands share a selectable Hann, causal Half-Hann,
  Half-Blackman, Asymmetric Kaiser, or Flanagan auditory-model window with
  coherent-gain normalization. Kaiser β is configurable from 0–16 and
  Flanagan α from 2–24; both default to 8. Half-Blackman is the default.
- A single instanced draw renders a window-width-dependent bar count (up to
  2,048); there are no CPU-built meshes.
- The GPU-rendered frequency axis defaults to 35 Hz–22 kHz and can be adjusted.
  Its actual upper limit is capped below the input Nyquist frequency. The
  vertical mapping uses a configurable smooth floor curve, defaulting to -56
  dB at the low endpoint and -78 dB at the high endpoint, with no fixed dB axis.
- Mailbox VSync (FIFO fallback) is automatically paced to the active display
  refresh rate without blocking the macOS event thread. GPU work is deferred
  briefly during live window movement and resize.

The current analyzer is a log-spaced GPU DFT, chosen because it directly spends
work on visually useful bins. For very large linear spectra (8K–64K samples), a
multi-pass radix-2 FFT would scale better.

An experimental GPU resonator-bank analysis method is also available for direct
comparison. It maintains one complex state per displayed frequency and normally
processes only newly captured samples. Its frequency-dependent Q controls are
smoothly interpolated between 50 Hz and 10 kHz. A separate output-decay control
expresses exponential amplitude decay as a dB amount over a time interval. Its
envelope can either decay the previous value before capturing a new peak, or capture
and display the peak first while saving its decayed value for the next update.
Neither mode uses a sample memory window.
Frequency mapping, delay, resonator tuning, or bar-layout changes reset and warm
those states from the available history.
