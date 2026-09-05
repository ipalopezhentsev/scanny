# scanny

Tethered control of a Nikon DSLR from Windows, in Python: live view in a
desktop window, magnification, click-to-focus, exposure control and shutter
release. Written against a **Nikon D750**.

```
uv run scanny
```

## Why this exists, and how it talks to the camera

libgphoto2 is not an option on Windows, and digiCamControl is C#. The awkward
part is that Windows binds MTP cameras to its own `WUDFWpdMtp` driver, so the
raw USB pipe that libgphoto2 uses is unavailable unless you replace the driver
for the device — which breaks every other application on the machine.

The way through is **WPD's MTP extension commands**. `IPortableDevice::SendCommand`
carries a PTP opcode, its parameters and a data phase straight to the camera,
without touching the driver binding. Two things about it are worth knowing,
because neither is obvious from the documentation:

- **`READ_DATA` will not allocate for you.** The caller has to put a
  correctly-sized buffer into the request under `WPD_PROPERTY_MTP_EXT_TRANSFER_DATA`;
  omit it and the read fails with `ERROR_NOT_FOUND` *and* wedges the transfer
  context, so the following `END_DATA_TRANSFER` fails too.
- **Standard PTP opcodes work.** `GET_SUPPORTED_VENDOR_OPCODES` lists only what
  the camera advertises as vendor extensions, but the driver forwards
  `GetDeviceInfo`, `GetDevicePropDesc` and `Set/GetDevicePropValue` as happily
  as it forwards Nikon's `0x9xxx` operations. Without that, exposure control
  would be impossible: Windows surfaces none of the camera's exposure
  properties through the ordinary WPD property API — only a firmware string, a
  model name and a serial number.

The COM interfaces are declared by hand in `wpd/_com.py` rather than generated
from the typelibs, because `comtypes.client.GetModule` mis-marshals them:
`IPortableDeviceManager::GetDevices` silently reports zero devices, and
`GetIPortableDevicePropVariantCollectionValue` loses its out parameter
entirely.

Measured on a D750, this transport sustains about **110 live-view reads per
second**, so the camera, not the plumbing, is the limit.

## Layout

| Module | What it does |
| --- | --- |
| `wpd/_com.py` | Hand-written WPD COM interfaces, `PROPVARIANT`, `PROPERTYKEY` |
| `wpd/constants.py` | Property keys from `PortableDevice.h` and `WpdMtpExtensions.h` |
| `wpd/device.py` | Device enumeration, and the three PTP transaction shapes |
| `ptp/codes.py` | Operation, response, event, property and datatype codes |
| `ptp/parser.py` | PTP's little-endian datasets: `DeviceInfo`, `DevicePropDesc` |
| `ptp/session.py` | Typed property get/set against a transport |
| `camera/nikon.py` | Live view, focus, zoom, exposure, capture |
| `camera/values.py` | Raw PTP integers to labels a photographer recognises |
| `ui/` | Qt window, live-view canvas, and the camera worker thread |

## The live-view header

Every live-view response is a big-endian header followed by a JPEG (384 bytes
of header on a D750; the parser scans for the SOI marker rather than trusting
that). The geometry was mapped against the camera itself:

| Offset | Meaning |
| --- | --- |
| 8, 10 | Size of the JPEG image |
| 12, 14 | The autofocus coordinate space — the full sensor frame, constant |
| 16, 18 | Size of the region on display, in AF-space units; shrinks as you zoom |
| 20, 22 | Centre of that region. When magnified, it follows the focus point |
| 24, 26 | Size of the focus box |
| 28, 30 | **Centre of the focus box** — the field that moves |

The focus point lives at offsets 28/30, not 20/22; 20/22 is the crop centre,
which sits dead centre of the frame at full-frame view and is easy to mistake
for the focus point. Coordinates are the box *centre*, and the camera clamps
them so the box stays inside the frame — exactly half a focus box in from each
edge, which on a D750 with its 324×270 box means x within 162–5854.

None of these numbers are hardcoded: the AF space depends on the body's Lv
selector. In photo position the frame is 6016×4016 (3:2); in movie position it
is 6016×3376 (16:9). Everything is read from each frame's header, so both work
without special cases.

Because 16/18 and 20/22 describe the visible crop, one mapping converts a click
to a focus coordinate at every magnification:

```python
af_x = crop_centre_x + (click_fraction_x - 0.5) * crop_width
```

## Live-view resolution

`NIKON_LiveViewImageSize` (0xD1AC) offers a D750 exactly two settings, and the
larger is requested every time live view starts:

| Setting | Frame (photo, 3:2) | Frame (movie, 16:9) |
| --- | --- | --- |
| 1 | 320×212 | 320×180 |
| 2 (used) | **640×424** | **640×360** |

So 640 pixels across is the ceiling over PTP — the full-frame view is
inherently soft, and that is the camera, not a bug in the decoding. Magnifying
does not resample: the camera re-renders a smaller part of the sensor into the
same 640-pixel frame, which is why a zoomed view looks sharper. The JPEG never
changes size; only the sensor area it covers does. Nothing is cropped on the
computer.

## Exposure preview

By default a Nikon holds the lens wide open in live view and normalises
brightness, so changing aperture, shutter or ISO makes no visible difference —
mean frame brightness stays flat around 85 from f/4 to f/22.
`NIKON_LiveViewExposurePreview` (0xD1A6) switches that off, and then the lens
physically stops down and live view shows the exposure that would be taken:

| Aperture | f/4 | f/5.6 | f/8 | f/11 | f/16 | f/22 |
| --- | --- | --- | --- | --- | --- | --- |
| Mean brightness, preview on | 231 | 209 | 168 | 130 | 104 | 79 |
| Mean brightness, preview off | 94 | 90 | 87 | 89 | 93 | 96 |

Two things about the property are easy to trip over: it is **inverted** (0
means preview on, 1 means off), and it is only *writable while live view is
running* — a descriptor read with live view stopped reports it read-only. The
camera therefore remembers the wanted state and applies it each time live view
starts. It is on by default, which does mean a badly set manual exposure shows
up as a very dark or very bright image; that is the point of it, and the
checkbox turns it off.

## How the click gestures fit together

| Gesture | Action |
| --- | --- |
| Click | Move the focus rectangle there. Nothing else |
| Double click | Autofocus, without magnifying |
| Right click | Magnify fully, or back to the whole frame if already magnified |
| Drag | Magnify onto the dragged region |
| Arrow keys | Pan the magnified view |
| Scroll | Step magnification |
| Enter | Autofocus |
| `[` `]` | Manual focus, minimum increment, nearer / further |
| `,` `.` | Manual focus, fine increment |
| `<` `>` | Manual focus, coarse increment |
| Esc or 0 | Back to the whole frame, whatever the current state |

Aiming, magnifying and focusing are three separate acts, which is what makes
the ordering work out. Qt delivers a double click as **press, release,
double-click, release** — verified against the real Windows input queue, not
just `QTest` — so the opening click is always acted on first. Here that click
only moves the rectangle, which is exactly the thing you want to have happened
before focusing, so nothing has to be suppressed and no gesture is held back
waiting to see whether a second click arrives.

That is also why `focusRequested` carries **no coordinates**: the rectangle is
already where the user pointed, so the focus simply runs there.

The usual sequence is click to aim, right-click to magnify and check, arrow
keys to explore, right-click again to come back out.

Right-click reads whether it is magnified from the live frame rather than a
remembered zoom level, so it still does the right thing if the magnification
was changed on the camera body.

### Testing pointer gestures honestly

`QTest` injects events straight into the widget and skips the platform layer,
so it cannot tell you whether a real click would arrive at all — and it does
not even reproduce the sequence: `QTest.mouseDClick` sends only the
double-click, omitting the ordinary click Windows delivers first. The tests
here therefore build that four-event sequence by hand.

To check the platform layer, drive the real input queue with `mouse_event`.
Two things will otherwise give you a confident false negative:

- **A background process cannot raise its own window.** `SetForegroundWindow`
  fails silently, the synthetic click lands on whatever is really in front,
  and the test reports a failure that has nothing to do with the code. Attach
  to the foreground thread's input state first, then assert
  `GetForegroundWindow()` is yours.
- **Qt reports logical coordinates; the input queue wants physical pixels.**
  At 150% scaling a point from `mapToGlobal` is two thirds of where you meant,
  so the click misses the widget — or the window. Multiply by
  `devicePixelRatio()`, and assert `WindowFromPoint` really is your window
  before clicking.

## Manual focus

`NIKON_MfDrive` takes a direction and a step count, and runs in about 0.1s.
There are four increments, and **how many drive steps each one is worth is
yours to set** -- the Focus panel has a box under each pair of buttons, and the
values are remembered between runs. Both the buttons and the keyboard read from
those boxes, so there is one place to change and nothing else has a step count
baked in.

| | minimum | fine | medium | coarse |
| --- | --- | --- | --- | --- |
| Default steps | 18 | 50 | 250 | 1000 |

The buttons auto-repeat, so holding one keeps driving. A single step is the
finest the body accepts, and it does take it -- useful for critical focus when
magnified, where holding the key repeats it.

The defaults come from measuring a D750: about 6000 steps covers the whole
range of travel, 800 clearly shifts focus, 400 is slight, and under about 100
is lost in frame-to-frame noise at full frame. How far a step actually moves
focus depends on the lens and the subject distance, which is exactly why the
values are adjustable rather than calibrated.

Two caveats worth knowing:

- **Which direction is nearer was not confirmed.** It follows Nikon's usual
  convention. It could not be settled from live view here, because at this
  subject distance both ends of travel blur the scene about equally (sharpness
  47.7 against 46.6, well inside the noise). If a lens focuses backwards, swap
  `_FOCUS_NEARER` and `_FOCUS_FURTHER`.
- **The body does not report the end of travel.** Driving hard into the stop
  keeps returning success rather than `MF_DRIVE_STEP_END`, so "already at its
  limit" only appears if the camera volunteers it.

Focus driving answers `DEVICE_BUSY` while the body settles after a zoom or a
previous move, which auto-repeat runs into readily, so it retries rather than
failing.

### The keyboard has to stay on the image

The shortcuts are handled by the live-view widget, so anything else taking
keyboard focus silently disables all of them until the image is clicked again —
which shows up as "the shortcuts only work sometimes". Every pointer-operated
control (buttons, checkboxes, the zoom slider) is therefore `NoFocus`, and the
combo boxes, which do need focus for their popups, hand it back once a value is
chosen. There are tests asserting this, because it is invisible until someone
notices the keys have gone dead.

## The level sensor

The header carries the body's virtual horizon: **offset 52 is roll** and
**offset 56 is pitch**, both in whole degrees, wrapping through 359 for
negative angles, with `0xFFFF` meaning the sensor has no reading. Pitch drops
out at steep angles; roll was always available in testing. Both are shown in
the status bar, and turn green when the camera is within a degree of level on
both axes.

The two fields were identified by recording the header for thirty seconds while
the camera was tilted: offset 52 tracked rolling left and right, offset 56
tracked pitching nose up and down. Offsets 54 and 58 change every single frame
with no relation to either movement, so they are left alone.

## Focus distance is not available

The camera does not report it. Neither the live-view header nor any of the 217
Nikon vendor properties changes with focus: driving focus from one end of
travel to the other moves only `0xD1B1`, the exposure meter, which is
responding to the scene going out of focus rather than reporting distance. Two
properties that looked promising at first, `0xD067` and `0xD07D`, turn out to
drift on their own with time — a control run with no focus change moves them
just the same.

## Zoom levels

`NIKON_LiveViewImageZoomRatio` advertises a 0-7 range, but a D750 rejects
level 1 outright with `InvalidDevicePropValue` -- it jumps from full frame
straight to 2.35x. The levels that work, and the magnification each gives:

| Level         | 0    | 2     | 3     | 4    |   5   |   6  |   7   |
| ------------- | ---  | ---   | ---   | ---  | ---   | ---  | ---   |
| Magnification | 1.0x | 2.35x | 3.13x | 4.7x | 6.27x | 9.4x | 18.8x |

The zoom slider steps through those levels rather than a raw 0-7 range, and the
table is corrected from live frames as levels are used, so drag-to-magnify
follows the body rather than a fitted curve.

## Capture

`InitiateCaptureRecInMedia` takes two parameters, and both are fixed: any
storage (`0xFFFFFFFF`) and a second parameter of **zero**. A second parameter
of one is accepted without error and then never completes -- no
`CaptureComplete` event ever arrives, and the pending capture wedges the body
so that live view will not restart until it is power cycled. Autofocus before a
shot is therefore driven as its own `AfDrive` step, so a failure to lock is
reported before the shutter is asked to fire.

A shot lands on the card at full resolution (a 23 MB NEF here) and downloads
over USB at about 23 MB/s. Live view keeps running throughout.

## Using it

- **Click** the image to move the focus rectangle there. It does not focus and
  does not magnify.
- **Right-click** to magnify all the way in on the rectangle, and right-click
  again to come back to the whole frame.
- **Double-click** to autofocus, without magnifying. The click that opens the
  gesture has already put the rectangle where you are pointing.
- **Drag a box** to magnify that region. The camera centres its magnified view
  on the focus point, so this puts the focus point at the middle of your
  selection and picks the strongest magnification that still shows all of it.
- **Scroll** to step magnification in and out. **Esc**, **0** or the button
  always returns to the whole frame.
- **Arrow keys** pan the magnified view. The camera has no pan command of its
  own — it centres the magnified view on the focus point, so panning moves that
  point, by an eighth of whatever is on screen per press. The view therefore
  travels the same visible distance at every magnification, and panning never
  drives autofocus.
- **Exposure preview** is on by default, so aperture, shutter and ISO changes
  are visible in live view, depth of field included.
- The status bar shows the live-view frame size and the rate it is arriving at
  (about 30fps).
- Shutter, aperture, ISO, exposure compensation and white balance are settable;
  the exposure mode, focus mode and drive mode are shown but are set on the
  body, so the camera reports them read-only.
- Shots go to the card at full resolution and are downloaded to
  `~/Pictures/scanny` unless you turn that off.

Exposure time is reported by PTP in whole tenths of a millisecond, which cannot
distinguish 1/8000 from 1/6400 from 1/5000. `camera/values.py` snaps to the
nearest standard speed from a table that omits the two ambiguous ones, so a
body whose fastest speed is 1/4000 reads back exactly.

## If live view will not start

`DEVICE_BUSY` on `StartLiveView`, while the camera answers every other command
and reports no prohibit condition, means the body is wedged by an operation
that never finished. Nothing clears it over USB — not `TerminateCapture`, not
`AfDriveCancel`, not `EndLiveView`, not reopening the device, not toggling
application mode. The camera has to be switched off and on, and the app says
so rather than showing the response code.

Idling is *not* a cause: ninety seconds untouched and live view still starts
normally. Every wedge seen while building this came from an operation left
outstanding, which is why the capture path now terminates a capture that fails
to complete.

Other refusals are reported in plain language from Nikon's prohibit bitmask
(card missing, battery exhausted, mirror up, and so on).

## Tests

```
uv run pytest
```

The tests cover the byte-level work — PTP encoding, the live-view header
geometry, and the value formatting — against data the camera actually sent.
They need no camera attached.


## TODO

- set minimum step size to 18 - only it makes sound, at least with 24-120
- light-gathering mode where several LV images are aggregated to cancel noise!
- persist such settings to user profile
- MLU
- filenames
- filmstrip/delete file?
- sharpness measurer - i.e. i manually focus via buttons and it evaluates. or it drives until maximizes sharpness in given area by me
- focus sweep - find&visualize depth map
- don't write file to card, just to pc
- histogram
- preview of full frame, so i can orient when zoomed-in