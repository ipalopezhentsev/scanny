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
second**, so the camera, not the plumbing, is the limit. A read costs about
9ms and the frame size does not change that — 320x180 reads no faster than
640x360 — so what the transport is spending is per-transaction latency rather
than bandwidth, and the larger live-view frame is free.

The camera draws about **44 new frames a second** (a fresh one every 23ms),
which is where the useful ceiling actually sits: past that, reads come back
holding a frame that was already here. Two things are worth knowing before
trusting that number anywhere:

- **Magnification decides it.** Measured on a D750, in new frames a second:

  | magnification | preview on | preview off |
  | --- | --- | --- |
  | 1.0x – 3.13x | **44** | 30 |
  | 4.7x – 18.8x | **16** | 16 |

  Past 4.7x the body draws at a third of the rate, and the exposure preview
  stops mattering. This is the camera, not the transport: reads are *faster*
  at high magnification (5.9ms against 8.2ms), and what stretches is the gap
  between distinct frames, from 23ms to 63ms.

  The **live-view frame size does not come into it** — 320x180 is drawn at
  exactly the rate 640x360 is, at every magnification — so asking for the
  small frame is not a way to buy the rate back. Beware of measuring this the
  obvious way: toggling the exposure preview resets the body's zoom to 1.0x,
  so a sweep that sets the preview between readings will report high
  magnifications that were never in force. Read `magnification` back off the
  frames you actually measured.

  It is, at least, indifferent to shutter speed: flat at 44fps across the
  whole range from 1/500 to a full second, so the body is not integrating the
  sensor for the set exposure to draw live view.
- **Nothing in the code relies on it.** `ui/worker.py` polls every 15ms,
  faster than the camera draws, and discards the reads that come back
  unchanged. That is what keeps frame integration honest when the rate moves —
  an averaged stack only cancels noise if its frames are genuinely different
  from each other, because a duplicate carries identical noise and adds
  coherently instead of averaging down.

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
| `ui/integration.py` | Averaging consecutive frames to cancel sensor noise |
| `ui/sharpness.py` | Scoring the contrast in the displayed picture, to focus against |
| `ui/trend.py` | The plot of recent readings that focus is driven against |
| `ui/hunt.py` | Walking focus to the top of that reading, without counting steps |
| `ui/navigator.py` | The whole frame, with the magnified view marked on it and draggable |
| `ui/histogram.py` | The levels in the picture on screen, per channel |
| `ui/pixels.py` | Getting at a QImage's bytes as numpy, shared by everything that reads them |
| `ui/naming.py` | The prefix-and-counter names that downloaded pictures are saved under |

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

## Integrating frames to cancel noise

Live view is 640 pixels across and the camera renders it fast rather than
well, so at anything above base ISO the picture is grainy -- and the grain is
different in every frame while the scene is not. Averaging N frames therefore
keeps the picture and divides the noise by roughly the square root of N. There
is no exposure control to reach for here: the camera sends what it sends, about
forty-four frames a second (see the table above — sixteen, if you are magnified
past 4.7x), so the only currency is frame rate. **Integrate**,
in the Live view panel, spends it: the mean of each N frames, arriving 44/N
times a second. Only frames the camera has actually redrawn are counted — a
re-read of the frame already on hand carries identical noise, which adds
coherently rather than averaging down, so counting one would finish the stack
early and leave it grainier than the number of frames claims.

Measured against synthetic frames with a known amount of noise on them, sixteen
frames behave exactly as the arithmetic says, and cost about a tenth of the
frame interval to compute:

| | one frame | 16 frames |
| --- | --- | --- |
| Noise, levels RMS | 8.68 | 2.19 |
| Mean brightness | 126.78 | 126.81 |
| Cost per frame | -- | 2.75 ms, against 33 ms between grabs |

The brightness row is the point of the second design decision below: the
picture gets cleaner without getting lighter or darker.

Four things it does deliberately:

- **Every frame is still published, only the picture waits.** The focus box and
  the level readout come from the frame header, not the JPEG, so they keep
  following the camera at its full rate while a stack fills. Only the image
  underneath them updates at 30/N.
- **A frame that does not belong to the stack is shown at once**, and starts a
  new one. Magnifying, panning or switching the body between its photo and
  movie positions all change what the camera is rendering, and waiting out the
  rest of a stack before showing the new view would make every pan feel like it
  had frozen. So the first frame of a new view appears immediately -- noisy --
  and the integrated version replaces it a stack later.
- **The mean is taken in the JPEG's own gamma-encoded values**, not in linear
  light. Averaging linearly is the physically correct way to add exposures, but
  it lifts the shadows and changes how the picture looks; the point here is the
  same picture with less noise in it.
- **The frame rate in the status bar is the rate the picture updates at**, not
  the rate the camera sends at, which integrating does not change. Reporting 30
  while the image changes four times a second would be a lie about what is on
  screen.

The switch and the count are remembered between runs. Sixteen frames is half a
second an image, which is comfortable for composing on a static subject; past
about 32 the view has stopped being live in any useful sense, and 64 is the
limit.

## Focusing against a number

Focus is whatever setting puts the most contrast in the picture, which is what
a camera's contrast-detect autofocus hunts for. A D750 will only hunt inside
its own focus box, and that box is 324 sensor pixels across -- far larger than
the things worth focusing this carefully on. **Measure sharpness**, in the
Focus panel, scores the picture instead: drive focus a step, and keep the
direction that raises the number.

Magnifying is the first half of aiming it. The reading covers what is on
screen, so at 18.8x the frame *is* a small part of the sensor -- smaller than
the focus box, and anywhere you like rather than where the box will go.

The second half is that magnification runs out. Level 7 is as far in as the
body goes, nothing is cropped on the computer, and the subject worth focusing
on may still be a small part of that frame -- one letter of a caption, one
strand of something. So **shift-drag on the image** marks out a rectangle and
the reading comes from inside it alone. The gesture switches measuring on by
itself: drawing the box is not an ambiguous thing to be doing.

The rectangle is a fraction of the *displayed picture*, not a place on the
sensor, so it stays where it was put on screen when the camera magnifies or
pans underneath it -- which is what you want while hunting focus, where the box
marks a place to look rather than a subject to follow. It is drawn in its own
colour, dashed, so it is never mistaken for the focus box: one is where the
camera would focus, the other is where the sharpness is being read, and they
are usually not the same rectangle. Anything smaller than 16 pixels across is
grown to it, because a reading off a handful of pixels is all noise and jumps
about far too much to focus against.

The score is the mean squared difference between neighbouring pixels -- the
gradient energy contrast autofocus is built on -- with the grain taken off it,
over the square of the mean level. Both of those corrections had to be there
before the number was worth watching.

### Dividing by the level, so brightness is not what it measures

Gradients scale with brightness, so without dividing by it the light changing
would move the reading as much as focus does. Over a 3.6x range of exposure on
one picture, the reading moves by 3% (22.21, 21.93, 21.60).

### Subtracting the grain, so exposure is not what it measures

Noise is contrast, and it was most of what the reading measured. A darker live
view is a grainier one, so a stop of exposure moved the number by about a
third -- as much as a visible focus error, which made it useless: change the
aperture and the reading jumped, and a thoroughly defocused subject scored
nearly as well as a sharp one because the grain scored for it.

White noise adds a known amount to gradient energy -- four times its variance,
whatever the subject is -- so given that variance it can simply be taken off.

**Where that variance comes from is the whole thing.** The obvious way is to
estimate it from the frame: a kernel that cancels anything smooth and leaves
only what changes from one pixel to the next, which is what noise looks like.
It is a standard trick, it is wrong here, and it fails in the worst way
imaginable. At the point of best focus, the finest detail in the picture *is*
what changes from pixel to pixel. The estimator reads the subject as grain and
subtracts it, so on a finely detailed subject the sharpest frame of all reads
**zero** while a blurred one reads well above it:

| Blur | none | slight | more | a lot |
| --- | --- | --- | --- | --- |
| Reading, grain estimated within the frame | **0.0** | 14.9 | 2.6 | 0.0 |
| Reading, grain measured between frames | **227.9** | 19.0 | 5.0 | 0.0 |

A hunt handed the first row walks away from focus having been told that focus
is the worst place it has been -- which is exactly what it did, and why the
answer was not steadier readings or a better search.

Two consecutive frames of a scene holding still, on the other hand, differ by
nothing but their noise, and no amount of detail in the subject changes that.
So the variance is measured from a pair of frames, and the smallest lately
seen is the one used: when the picture is moving, frames differ by more than
noise, and the smallest measurement is the one with no movement in it. Nothing
is subtracted before anything has been measured.

The same simulated focus sweep, on a subject with detail down at the pixel:

| Blur | none | slight | moderate | thoroughly defocused |
| --- | --- | --- | --- | --- |
| Clean frames | 280 | 17.8 | 0.9 | 0.1 |
| One noisy frame, grain counted | 299 | 39 | 22 | 21 |
| One noisy frame, grain measured and removed | 278 | 17.7 | 1.0 | 0.0 |

Counting the grain, in focus reads **14x** what thoroughly defocused reads,
and everything below a slight blur is lost in the floor. Measuring it and
taking it off, a single noisy frame tracks the clean sweep almost exactly the
whole way down, and a stop of exposure moves the reading by nothing measurable
rather than by a third.

Two things follow. A picture whose detail is not clear of its own grain reads
**zero** -- not a small number, zero -- because a small number there is really
noise, and a hunt handed noise will chase it, lock onto whichever grain read
highest, and drive off to somewhere with nothing in it. Zero it reads as "no
hill here", and goes back where it came from. And integrating still earns its
place: it lowers the grain for real rather than accounting for it.

A reading costs about 3ms, and is taken once per displayed picture.

### What still moves it, and why that is honest

Changing the aperture with exposure preview on does change the picture: the
lens physically stops down, so the depth of field really does get deeper and a
defocused subject really does get sharper. Blown highlights lose their
gradients too. So the readings either side of an exposure change are of
different pictures, and comparing them is meaningless -- which is why changing
any camera setting, or the exposure preview, starts the readings again rather
than letting a stale best sit there being compared against.

### The line, not the number

The number on its own means nothing: there is no scale it belongs to and no
value that means "sharp". Only its direction matters.

A bar showing the reading as a fraction of the best seen was the obvious way
to show that, and it is useless for the half of the job that matters. While
focus is improving, every reading *is* the best one, so the bar sits at the top
throughout and only moves once you have overshot -- "it is already at the
maximum, and it stays at the maximum while I focus".

So the panel plots the readings instead, scaled to whatever range they have
lately covered rather than to zero. That is what makes a small change visible:
when the reading wanders between 410 and 430, the line spends its whole height
on that twenty. Drive focus one step and watch which way it goes; the shape of
the curve says whether the top has been passed. The numbers underneath say
where the reading sits against the best so far.

The best, and the line with it, start again whenever the crop moves, the
measured area moves, the integration settings change, or any camera setting
does -- all of them make a different measurement of a different picture.
**Reset best** does the same by hand, for when none of those has happened but
the subject has.

## Walking focus to the sharpest point

**Fine tune from here** does by motor what the reading is there to be driven
against. It is the camera's contrast autofocus, except that it works on the
measured area rather than on the body's own focus box -- which is the whole
point, since that box is 324 sensor pixels wide and the subject may be a tenth
of that. It is a separate button from **Autofocus** on purpose: that one is
the camera's own, over the camera's own box, and it is still the right thing
when the subject is large and roughly where the box is.

### Nothing counts steps

Everything below rests on one decision. Focus is driven by **making steps and
watching what the reading does**, never by remembering that the best reading
was so many steps back and driving that far.

The reason is the lens. Focus gearing has play in it, so the same number of
steps moves the optics differently depending on which way they were last
driven, and a reversal moves nothing at all until the play is taken up. A hunt
that navigates by step count therefore has to know how much play a lens has,
and drive past every target and back again to take it up -- a setting to get
right, an overshoot on every backward move, and readings that are only
comparable if the setting was right. All of it to make a step count mean
something it does not naturally mean.

Watching the reading needs none of it. The play shows up as steps where
nothing happens, and the walk keeps walking until something does. There is no
allowance to set, nothing is driven past its target, and a lens with a lot of
play costs a few extra probes rather than a wrong answer.

### The walk

What a hand does, in the panel's **minimum** increment:

- step until the reading **rises**; if it falls instead, turn round and walk
  the other way;
- keep going while it rises, remembering the best reading seen;
- when it **turns over**, walk back until the reading is as good as that best
  one again, and stop there.

It expects focus to be close already -- get roughly there by eye or with the
camera's own autofocus first. That is not much of a limitation: on a magnified
macro subject, which is what the measured area is for, focus is either close
or nowhere, and a search casting about in medium steps spends its probes at
positions where nothing in the picture could be sharp. There was such a search
here, in three increments, and it was worse than useless on exactly the
subjects this is for; the walk replaced it.

Against a simulated lens with twenty steps of depth of focus, from forty steps
out, with sixty steps of play in its gearing: **two steps** of error, two
hundred steps of lens travel, about fifteen seconds. Nearer to start with, it
is exact and quicker.

Four things about it are worth keeping.

**It comes back by reading, not by step count.** Coming back, the first steps
take up the play and the picture does not move at all; the walk simply
continues until it does. That is the whole of what replaced the backlash
machinery.

**Rising and falling are judged against the previous reading, not the best
one.** It sounds like a detail and it is the difference between working and
not: every reading after the first is below the best, so a walk that asks "is
this below the best?" answers yes to everything and gives up the moment it
turns round. That comparison is also the noisiest one available, so it takes
two falls in a row to turn the walk round -- one is as likely to be the
reading wandering as the lens going the wrong way.

**A reading that has not changed is not a reading that got worse.** Play shows
up as readings that are identical, and those are walked through; going the
wrong way shows up as readings that fall.

**On the way out the step grows while nothing is happening, and drops back the
instant it does.** Crawling through a lens's play a minimum step at a time is
a probe a second, every one of them reading exactly what the last one read, so
after a couple the step doubles, up to four times the increment. The way back
never grows: the step that finally takes up the last of the play also moves
the optics by whatever is left of it, so a long step there can carry the lens
clean past the reading it came back for. That was a real error, watched in a
trace. Speed where nothing is changing, and never where something is.

The walk is bounded -- forty probes, three thousand steps from where it began,
and a direction that says nothing for four hundred steps is abandoned --
because the body does not report the end of its travel.

**How much better counts as better** is what decides where it stops. Too low
and it chases the wander in a steady reading; too high and it stops while
there is still focus to be had. Two per cent was too high: against a modelled
focus curve it stopped a fine step short of focus on a broad peak almost every
time. One per cent finds it.

### Focus breathing, and why the area is left alone

A lens does not only change how sharp the picture is as it focuses. It changes
how big it is: the frame grows or shrinks a little with every move and
everything in it slides. The sharpness is read over a rectangle of the
*screen*, so a picture that slides underneath it is read over different
content -- and if the subject is one small thing the rectangle was drawn
snugly around, sliding it a few pixels puts half the subject outside.

The measured area used to follow the picture for that reason, by phase
correlation against the frame the hunt started from. It is gone, and the
reason it is gone is worth recording. The walk steps in the finest increment,
so the picture breathes by about a pixel between probes -- less than the
correlation can even place -- while a walk decides on one reading against the
one before it. Moving the rectangle between those two readings changes *what*
is being compared, and at that scale the following cost more in reading noise
than the drift cost in content: measured on a macro subject that breathes
hard, a walk that followed the picture landed several times further from focus
than one that left the rectangle where it was.

Following earned its place only for the search's coarse steps, which slide the
picture far enough to matter, and the search is gone too.

### Waiting for the picture to settle

This is the part that is easy to get wrong, and getting it wrong does not look
like a failure -- the hunt is simply told about the focus position it has just
left, and walks away from focus rather than towards it. It was a fixed wait of
three frames, and a fixed wait is a guess.

It now watches the picture instead. After a move, each frame is read on its
own -- not through the integrator, which is still holding frames from before
the move -- and the hunt waits for two frames that agree with each other
before starting a stack.

Two frames agreeing is not enough on its own, and the reason is the whole
trick. For the first frames after a move the picture has not started changing
yet, because live view runs behind the lens: those frames agree with each
other perfectly while showing exactly the focus position the hunt is trying to
leave. So the hunt also waits for the picture to **change**, and how many
frames that takes is the depth of the pipeline. Every move after it waits at
least that long before stillness is allowed to mean anything, and the
measurement is repeated on every move with the longest answer kept -- a hunt
that begins thoroughly defocused has nothing but noise to measure against, and
that answer must not be allowed to stand once there is a real picture to
measure with.

Two more details, both of which were bugs first:

- **A move is only a move if it is large, and only the frame it first shows
  up in counts.** A single frame can read a quarter higher than the last on
  grain alone, and taking that for the move measures the pipeline as shorter
  than it is -- which then has every reading after it taken too early. So a
  move has to change the reading by a quarter, twice in a row. The frames
  after that are still different from before the move, of course, and counting
  those as well pushed the measurement out to wherever the settle happened to
  end and made every probe wait the maximum. There is a floor of six frames
  under the whole thing, and a ceiling of twelve on what will be believed.
- **Differences are judged against the best reading of the hunt**, not only
  against the two readings being compared. Thoroughly defocused, the reading
  is nearly zero, and one grain of noise on nearly zero is a difference of
  hundreds of per cent: the picture would read as changing constantly and
  never as still.

And the stack itself must not straddle a move. It is reset once the picture
has settled -- resetting at the moment of the drive would only fill the bottom
of the new stack with frames from the old focus -- and then the hunt waits for
a **whole** stack. The single frame live view shows the instant a stack
restarts is deliberately not believed, which is why the integrator says
whether the picture it handed over was a finished stack or that one frame.

So a probe costs a drive, the settling, and a whole stack. Against a simulated
lens, a whole hunt takes about **four seconds** with integration off, **six**
integrating four frames, and **twelve** integrating sixteen -- and the trend
line draws itself as it goes, so what the hunt is doing is visible rather than
a frozen button.

Against a simulated lens, hunting the same subject from six starting points,
the error is where the optics finished against where focus really was:

| | ordinary lens | live view six frames behind | that, with slack and grain |
| --- | --- | --- | --- |
| Fixed three-frame wait | 2 | 122 | 122 |
| Watching the picture | 2 | 2 | 2 |

### Getting into the neighbourhood

A thoroughly defocused frame reads zero -- correctly, since it has no detail
above its own grain -- and zero is not a hill that can be climbed. So if the
first settled reading of a hunt is zero, the camera's own autofocus runs to get
roughly there and the hunt takes over from wherever that left it.

That decision waits for a real reading rather than being taken when the button
is pressed, and the difference matters: the meter having read nothing yet is
not the same as the picture being defocused, and treating it as such would
throw away good manual focus on the first press. If the reading is still zero
after the camera has had its go, the hunt works through its increments once
and then says there is nothing in the area to focus on, rather than driving
about hopefully.

### What stops it

Taking the focus by hand, magnifying, moving the measured area, changing the
integration, changing any camera setting, or stopping live view. All of them
mean the next reading would be of a different picture from the last one, and
comparing across that is exactly the mistake the hunt is made of. The button
says **Stop hunting** while one is running.

## How the click gestures fit together

| Gesture | Action |
| --- | --- |
| Click | Move the focus rectangle there. Nothing else |
| Double click | Autofocus, without magnifying |
| Right click | Magnify fully, or back to the whole frame if already magnified |
| Drag | Magnify onto the dragged region |
| Shift-drag | Mark out the area whose sharpness is measured |
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

## The controls panel scrolls rather than squeezing

With a camera connected the exposure form fills with eight rows, and the panel
then wants more height than the window has. A layout answers that by handing
every widget less than it asked for, and the result does not look like a panel
that is too long -- it looks like broken controls: a spin box flattened to two
thirds of its height, and the last line missing from every wrapped hint.

Two things fix it, and both are needed:

- **The panel is in a scroll area**, so the shortfall becomes scrolling instead
  of being taken out of the controls.
- **The wrapped hints resolve their own height.** Qt asks a label how tall it
  wants to be at its *hint* width, not the width the column will actually give
  it, so a wrapped label under-reports and the panel's minimum height comes out
  too small -- which is the number a scroll area goes by. `WrappedLabel`
  measures itself against the width it has been given and fixes its height
  there. Safe only because the column is a fixed width: height follows width,
  and nothing follows height, so there is no loop.

The scroll area is `NoFocus`, like every other pointer-operated control here,
for the reason in the section above.

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

### Waiting for the mirror to stop moving

Nikon's exposure delay mode -- custom setting d4 on a D750, and
`ExposureDelayMode` (`0xD06A`) over PTP -- lifts the mirror, waits one, two or
three seconds, and only then releases the shutter, so what the mirror shook has
settled before the exposure starts. On a copy stand that is the largest thing
that moves, and the wait is free sharpness.

`0xD06A` is not in the body's list of supported properties, in the same way as
the rest of Nikon's vendor properties over WPD, but `GetDevicePropDesc` answers
for it: a `UINT8` with a range form of 0 to 3 in steps of one, writable. Which
of those a body offers varies by model, so the choices in the panel are the
ones this camera reported rather than a fixed four.

The value is **not** the number of seconds. The property counts *down* to the
delay: its highest value is off, and each step below it adds a second. Timed
against the camera, release to the picture arriving, at 1/8s in live view:

| value | picture arrives | delay |
| --- | --- | --- |
| 3 | 1.5s | off |
| 2 | 2.5s | 1s |
| 1 | 3.5s | 2s |
| 0 | 4.4s | 3s |

Reading it the obvious way round is how the first version of this asked for
three seconds, wrote 3, and got no delay at all -- while "off" wrote 0 and
waited the full three. So the seconds are turned into a value by subtracting
them from the top of the range, and the top of the range is read from the body
rather than assumed. End to end through `capture()`, the same shot takes 7.00s
with no delay, 7.70s at one second and 10.39s at three.

The delay is written for each shot and the camera's own setting handed back
immediately afterwards, including when the shutter refuses to fire. Two reasons.
Nikon lists exposure delay mode among the conditions that prohibit live view
(bit 20 of `LiveViewProhibitCondition`), so a body left with it on may refuse to
start live view next time. And it is the camera's setting, not this program's,
to leave as it was found.

That cuts both ways, which is why *off* is written too rather than the property
being left alone: a body carrying a delay of its own would make "off" here mean
"whatever the camera says". For the same reason the control starts at whatever
the connected body is set to -- a rig already configured this way keeps its
delay -- and once a delay has been chosen in the panel, that is what is
remembered and used.

### Naming what lands on the computer

The camera's own names are no use for a scan. `DSC_1234.NEF` comes from a
counter that belongs to the body: it wraps at 9999 and starts again from one
whenever the card is formatted, so the order the pages were shot in is not
recoverable from the folder afterwards.

So the Capture panel offers a sequence of its own -- a prefix and the number
the next picture will get, giving `page_0084.NEF`. Three things about it:

- **It starts where you say.** A book resumed at page 84 is set to 84 before
  shooting, not renamed afterwards.
- **It can be overridden at any time.** Both boxes are live. Type over either
  between two shots and the next shot uses what was typed; nothing has to be
  switched off and on again.
- **Counting carries on from the override.** Set 84 and the shots after it are
  85, 86, and so on. The counter's only state is the next number, so there is
  no earlier sequence hiding behind an override to snap back to.

The number box is also the readout: it always shows what the next picture will
be called, and moves by itself as shots use numbers up. Where it reached is
remembered between runs, so a scan spread over two sittings is one sequence.

A number some file in the folder is already using is skipped rather than
overwritten -- whatever extension that file has, since a RAW and its JPEG are
one picture under one number. Pointing at a half-scanned folder therefore adds
to it instead of writing over page one. The extension always stays the
camera's: it is what says whether the file is a NEF or a JPEG.

Switched off, the camera's own names are kept, and a collision is decorated
(`DSC_0001_2.NEF`) rather than overwritten.

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
- **Integrate** averages several frames into each displayed image, which cancels
  the noise at the cost of frame rate: eight frames is about four frames a
  second and roughly three times less grain. The panel says what the count you
  have chosen will cost before you switch it on.
- **Measure sharpness** scores the contrast in the picture and plots the recent
  readings, so focus can be set by driving the line up rather than by eye.
  Magnify onto the subject first. The line starts again whenever the view, the
  measured area or any camera setting changes, because readings either side of
  those are of different pictures.
- **Shift-drag** on the image to measure one rectangle of it rather than the
  whole frame. That is how to focus on something smaller than the camera's
  focus box, or smaller than its strongest magnification shows.
- **Fine tune from here** then walks focus to the top of that reading by
  itself -- contrast autofocus on the area you marked out, rather than on the
  camera's focus box. Out in minimum steps until the reading turns over, then
  back until it is as good as the best it saw. Get roughly close first; it is
  for tidying up, not for finding focus from nowhere. It counts no drive steps
  at all, so it does not care how much play the lens has. Anything you do to
  the focus, the view or the exposure stops it.
- The status bar shows the live-view frame size and the rate it is arriving at
  (about 44fps, or 44/N while integrating N frames). It drops to 16fps when
  the view is magnified past 4.7x, which is the camera and not the connection.
- Shutter, aperture, ISO, exposure compensation and white balance are settable;
  the exposure mode, focus mode and drive mode are shown but are set on the
  body, so the camera reports them read-only.
- **Mirror-up delay** holds the shutter back for one to three seconds after the
  mirror lifts, so nothing is shaking when the exposure starts. It is the
  camera's own exposure delay mode, switched on for the shot and off again
  afterwards, and it starts at whatever the camera is already set to.
- Shots go to the card at full resolution and are downloaded to
  `~/Pictures/scanny` unless you turn that off.

- **Number the files myself** saves each picture as a prefix and a counter --
  `page_0084.NEF` -- instead of under the camera's name. Start the counter
  wherever the batch starts; type over it or the prefix at any time and the
  counting carries on from there. The box always shows what the next shot will
  be called, and numbers already used in the folder are skipped, never
  overwritten.

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

- filmstrip/delete file?
- focus sweep - find&visualize depth map
- wb - set to fix value

