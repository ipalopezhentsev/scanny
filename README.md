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
| `ui/hunt.py` | Autofocus, then bettering it a single step at a time, without counting steps |
| `ui/homing.py` | Back to the top of a stretch walked one way, measuring the gearing's play on the way |
| `ui/regions.py` | Each region's best, then the one focus that does best by all of them |
| `ui/report.py` | That compromise laid out: every region at its best and at the compromise |
| `ui/reportfile.py` | A report saved whole as one file, and opened again |
| `ui/film.py` | The film's shape from the regions' depths -- edges' plane and bulge -- drawn in 3D over the sensor |
| `ui/regionchart.py` | Every region's sharpness through the search for the compromise, one line each |
| `ui/activity.py` | The activity log: everything said, with the time, in a window and a file |
| `ui/navigator.py` | The whole frame, with the magnified view marked on it and draggable |
| `ui/orientation.py` | Turning, mirroring and inverting the picture on its way to the screen |
| `ui/histogram.py` | The levels in the frame the camera sends, per channel |
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

The body also turns live view off by itself after its monitor-off delay for
live view (custom setting c4 on a D750, ten minutes by default), without a
word -- frames simply stop coming. When that happens live view is started
again and the zoom and focus point put back, so a long calibration or depth
map carries on across it; see *What stops it* under focus regions.

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

### The rectangle is a place on the sensor, not a place on the screen

It used to be a fraction of the *displayed picture*, and that was wrong in a
way that only shows up the moment you magnify. A rectangle drawn round one
letter of a caption at full frame is, say, a third of the way across the
picture; magnify to 18.8x and it is still a third of the way across the strip
the camera now shows, which is a different piece of the world entirely. So the
measured area jumped to somewhere nobody chose on every change of
magnification -- and the readings either side of that jump, which is exactly
what focusing against a number is comparing, had nothing to do with each
other.

It is kept in fractions of the **whole frame** now: the same coordinates the
camera's own focus point lives in, and the same ones `ui/regions.py` keeps the
focus regions in, for the same reason. It does not move when the view
magnifies or pans; where it falls on the picture is worked out afresh for
every frame from that frame's crop rectangle. Magnify somewhere else entirely
and it is not on the picture at all, which is drawn as no box and read as no
reading -- rather than as the whole frame, which is a different question
answered with a number that looks just like the ones being compared.

The box is drawn in its own colour, dashed, so it is never mistaken for the
focus box: one is where the camera would focus, the other is where the
sharpness is being read, and they are usually not the same rectangle. Anything
smaller than 16 pixels across is grown to it, because a reading off a handful
of pixels is all noise and jumps about far too much to focus against.

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

## Fine tuning focus

**Fine tune focus** does by motor what the reading is there to be driven
against. It is the camera's contrast autofocus, except that it works on the
measured area rather than on the body's own focus box -- which is the whole
point, since that box is 324 sensor pixels wide and the subject may be a tenth
of that. It is a separate button from **Autofocus** on purpose: that one is
the camera's own, over the camera's own box, and it is still the right thing
when the subject is large and roughly where the box is.

### What it does

- **It magnifies onto the measured area first**, as far as the body will go
  and still show the whole of it. One command, and the largest single thing
  that can be done for the quality of the answer.
- **Then the camera's own autofocus**, aimed at the measured area rather than
  at wherever the focus box was left. That is a rough answer got in one go,
  and it is the *baseline*: the reading taken there is a floor the rest of the
  procedure is not allowed to end below.
- **Then it walks**, one increment at a time, keeping the best reading it has
  seen and where it saw it. While the reading rises it keeps going. When the
  reading gets worse it **turns round** -- because a reading getting worse
  means the other way, and that is the only thing it can mean.
- Once the reading has been watched to fall away on **both sides of the best**,
  that best is a peak rather than a slope and there is nothing left to explore.
- **It goes back to it** along the stretch it last walked in one direction,
  which went over the top: that stretch is a profile of the peak, and the way
  back measures the gearing's play against it and stops on its middle. Only if
  that loses the way does it walk the rest by the reading alone, turning round
  whenever it gets worse and stopping when it is back to what it was.
- If what it ends on is worse than the baseline after all, **it autofocuses
  once more** and says so.

Earlier versions of this walked one way only; grew their step while the picture
was quiet; came back to within three per cent of the best they saw; waited for
a second falling reading before turning round; searched the two directions as
separate excursions with an autofocus in between to reset; and, when a
direction stopped paying, **gave up on the spot**. Every one of those is now
different, and the reason each one changed is below. What they had in common is
the symptom: it stopped somewhere you could better by hand with the minimum
increment, having already seen the better reading during its own search.

### Nothing counts steps to drive by

Everything below rests on one decision. Focus is driven by **making steps and
watching what the reading does**, never by remembering that the best reading
was so many steps back and driving that far.

The reason is the lens. Focus gearing has play in it, so the same number of
steps moves the optics differently depending on which way they were last
driven, and a reversal moves nothing at all until the play is taken up. A
search that navigates by step count therefore has to know how much play a lens
has, and drive past every target and back again to take it up -- a setting to
get right, an overshoot on every backward move, and readings that are only
comparable if the setting was right. All of it to make a step count mean
something it does not naturally mean.

Watching the reading needs none of it. The play shows up as steps where
nothing happens, and the walk keeps walking until something does. There is no
allowance to set, nothing is driven past its target, and a lens with a lot of
play costs a few extra probes rather than a wrong answer.

Step counts are kept, and they are read for exactly one thing: **which side of
the best reading a place was on**, which is how the walk knows it has explored
both sides and can stop. A step count is not trustworthy enough to drive to;
it is quite trustworthy enough to say which side of something you were
standing.

### A direction that is not working is never a reason to stop

This was the worst of the old faults and the one that showed. Legs that walked
towards a reading had a probe budget, and running out of it **ended the whole
search** wherever it happened to be standing -- which, if the direction was the
wrong one, was as far from focus as that direction had managed to drag it. The
status line said so in as many words, and it was right: *walked 38 probes
without finding the 6 it saw again*. It had spent all 38 of them walking away.

A reading that is getting worse is information, not failure. What it says is
*go the other way*, and that is now the only thing it does. The walk turns
round; it does not stop. What ends a walk is standing on the best reading it
has seen.

There is still a limit on the turns, but it is not a budget to be spent. Two
turns is what a single-humped reading needs: one to find out the first guess at
the direction was wrong, and one to come back off the far side of the peak. The
rest are for a reading unsteady enough to send the walk the wrong way -- and a
turn only counts against the limit when the walk **got nowhere** with it, since
a walk that has found something better since it last turned is not thrashing
whatever its count says. Reaching the limit does not end the search either: it
ends the *looking*, and the walk then goes and stands on the best reading it
found.

### Looking further, instead of ending the loop

The second thing that showed, and the one that needed the search reshaped
rather than patched. The walk would go back and forth a few times inside a
small stretch and announce it was finished -- and moving focus by hand, outside
that stretch, found something better.

Two things were wrong, and they made each other worse.

**Saying nothing counted as evidence.** A direction was marked explored
whenever the walk turned round in it, for *any* reason: a wobble in the
reading, or simply having walked a while with nothing happening. Once both
directions were marked, the walk concluded it had straddled a peak and went
home. But a direction that said nothing has not been explored, it has been
glanced at. Only the reading actually **falling away** -- clearly, five per
cent below the best, while walking outward from it -- is evidence that the best
has been walked past. Nothing else ends a search now.

**And the reach was a fixed box.** Sixteen probes of the minimum increment is
about ninety drive steps on a lens with six thousand of travel, so a peak two
hundred steps from wherever the autofocus stopped was simply outside what the
walk would ever see. That reach now **doubles** every time a direction is
walked to the end of it without the reading falling away: sixteen increments
either side of the best, then thirty-two, then sixty-four. It costs the
ordinary case nothing, because a peak near the autofocus is found inside the
first reach and the growth never happens.

Two details make the growth actually reach somewhere:

- **The reach is measured from the best reading, not from where the last leg
  stopped.** Counting quiet probes from wherever you happen to be makes the
  legs cancel -- ninety steps out, ninety steps back, thirty probes spent and
  the walk is where it started. Measured from the best, the legs are a bracket
  that opens.
- **A side that has already given its answer is not walked again.** If one
  direction has been watched to fall away and the other still owes an answer,
  there is nothing to go back for: the walk carries straight on, further than
  last time, instead of covering the same ground twice.

Measured against a modelled curve, with the peak a given distance from where
the autofocus left the lens:

| peak is | before | now |
| --- | --- | --- |
| 30 steps away | found | found, 23 probes |
| 60 steps | **gave up** | found, 56 probes |
| 90 steps | **gave up** | found, 61 probes |
| 150 steps | **gave up** | found, 167 probes |
| 240 steps | gave up | says there is nothing there |

The last row is honest rather than fixed. Two hundred and forty drive steps of
picture that reads exactly the same all the way across is a blind search, and
at a six-step increment it is hundreds of probes however it is organised. The
only way to cover that ground quickly is a coarser step, and a coarser step is
how a search walks over the peak it is looking for -- so it covers what it can
and then says so. The reach is counted in *increments*, not steps, so a lens
whose minimum increment is eighteen reaches three times as far.

### A reading of nothing is a reason to look, not a reason to stop

A thoroughly defocused frame reads zero -- correctly, since it has no detail
above its own grain. That used to end the search on the spot, on the grounds
that zero is not a hill that can be climbed.

It is not a hill, but it is not a verdict either: it is what the whole
neighbourhood reads when focus is a couple of hundred steps away, which is
exactly the case the growing reach exists for. So the walk covers its reach in
both directions first, and says there is nothing in the measured area only once
that has come back empty.

### There is exactly one autofocus, at the start

There used to be a second one, to reset between searching one way and the
other. It was a mistake, and the reason is worth recording because it is not
obvious: **autofocus on the same patch does not land on the same place twice.**
It is a search of its own, with its own noise, and it stops wherever it stops.

So everything the search had learnt about which way things lay was worthless
the moment the second autofocus ran. Worse, the step-count bookkeeping that
decided which way to walk afterwards was built on the assumption that it
*would* land in the same place -- so when it did not, the walk confidently set
off in the direction away from the peak it was trying to return to.

One walk that turns itself round needs no datum to return to and no second
opinion about where it is. It also costs one autofocus and a good many fewer
probes.

### Nothing takes a longer step, either

An earlier version doubled its step while the picture was not answering, up to
four times the increment, on the reasoning that crawling through a lens's play
a minimum step at a time is a probe a second spent reading exactly what the
last one read. There is nothing to lose by hurrying through the play.

Except that the step which finally takes up the *last* of the play is also the
one that moves the optics, by however much of it was left over -- and a long
step there walks straight over the peak. What that costs is not a slower
search but a worse answer, which is the one thing this button exists not to
give. Every step it makes is the one increment it was given.

### One reading off a cliff is enough

Turning round took **two** falling readings, because one fall is as likely to
be the reading wandering as the lens going the wrong way -- which means it
always took one more step in a direction that had already got worse.

On a subject with any depth to it that second step is the expensive one. A
magnified macro scene has a couple of drive steps of depth of focus, so the
reading goes over the top at the best it will ever read, is a **fifth** of that
one step later, and a twentieth the step after. Waiting for confirmation means
standing somewhere nothing in the picture could be sharp, and then walking all
the way back through the lens's play to undo it.

So a single reading **a tenth below the best of the stretch being walked**
turns the walk round on the spot, with no confirming step. A tenth is five
times the worst grain on an integrated stack, so nothing is lost by acting on
the first one; it is also the judgement a person watching the trend line makes.

### A fall means two different things, and which one depends on the play

The two-falls rule is still there for *small* falls, and on its own it is not
enough, because a fall means different things depending on whether the optics
are moving at all.

Crossing a lens's play they are not. The reading wanders around one value and
does not trend, and at one per cent of noise against a one per cent threshold
about one pair of readings in sixteen falls twice in a row by luck alone. A
walk that turns round on that **never gets out of the play**: it turns, crosses
back, turns again, and settles in the middle of it having learnt nothing. That
was measured -- on a shallow subject with sixty steps of play and one per cent
of noise it was finishing at a ninth of the peak reading.

So a fall only counts once the reading has **sagged a twentieth below the best
of the stretch it is on**. Below that it is wander and is treated as nothing
having changed; above it, it is downhill. Note the reference: the best of the
current stretch, since the last turn. The best of the whole walk is no use,
because a walk on its way back from far out starts every stretch a long way
below that -- judging against it calls the entire journey home a collapse.

### Coming back to the peak, and not to three per cent of it

The walk back stops when the reading it left behind is back. How near counts
as back is the single number that decides how good the answer is, and it used
to be **three per cent** below the best. On a magnified macro subject three per
cent is a step short of focus -- reliably, on almost every run, and visibly
enough that anyone watching could better it by hand. It is **one per cent**
now.

The target that one per cent is measured against **is frozen when the walk
turns for home**. Letting it follow the best -- carrying on because the reading
is still rising and has already beaten what the walk set out for -- is the
obvious improvement and it is wrong: the step that beats the target is usually
the peak itself, so carrying on from there steps over it, and a step past a
peak cannot be taken back without crossing the whole of the gearing's play
again.

A walk back that goes over the top without ever matching its target stops
anyway: the best reading of a noisy walk is the luckiest of them and may not
come again, so having risen and then fallen counts as having arrived. **Only
when it is genuinely near the target**, though, and that condition is doing
real work: crossing the play the reading wanders, so a rise of two per cent
followed by a fall of three is ordinary, and without the test it reads as
"climbed to the target and went over it" while standing at a tenth of the
target -- which ends the search a dozen steps from focus. That was a real
trace, not a hypothetical.

And the test was still not enough, which a real calibration showed: three
regions of four ended **seven per cent short** of the best their own walks had
seen. The walk turns for home once the reading has sagged a twentieth or so
below its best, so it crosses the play on the way back at around nine tenths
of it -- inside the "genuinely near" band, where the grain alone rises and
falls. So the rule now also asks that the top it went over was itself within
a twentieth of the target, and above all it is no longer how the walk gets
home: that is the next section.

### Going back along the stretch that went over the top

The walk turned for home because the reading fell away on the far side of the
best, so the stretch it walked since it last turned round went over the top --
all of it in one direction, where the steps are honest: the play was taken up
at its start and stayed taken up. That stretch is a **profile of the peak**.
Its best is found as the middle of its top, from every reading on the top
rather than the luckiest one, and the way back is the same stretch in reverse.
The one number missing is how much play the way back takes up before the
optics move, and that is **measured**: the readings on the way are matched
against the profile, and the play is whatever lines them up. Readings that
have not moved yet say only that the play is not taken up, so the grain on one
of them cannot end the walk. It goes on an increment at a time, matching again
with every reading, until the best is less than half an increment away
(`ui/homing.py`, the same arithmetic the compromise goes home by, below).

If the reading there is more than a twentieth below the top of the profile,
the way was misjudged -- the play placed a step out on a top a step wide -- and
the way just walked is itself an honest stretch in one direction, so it goes
back along that; if it is still climbing it carries on over the top first.
Only after two of those, or if the profile has no top standing clear of the
grain -- or its top is well below the best the walk saw, which is the grain
making a hump out of a stretch spent crossing a long play -- is the rest walked
by the reading, as before.

Against a modelled lens with 20 or 40 steps of play and 1% grain on every
reading, taking what the optics read where the walk stopped against the best
they read anywhere on it: walking back by the reading alone stopped at 99.2%
on average and 93.6% at worst; going back along the profile, 99.85% and 98.7%.

### Turning round, and what a falling reading means

Rising and falling are judged against the **previous reading, not the best
one**. It sounds like a detail and it is the difference between working and
not: every reading after the first is below the best, so a walk that asks "is
this below the best?" answers yes to everything and turns round on the spot,
for ever.

**A reading that has not changed is not a reading that got worse.** Play shows
up as readings that are identical, and those are walked through; going the
wrong way shows up as readings that fall.

A direction that says nothing at all for sixteen probes, or four hundred steps,
is turned round too -- not given up on. A direction with nothing in it is a
direction explored, and the other one is still there to try.

**How much better counts as better** is what decides where it stops climbing.
Too low and it chases the wander in a steady reading; too high and it stops
while there is still focus to be had. Two per cent was too high: against a
modelled focus curve it stopped a fine step short of focus on a broad peak
almost every time. One per cent finds it.

### The floor, and why it is wider than everything else

The baseline is one reading, and so is the one it is compared with at the end.
A reading wanders. On a subject where it wanders a couple of per cent, a band
as narrow as the one per cent used everywhere else would call a tie a loss
about half the time, and every one of those throws away a real improvement in
order to go back to a rougher answer. So the floor is **five per cent**: what
it is for is the search having ended up somewhere genuinely worse, which is not
a thing that happens by one per cent.

### What it costs, and what it is worth

Against a modelled focus curve with a lens whose gearing has play in it, from
ten starting points either side of focus, taking the true sharpness where the
camera's own autofocus left it against the true sharpness where the walk
stopped -- both read noise-free, so what is compared is the focus position
rather than the luck of a reading. A shallow subject where the whole depth of
focus is a step or two, which is what the measured area is for, and an ordinary
one:

| | autofocus | after fine tuning | probes |
| --- | --- | --- | --- |
| shallow, no play | 1% of peak | 100% | 9 |
| shallow, 30 steps of play | 1% | 87% | 22 |
| shallow, 60 steps of play | 1% | 100% | 34 |
| shallow, 90 steps of play | 1% | 87% | 47 |
| ordinary, no play | 70% | 100% | 12 |
| ordinary, 60 steps of play | 70% | 100% | 37 |

**Those numbers do not move when noise is put on the readings** -- half a per
cent, one per cent and two per cent all give the same column, which is what the
two rules about what a fall means bought. The 87% rows are not the search
falling short: on a curve that narrow, an odd number of steps of play offsets
the grid by half an increment, and 87% is the best reading any whole number of
steps can reach. Lower the minimum increment and they go to 100 as well.

A probe costs a drive, the settling and a whole stack -- around a second -- so
with the peak near where the autofocus left it the whole thing is ten seconds
on a lens with no play and under a minute on one with a great deal. A peak
further out costs whatever the reach has to grow to: a minute or two, which is
what the doubling buys and what the **Stop** button is for. The status line
says whether it is still looking or on its way back, so a long search is legible
rather than a rising number.

That is the price of the no-shortcuts rule, and it is the right way round: this
is the button for the last hair of focus on a subject that took a while to set
up.

### Focus breathing, and why the area is left alone

A lens does not only change how sharp the picture is as it focuses. It changes
how big it is: the frame grows or shrinks a little with every move and
everything in it slides. Keeping the measured area in sensor coordinates does
not help with this, and it is worth being clear about why -- the subject moves
*across the sensor*, so a rectangle nailed to the sensor is read over
different content just as a rectangle nailed to the screen is. If the subject
is one small thing the rectangle was drawn snugly around, sliding it a few
pixels puts half the subject outside either way.

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

So a probe costs a drive, the settling, and a whole stack: around a second
with integration off, half as long again integrating four frames, and three
times as long integrating sixteen. The trend line draws itself as it goes, so
what is happening is visible rather than a frozen button.

Against a simulated lens, tuning the same subject from six starting points,
the error is where the optics finished against where focus really was:

| | ordinary lens | live view six frames behind | that, with slack and grain |
| --- | --- | --- | --- |
| Fixed three-frame wait | 2 | 122 | 122 |
| Watching the picture | 2 | 2 | 2 |

### Magnifying onto the area before anything is read

**A drive step moves the picture far more when the view is magnified.** That
is worth more here than any amount of care in the search: the focus error that is lost in the grain at
full frame is obvious at 18.8x, which is the difference between a search that
can tell one step from the next and one reading its own noise. It costs one
command, so the fine tune now does it for you rather than leaving it as
something to remember.

As far as the body will go **and still show the whole area**, rather than
simply as far as it will go. Magnifying past the rectangle would leave the
reading taken over whichever part of it stayed on screen -- a different
question from the one the rectangle was drawn to ask, and one nobody chose.
For a D750's levels that works out as:

| the area, across the frame | magnification | it shows |
| --- | --- | --- |
| a fiftieth | 18.8x | a twentieth of the frame |
| a sixteenth | 9.4x | a tenth |
| a fifth | 4.7x | a fifth |
| two fifths | 2.35x | four tenths |

With **no area marked out the view is left alone**. There is nothing chosen to
magnify onto, and going to 18.8x anyway would quietly replace "the whole
frame" with a twentieth of it -- which is a different measurement, not a
better one.

It happens once, before the first reading is taken, and never again while the
search is running: every reading a search makes has to be of the same picture
as the last, which is why changing the magnification by hand stops it. The
view is **left magnified** when it finishes, which is where you want to be to
look at what it did; Esc or 0 goes back to the whole frame.

The one cost is frame rate -- a D750 draws 44 frames a second out to 3.13x and
16 from 4.7x up -- so each probe takes a little longer at the magnifications
this picks. It buys far more than it costs.

### Aiming the camera's autofocus at the measured area

The autofocus that opens the procedure moves the camera's focus box to the
**middle of the measured area** first, and so does the one that puts focus
back if the walk ends below the floor. Focusing wherever the box was last left
would hand the walk a starting point with no relation to what it is climbing:
the reading is about the measured area and nothing else.

Moving the focus point is also what pans a magnified view on a D750, so this
has a second effect worth having -- it brings the measured area on screen.
That only works because the area is a place on the sensor: a screen-fraction
rectangle would have panned along with the picture and stayed exactly where it
was, marking out whatever the pan brought under it.

What the camera leaves behind is not required to be any good. If it reads zero
the walk still goes and looks -- see *A reading of nothing is a reason to look*
above.

### What stops it

Taking the focus by hand, magnifying, moving the measured area, changing the
integration, changing any camera setting, or stopping live view. All of them
mean the next reading would be of a different picture from the last one, and
comparing across that is exactly the mistake the whole thing is made of. The
button says **Stop** while it is running.

## One focus for several places: focus regions

Fine tuning answers where one rectangle is sharpest. A frame of film is not one
rectangle: it curls, it sags in the carrier, and the lens has a curved field,
so the corners and the middle come into focus at slightly different places --
and there is only one focus position to give them. What is wanted is not how
far apart they are but **the position that does best by all of them together**,
and an honest account of what each of them gave up for it. `ui/regions.py` is
that, `ui/report.py` shows it.

Ctrl-drag the picture to draw a region round something that has to be sharp,
up to five; ctrl-click inside one to take it away. They are remembered between
runs, since a copy stand is set up once and scanned from for hours.
**Calibrate** does two things, and the second only means anything because of
the first:

1. **Each region's best, by fine tuning on it alone.** What the Fine tune
   button does -- magnified onto the region, the camera's autofocus aimed at
   it, then walked in minimum steps until the reading has fallen away on both
   sides of its best -- once per region. The best reading the walk saw is that
   region's *peak*: the reading, and the picture of the region at that
   moment. Along with where the camera was pointed when it read it, because a
   reading is only comparable with another taken through the same crop at the
   same magnification. It does *not* walk back to stand on the peak, as the
   button does: the lens is walked somewhere else next anyway, and whatever
   the walk back lands on becomes the yardstick every share of that region is
   measured against. On a real calibration that was 7% short of the top, and
   the compromise then read three regions of four at up to 113% of "their
   best".
2. **Then one position for all of them.** At every probe the camera is panned
   to each region's view in turn -- panning moves the focus point and nothing
   else, so every region is read at the one focus position -- and each is
   read against its own peak. Those shares are combined into the one number
   the search climbs, in the way **Aim for** says, chosen before it starts:
   *the best average* of them, or *the best worst region* -- the softest one
   made as sharp as it can be, whatever that costs the sharpest.

When it finishes, focus is left on the compromise, the view is put back where
it was, the regions are coloured by how near their best they ended up (green
within 5%, amber within 15%, red further), and the **report** opens: every
region at its own best and at the compromise, side by side at the same size
with the numbers, so a person can look at what was possible and what was
chosen rather than take a percentage's word for it. The pictures are shown
the way the view is set -- turned, mirrored and inverted as the View panel has
it now -- and enlarged pixel for pixel, never smoothed. The system bell
sounds as it ends, unless it was stopped from the panel: it takes minutes, and
whoever started it has usually gone to do something else.

**Save report...** keeps the whole report as one `.focusreport` file, to be
opened again with **File > Open focus report** (Ctrl+O) -- each in a window of
its own, so an old one can be set beside the one just made. It is a zip
archive: `report.json` holds every number the report has, `pictures/` every
region at its best and at the compromise as it was cut out of the live view,
`activity.log` everything said while the calibration ran, and `readings.csv`
**every reading of every region it took**, one to a row -- which region, which
part of the calibration (`tune`, or the search's `out`, `across`, `home`,
`climb`), the step count, the reading, the seconds since the calibration began
and the seconds since the lens last moved -- for a spreadsheet, when a number
in the report wants explaining. Opened again, it is the same report, pages and
3D film included, shown the way the view is set now. **Save page as
picture...** keeps the page on show as a PNG.

While it runs, the panel keeps a clock of how long it has been going and says
what it is doing, and once the compromise is being sought it charts **every
region's sharpness, one line each**, as a share of its own best, with the
combined number dashed over them and what the search was doing shaded
behind -- so region 1 coming to its best and going over it while region 2 is
still climbing can be watched, not guessed from the one number. The report
says what it all took: the time, each region's fine-tune probes, the
compromise's, and how many focus moves and drive steps that came to.

The report has four pages. **Regions** is the pictures. **Film shape** is what
levelling the film needs: **how far apart in focus the regions are**, in drive
steps, nearest first with the doubt on each -- `1 nearest,  3 +42 ±2,  2 +95
±3` -- and the film itself, drawn in three dimensions over the sensor: each
region's rectangle on the sensor, a line up from it to where that region is
on the film, and the film bent through those points. Drag it round, scroll to
come closer; the height is exaggerated, since the depths are drive steps and
have no length in common with the frame. **The search** is the chart of every
region, full size. **Activity log** is everything said while it ran.

### The film is foil, held at its edges

A frame of film in a holder is not a plane. The holder grips it along its four
sides, which lie more or less on a plane -- one that may lean against the
sensor, which is what levelling corrects -- and inside them the film is free
to bow towards the lens or away from it. So that is the shape fitted to the
depths: **a plane for the edges, and a bulge that is nothing at the edges**,
made of the shapes a sheet clamped along its sides takes, and the gentlest of
those that passes through every measured depth -- each weighted by how much
bending it costs, so nothing is invented between the regions that they do not
ask for. Three regions are always a plane, so it takes a fourth -- one in the
middle says most -- to see the bow at all.

That separation is the point of it. A plane pulled through all the regions,
the obvious fit, leans the wrong amount as soon as one of them is on a bow:
the holder tips the edges, not the middle. So what the report gives as the
lean to correct is the **edges'** -- how much further the right edge focuses
than the left and the bottom than the top, in the picture's directions as it
is shown, with the doubt on each carried through from the depths -- and,
separately, how far the film bows between them, which no levelling removes.
The edges are taken to be the picture's: true when the film frame fills the
camera's frame, as a copy stand is set up to.

### Depth, when it is asked for, and why it can be trusted

Depths cost walking the compromise does not need, so they are asked for:
**Measure depths for levelling**, off out of the box. Levelling is done once,
when the stand is set up, and frames are calibrated many times after. With it
on, the walks go on past the turn-back line until every region has fallen a
quarter below its best on both sides of its peak -- far enough to place the
peak to a step or two, where a sixth left the worst five steps out -- and no
further than twenty-five increments for that. With it off, the report's Film
shape page says how to get them rather than showing a few that happened to
be measured.

The search's walk across (below) crosses every region's peak in one
direction. Within one direction the steps are honest -- the play in the
gearing was taken up at the start of the walk and stayed taken up -- so where
on it each region peaked is its depth against the others. The play moves every
position on that walk by the same amount, which is why only the differences
are reported: they are the answer, and the positions themselves mean nothing.
This is the same reasoning the old point scan was built on, without its
parking or its bracket, because the walk across was happening anyway.

Two things were needed to make the number worth levelling by. The walk has to
take in the **top** of each region's curve with some of both its sides, not
just its peak: the depth is the middle of that top, drawn at the same height
on both sides so that it stays symmetrical, and a curve cut off on one side
has a middle pulled towards the cut. That is what the quarter's fall on each
side is for, whether or not the compromise needs that region -- a region a
little outside the others is still one the film is levelled by. And each
depth carries its doubt, worked out from the grain on the readings
themselves, and the lean carries that doubt through the fit, so a lean
smaller than it can tell is called level. Against made-up scenes of regions
on one frame, with up to sixty steps of play and 2% grain on every reading,
depths came back within two steps of the truth, most within a fraction of
one, for about a dozen more probes than the compromise alone.

**Which way is further is the drive's word, not a measurement.** The focus
drive's two directions are named nearer and further after Nikon's usual
convention, and `camera/nikon.py` could not confirm from live view which way
a lens actually turns. Before levelling by the report, drive focus a few
steps further with `>` and check that the edge it says is further is the one
that sharpens. The steps are the lens's, too: how much film height one of
them is depends on the lens and the magnification, and a shim of known
thickness under one edge, calibrated again, is the way to find out.

### Shares of each region's own best, not the readings

A reading has no units: its size is set by how much detail is in the box and
how bright it is. Averaging raw readings would hand the compromise to
whichever region has the most texture in it, and a faint region would count
for nothing. As shares of their own peaks, every region has the same say.

The two ways of combining them part company when regions are further apart
in focus than each is deep. The best average can then be one region's own
peak, with the others well short of theirs; the best worst region never
sacrifices one, and pays for that in the average. Both only ever go up when a
single share does, so the search is the same for either.

**A region's peak can go up while the search runs.** Its peak is the best it
has read *anywhere* -- its fine tune, or any probe of the search -- and the walk
across crosses every region's peak, read the same way every later reading
is. A region the search reads higher than its fine tune did has that for its
peak from then on, picture and all, and the log says so. It matters beyond
the report showing more than 100%: a peak read low makes that region's share
too big, so the average leans towards it and the worst region protected may
not be the one that needs it. Everything the search judges by is worked out
again from the readings with the new peak; only the way home keeps the peaks
it set out with, since it matches readings against a profile and a profile has
to hold still. The report measures everything against the peaks as they
finally stood, and where one was raised it says by how much, next to what the
fine tune alone found -- a fine tune that keeps reading a region lower than the
search is worth looking into, and `readings.csv` is where to look.

### Why the compromise is not a fine tune on the average

That was the first version, and two things about the average that are not
true of one region's reading each cost it something real.

**It can have more than one hill.** A region's reading against focus has one
peak. The average of several has one per region wherever the regions are
further apart than each is deep, and a climb finds the hill it starts on --
which, started as it has to be on the last region's own peak, is that
region's, when the hill in the middle serves all of them better. Against a
simulated rig with three regions sixty steps apart, the climb stood on 40%
where 44% was there to be had.

**Its top is flat.** Several peaks side by side add up to a broad top whose
readings are within a per cent of each other over many steps. A fine tune
walks home until the reading is within a per cent of the best it saw, which
on one region's steep peak is the top -- and on the average's flat top is its
edge. Two regions eighty steps apart came back as 85% and 42% where the
middle had both at 64%.

So the search is three legs, and none of them counts a step across a reversal:

- **Out**, one way until it is plainly the wrong way: the number being
  climbed has fallen below **Turn back below** -- 80% out of the box -- of the
  best it reached on this walk, and it turns at the first reading under that
  line. The one thing it waits for is a region *visibly climbing* towards its
  own best, and then only ten increments: that is how a hill just beyond a
  valley is still found. It used to walk on while anything ahead could in
  principle beat the best so far, and then until every region had fallen well
  below its own best for the sake of the depths -- which is how a walk told
  to turn back at 80% was seen going on to where the sharpest region was at
  38% of its best, waiting for three broad ones around it to fall. On one
  frame of film turning back at 80% loses nothing and saves about a third of
  the probes; set it lower only when regions are so far apart in focus that
  the best compromise lies beyond a valley where they are all soft. Regions
  so far apart that each reads nothing where another is sharp are beyond any
  walk: nothing on the way says there is anything further on.
- **Across**, back the other way until the same holds. That one leg crosses
  every region's peak in a single direction, so its steps are honest -- the
  play was taken up at its start -- and it is a true profile of the average
  against focus. Its best is the best of all of it, found as the middle of
  its top by a weighted middle of the readings on it -- with the top drawn
  near the top: a line half way down took in enough of a lopsided top to pull
  its middle several steps off the best.
- **Home**, back again. The way home retraces the profile, but how much play
  was taken up first is not known, and it is the one number the walk needs.
  So it is measured: every reading on the way home is matched against the
  profile, and the play is whatever lines them up -- the arithmetic the fine
  tune now goes home by as well (`ui/homing.py`). The walk goes on an
  increment at a time, matching again with every reading, until the best is
  less than half an increment away. If the reading there does not agree --
  the top was one increment wide, or the walk passed the best before it could
  tell -- the walk home is itself an honest profile, and it goes home again
  from that; only after that does a fine tune take over and climb the hill it
  is next to.

Against made-up scenes of two to five regions at random depths, with up to
sixty steps of play and 2% grain on every reading, it ends on average 0.2%
short of the best average any focus position gives, 1.3% at the 99th
percentile. The worst region is harder, because its top is a point where two
regions' curves cross: an increment off it, which the play can make
unavoidable, costs more there, and it ends on average 0.9% short. It takes
about sixty probes, each a pan and a stack per region -- a few minutes for
three regions magnified.

### What stops it

Anything that would make the readings after it incomparable with the ones
before -- focus taken by hand, the view magnified, the exposure or the
integration changed, the regions or the measured area moved -- stops it and
puts the view back. What it had found by then is kept: the peaks of the
regions it reached still colour them and still fill the report.

Live view going off does not stop it. A D750 ends live view after its own
monitor-off delay for it -- custom setting **c4 > Live view**, ten minutes out
of the box -- whoever is driving it, and a calibration can run longer than
that. It shows two ways: frames stop coming, and the body refuses
to move the focus point or drive focus, "not in live view" -- the second is
what a calibration meets first, between two probes, and missing it is what
used to end calibrations at the ten-minute mark. Either way live view is
started again, patiently -- a body still putting its mirror down refuses the
first attempt -- the zoom and the focus point are put back through the busy
answers that follow a restart, and the picture is let settle before anything
is read. Focus never moved, so whatever was running carries on: a probe is
taken again, and a move the search had asked for when live view went is made
first. Up to three times in two minutes: a body that ends it faster than that
is refusing for some other reason, and live view is shut down as before.
Setting c4 to its longest on the body saves the restarts altogether.

Everything along the way is in **View > Activity log** (Ctrl+L): every
setting it started under, one to a line -- its own choices and regions, every
exposure setting read from the body, live view's frame, zoom, integration and
grain, all the focus increments, the view, capture and naming -- every region
tuned, every probe of the compromise with every region's share, every
attempt at getting live view back with the camera's own reason for refusing,
and why anything stopped. The same lines are written to `activity.log` in the
program's data folder, so they are there after the window or the program is
closed.

## How the click gestures fit together

| Gesture | Action |
| --- | --- |
| Click | Move the focus rectangle there. Nothing else |
| Double click | Autofocus, without magnifying |
| Right click | Magnify fully, or back to the whole frame if already magnified |
| Drag | Magnify onto the dragged region |
| Shift-drag | Mark out the area whose sharpness is measured |
| Ctrl-click | Put down a point to be measured, or take away the one there |
| Arrow keys | Pan the magnified view |
| Scroll | Step magnification |
| Enter | Autofocus |
| `[` `]` | Manual focus, minimum increment, nearer / further |
| `,` `.` | Manual focus, fine increment |
| `<` `>` | Manual focus, coarse increment |
| Esc or 0 | Back to the whole frame, whatever the current state |
| Ctrl+I | Invert the colours |
| Ctrl+R / Ctrl+Shift+R | Turn the view right / left |
| Ctrl+H / Ctrl+Shift+H | Mirror the view left-right / top-bottom |

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

## Turning, mirroring and inverting the view

A copy stand is built the way the room allows, not the way the sensor is wired.
The body ends up on its side because that is how the frame fits a strip of
film; the film ends up emulsion-towards the lens because that is the way round
it lies flat. Negative film adds a third mismatch that is not geometry at all —
what reaches the screen is the complement of what was in front of the lens, and
judging a face, a sky or a skin tone by its negative is guesswork.

So the View panel turns the picture by right angles, mirrors it in either axis,
and inverts its colours. All of it is a **display transform** and nothing else:

| Follows the view | Does not |
| --- | --- |
| The image, and every overlay on it — focus box, measured area, regions | The focus coordinates sent to the camera |
| The navigator map and its crop rectangle | The histogram |
| The pictures in the focus report | The sharpness reading, and what the calibration measured |
| | The pictures the camera saves, and the pictures in a saved report |

The **histogram** is the deliberate one. It is read to judge exposure and
clipping in what the camera is actually recording, and an inverted copy would
report a blown highlight as a blocked shadow — the one thing the readout exists
to catch, said backwards.

Inverting is a straight complement and nothing more. Colour negative carries an
orange mask, so an inverted frame comes out cold until the white balance is set
for the light coming through the film — which is what the custom white balance
input is for.

### The widget is where the transform stops

Everything the panes hand back is in the frame's own coordinates, turned back
before it leaves. That boundary is the point: a click on a picture being shown
upside down still asks the camera to focus on the thing under the pointer, and
nothing behind the screen — not the worker, not the sharpness meter, not the
calibration — has to know the view has been touched at all.

The price is that the widget drawing a turned picture has to turn the overlays
*with* it. They arrive in the frame's coordinates: the focus box comes off the
camera's own header, and a box drawn without the turn applied would sit in the
wrong corner of a picture that otherwise looks perfectly right. One other thing
falls out of this — a right angle swaps which corner of a dragged rectangle is
the top left one, so a rectangle mapped by subtracting its corners comes back
with a negative width, which draws as nothing and contains nothing. Both
rectangles are mapped corner-wise and put back the right way round.

The window's opening size follows the transform too. It opens at the shape of a
live-view frame so the black strips start at nothing, and a quarter turn swaps
the frame's two sides, so a sideways rig opens a portrait window.

### Eight arrangements, not three switches

Right angles and mirrors generate eight arrangements in all, and three
independent flags cannot name them: mirror left-to-right and then top-to-bottom
and you have not got a doubly mirrored picture, you have got a picture turned
through 180°. The state kept is therefore the eight arrangements themselves,
written canonically as *a mirror across the vertical middle, then some number
of quarter turns*. Every arrangement has exactly one such spelling, so two
routes to the same picture cannot leave the readout and the controls disagreeing
about what is on the screen.

That is also why turning and mirroring are buttons rather than tick boxes: they
compose. Each one does the plain thing to what is on the screen at the moment it
is pressed — *rotate right* turns whatever is showing a quarter turn clockwise,
*mirror left-right* mirrors whatever is showing about the screen's vertical
middle — and that has to hold when the picture is **already** turned. Mirroring
the screen is not the same as mirroring the frame once a turn is in the way, so
a mirror rewrites the turns as well as the flag, out of `H·Rᵏ = R⁻ᵏ·H`. Get that
wrong and the button still works from the camera's own way up, which is the one
arrangement in which the mistake does not show.

The arrangement is remembered between sessions, because a copy stand is not
rebuilt between them.

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
- **Invert colours** shows the complement of what the camera sends, so a
  negative can be judged as the picture it is going to be rather than as its
  opposite. Set the white balance for the light coming through the film as
  well, or the result stays cold: colour negative carries an orange mask, and
  inverting alone does nothing about it.
- **Rotate** and **mirror** turn the view by right angles and flip it in either
  axis, for a body mounted on its side or film lying emulsion-up. Each button
  does the plain thing to what is on the screen, so they can be pressed in any
  order until the picture looks right. Nothing behind the screen moves with
  them: clicks still aim at what is under the pointer, the readings are
  unchanged, and the saved pictures come off the card as the camera took them.
  What is set is remembered for next time.
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
  focus box, or smaller than its strongest magnification shows. The rectangle
  marks a place on the sensor, so it stays on the same part of the subject
  when you magnify or pan.
- **Fine tune focus** then drives focus to the top of that reading by itself
  -- contrast autofocus on the area you marked out, rather than on the
  camera's focus box. It magnifies onto that area as far as the body will go
  and still show it, autofocuses there, and then walks in minimum steps and
  nothing coarser -- turning round whenever the reading gets worse, since that
  is the only thing a reading getting worse can mean -- until it is standing
  on the best reading there is. It will not leave focus worse than the
  camera's own autofocus managed. It counts no drive steps to drive by, so it
  does not care how much play the lens has; it pays for that in probes
  instead, ten seconds on a tight lens and under a minute on a loose one. The
  view is left magnified afterwards; Esc or 0 goes back. Anything you do to
  the focus, the view or the exposure stops it.
- **Ctrl-drag** the picture to draw a focus region round something that has
  to be sharp, up to five, and ctrl-click inside one to take it away.
  **Calibrate** fine tunes each region on its own and keeps what it read and
  looked like at its best, then walks focus to the one position where the
  regions are, on average, nearest their own best -- panning to every region
  at each step. Focus is left there, the regions are coloured by what the
  compromise cost them, and the **Report** shows each region at its best and
  at the compromise side by side, with how far apart in focus they are and
  how the frame leans. **Aim for** chooses, before it starts, between the
  best average and the best worst region, and **Turn back below** how far a
  walk goes the wrong way before it turns round. It takes a few minutes;
  anything you do to the focus or the view stops it, live view going off does
  not, and **View > Activity log** says what it did. The system bell sounds
  when it ends by itself, finished or stopped by trouble, but not when you
  press Stop. **Save report...** in the report keeps the whole of it as one
  `.focusreport` file, and **File > Open focus report** (Ctrl+O) opens one
  again in a window of its own.
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
- 3d looks buggy
- sometimes calibrated compromise shows actually better sharpness than initially found "best" fine tuned.