"""A high-level interface to a Nikon DSLR over PTP.

Written against a D750; the operations used are common to Nikon's DSLR vendor
extension, so other bodies of the same generation should work with little or no
change.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..ptp.codes import Event, Op, Prop, Response
from ..ptp.parser import Reader
from ..ptp.session import PtpSession
from ..wpd.device import DeviceInfo as WpdDeviceInfo
from ..wpd.device import MtpError, WpdCommandError, WpdMtpTransport, enumerate_devices
from . import values as v

__all__ = ["NikonCamera", "LiveViewFrame", "Setting", "CameraError", "NIKON_VID"]

NIKON_VID = 0x04B0


class CameraError(RuntimeError):
    """An operation the camera refused."""


@dataclass(frozen=True)
class LiveViewFrame:
    """One live-view frame plus the focus metadata Nikon packs ahead of it.

    The header is big-endian and, on a D750, 384 bytes long. Its geometry was
    mapped against the camera itself; the fields that matter are:

    ===========  ==========================================================
    offset 8/10  size of the JPEG image
    offset 12/14 the autofocus coordinate space -- the full sensor frame,
                 constant regardless of magnification
    offset 16/18 size of the region currently displayed, in AF-space units;
                 this shrinks as the live view is magnified
    offset 20/22 centre of that displayed region, in AF space. When
                 magnified, the camera pans this to follow the focus point
    offset 24/26 size of the focus box, in AF space
    offset 28/30 centre of the focus box, in AF space
    offset 52    roll, in whole degrees, wrapping through 359 for negative
    offset 56    pitch, same encoding, 0xFFFF when out of the sensor's range
    ===========  ==========================================================

    The two level fields were found by recording the header while the camera
    was tilted: offset 52 moved as the body was rolled left and right, offset
    56 as it was pitched nose up and down. Offsets 54 and 58 change every
    frame with no relation to either, so they are left alone.

    Expressing clicks and overlays through the crop rectangle means one
    mapping works at every zoom level.
    """

    jpeg: bytes
    #: Size of the JPEG image itself.
    width: int
    height: int
    #: The autofocus coordinate space: the full sensor frame.
    image_width: int
    image_height: int
    #: Size and centre of the region on show, in AF-space units.
    crop_width: int
    crop_height: int
    crop_center_x: int
    crop_center_y: int
    #: Size of the focus box, in AF space.
    af_width: int
    af_height: int
    #: Centre of the focus box, in AF space.
    af_x: int
    af_y: int
    #: The body's level sensor, in degrees, or None where it has no reading.
    #: Positive roll is clockwise; positive pitch is nose up.
    roll: "int | None" = None
    pitch: "int | None" = None

    @property
    def is_level(self) -> bool:
        """Whether the camera is within a degree of level on both axes."""
        return (
            self.roll is not None
            and self.pitch is not None
            and abs(self.roll) <= 1
            and abs(self.pitch) <= 1
        )

    @staticmethod
    def _angle(raw: int) -> "int | None":
        """Nikon reports 0-359; fold that to -180..180, and 0xFFFF to None."""
        if raw == 0xFFFF:
            return None
        wrapped = raw % 360
        return wrapped - 360 if wrapped > 180 else wrapped

    @property
    def magnification(self) -> float:
        """How much the displayed region is magnified relative to the frame."""
        if not self.crop_width:
            return 1.0
        return self.image_width / self.crop_width

    def to_af_coords(self, nx: float, ny: float) -> "tuple[int, int]":
        """Map a click at fractional position (nx, ny) of the displayed image
        to a focus-point coordinate, clamped to where the box actually fits."""
        crop_w = self.crop_width or self.image_width
        crop_h = self.crop_height or self.image_height
        x = self.crop_center_x + (nx - 0.5) * crop_w
        y = self.crop_center_y + (ny - 0.5) * crop_h
        half_w, half_h = self.af_width // 2, self.af_height // 2
        x = min(max(x, half_w), max(self.image_width - half_w, half_w))
        y = min(max(y, half_h), max(self.image_height - half_h, half_h))
        return int(round(x)), int(round(y))

    @property
    def af_box_normalised(self) -> "tuple[float, float, float, float]":
        """The focus box as (x, y, w, h) fractions of the displayed image."""
        crop_w = self.crop_width or self.image_width
        crop_h = self.crop_height or self.image_height
        if not crop_w or not crop_h:
            return (0.0, 0.0, 0.0, 0.0)
        cx = (self.af_x - self.crop_center_x) / crop_w + 0.5
        cy = (self.af_y - self.crop_center_y) / crop_h + 0.5
        w = self.af_width / crop_w
        h = self.af_height / crop_h
        return (cx - w / 2, cy - h / 2, w, h)

    @classmethod
    def parse(cls, data: bytes) -> "LiveViewFrame":
        start = data.find(b"\xff\xd8\xff")
        if start < 0:
            raise CameraError("live view response contained no JPEG image")
        end = data.rfind(b"\xff\xd9")
        jpeg = data[start : end + 2] if end > start else data[start:]

        def u16(offset: int) -> int:
            if offset + 2 > start:
                return 0
            return struct.unpack_from(">H", data, offset)[0]

        return cls(
            jpeg=jpeg,
            width=u16(8),
            height=u16(10),
            image_width=u16(12),
            image_height=u16(14),
            crop_width=u16(16),
            crop_height=u16(18),
            crop_center_x=u16(20),
            crop_center_y=u16(22),
            af_width=u16(24),
            af_height=u16(26),
            af_x=u16(28),
            af_y=u16(30),
            roll=cls._angle(u16(52)),
            pitch=cls._angle(u16(56)),
        )


@dataclass(frozen=True)
class Setting:
    """A camera property presented for display: current value plus its choices."""

    code: int
    name: str
    value: Any
    label: str
    writable: bool
    choices: "tuple[tuple[Any, str], ...]"

    @property
    def choice_labels(self) -> "list[str]":
        return [label for _, label in self.choices]

    def value_for_label(self, label: str) -> Any:
        for value, text in self.choices:
            if text == label:
                return value
        raise KeyError(label)


#: The properties the UI offers, in display order.
_SETTINGS: "tuple[tuple[str, int, Callable[[Any], str]], ...]" = (
    ("Shutter", Prop.EXPOSURE_TIME, v.format_shutter),
    ("Aperture", Prop.F_NUMBER, v.format_aperture),
    ("ISO", Prop.EXPOSURE_INDEX, v.format_iso),
    ("Exp. comp.", Prop.EXPOSURE_BIAS_COMPENSATION, v.format_exposure_bias),
    ("White balance", Prop.WHITE_BALANCE, v.format_white_balance),
    ("Mode", Prop.EXPOSURE_PROGRAM_MODE, v.format_program_mode),
    ("Focus mode", Prop.FOCUS_MODE, v.format_focus_mode),
    ("Drive", Prop.STILL_CAPTURE_MODE, v.format_capture_mode),
)


class NikonCamera:
    """A connected Nikon body.

    Instances are safe to drive from more than one thread: every PTP
    transaction is serialised by the underlying :class:`PtpSession`.
    """

    #: Zoom levels NIKON_LiveViewImageZoomRatio actually accepts. The property
    #: advertises a 0-7 range, but a D750 rejects level 1 as an invalid value:
    #: it jumps from full frame straight to 2.35x. Measured magnifications are
    #: 1.0, 2.35, 3.13, 4.7, 6.27, 9.4 and 18.8.
    ZOOM_LEVELS = (0, 2, 3, 4, 5, 6, 7)

    def __init__(self, transport: WpdMtpTransport) -> None:
        self._transport = transport
        self.session = PtpSession(transport)
        self._live_view = False
        self._exposure_preview = True
        self._lock = threading.RLock()

    # -- discovery and lifecycle ------------------------------------------

    @staticmethod
    def discover() -> "list[WpdDeviceInfo]":
        """Every attached Nikon camera Windows can see."""
        return [
            d
            for d in enumerate_devices()
            if d.vid_pid is not None and d.vid_pid[0] == NIKON_VID
        ]

    @classmethod
    def open(cls, device: "WpdDeviceInfo | None" = None) -> "NikonCamera":
        if device is None:
            found = cls.discover()
            if not found:
                raise CameraError(
                    "no Nikon camera found -- check it is switched on, connected "
                    "by USB, and not in Mass Storage mode"
                )
            device = found[0]
        camera = cls(WpdMtpTransport(device.pnp_id).open())
        camera._on_open()
        return camera

    def _on_open(self) -> None:
        # Leaving live view running from a previous process wedges the next
        # StartLiveView, so always begin from a known state.
        self.session.try_execute(Op.NIKON_END_LIVE_VIEW)

    def close(self) -> None:
        if self._live_view:
            self.stop_live_view()
        self._transport.close()

    def __enter__(self) -> "NikonCamera":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def model(self) -> str:
        info = self.session.device_info()
        return f"{info.manufacturer.strip()} {info.model.strip()}".strip()

    @property
    def firmware(self) -> str:
        return self.session.device_info().device_version

    @property
    def serial_number(self) -> str:
        return self.session.device_info().serial_number.lstrip("0") or "0"

    # -- readiness ---------------------------------------------------------

    def wait_ready(self, timeout: float = 8.0) -> bool:
        """Block until the camera finishes whatever it is doing."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.session.execute(Op.NIKON_DEVICE_READY)
                return True
            except MtpError as exc:
                if exc.response in (Response.DEVICE_BUSY, Response.NIKON_INVALID_STATUS):
                    time.sleep(0.02)
                    continue
                return False
            except WpdCommandError:
                return False
        return False

    # -- live view ---------------------------------------------------------

    @property
    def live_view_active(self) -> bool:
        return self._live_view

    def start_live_view(self, timeout: float = 8.0) -> None:
        """Flip the mirror up and begin streaming frames."""
        with self._lock:
            if self._live_view:
                return
            prohibit = self.live_view_prohibit_condition()
            if prohibit:
                raise CameraError(
                    f"camera will not enter live view: {v.describe_lv_prohibit(prohibit)}"
                )
            # The body is often still busy settling from a previous command
            # (an EndLiveView, or a shot writing to the card), and answers
            # StartLiveView with DEVICE_BUSY until it is done.
            self.wait_ready(timeout)
            deadline = time.monotonic() + timeout
            while True:
                try:
                    self.session.execute(Op.NIKON_START_LIVE_VIEW)
                    break
                except MtpError as exc:
                    busy = exc.response in (
                        Response.DEVICE_BUSY,
                        Response.NIKON_INVALID_STATUS,
                    )
                    if not busy or time.monotonic() >= deadline:
                        raise CameraError(self._live_view_failure(exc)) from exc
                    time.sleep(0.2)
            self.wait_ready(timeout)
            # The first frames are not available immediately after the mirror
            # lifts; poll until one arrives rather than guessing a delay.
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    self.session.read(Op.NIKON_GET_LIVE_VIEW_IMAGE)
                    break
                except (MtpError, WpdCommandError):
                    time.sleep(0.05)
            self._live_view = True
            self._request_largest_frame_size()
            self._apply_exposure_preview()

    @staticmethod
    def _live_view_failure(exc: MtpError) -> str:
        if exc.response == Response.DEVICE_BUSY:
            # Nothing in software clears this: not TerminateCapture, not
            # AfDriveCancel, not reopening the device. The body has to be
            # power cycled.
            return (
                "The camera reports it is busy and will not start live view. "
                "This happens when a previous capture or live-view session was "
                "left unfinished, and it cannot be cleared over USB -- switch "
                "the camera off and on again."
            )
        return f"Could not start live view (0x{exc.response:04X})."

    def stop_live_view(self) -> None:
        with self._lock:
            self._live_view = False
            self.session.try_execute(Op.NIKON_END_LIVE_VIEW)

    def live_view_frame(self) -> LiveViewFrame:
        """Grab the current live-view frame."""
        data, _ = self.session.read(Op.NIKON_GET_LIVE_VIEW_IMAGE)
        return LiveViewFrame.parse(data)

    # -- exposure preview --------------------------------------------------
    #
    # With preview off the camera normalises live-view brightness for a clear
    # view, so changing aperture, shutter or ISO does nothing visible. With it
    # on the lens actually stops down and the image shows the exposure that
    # would be taken -- measured on a D750, mean frame brightness falls from
    # 231 at f/4 to 79 at f/22, and depth of field changes with it.
    #
    # Nikon encodes this inverted: 0 means preview on, 1 means off. The
    # property is only writable while live view is running, so the desired
    # state is remembered and applied each time live view starts.

    @property
    def exposure_preview(self) -> bool:
        """Whether live view shows the exposure that would actually be taken."""
        return self._exposure_preview

    def set_exposure_preview(self, enabled: bool) -> None:
        self._exposure_preview = bool(enabled)
        if self._live_view:
            self._apply_exposure_preview()

    def _apply_exposure_preview(self) -> None:
        try:
            self.session.set_prop(
                Prop.NIKON_LIVE_VIEW_EXPOSURE_PREVIEW, 0 if self._exposure_preview else 1
            )
        except (MtpError, WpdCommandError):
            # Not every body offers it; live view is still perfectly usable.
            pass

    def _request_largest_frame_size(self) -> None:
        """Ask for the bigger of the two live-view frame sizes the body offers."""
        try:
            self.session.set_prop(Prop.NIKON_LIVE_VIEW_IMAGE_SIZE, 2)
        except (MtpError, WpdCommandError):
            pass

    def live_view_image_size(self) -> "tuple[int, int]":
        """The live-view frame size the camera is set to send.

        A D750 offers only two: 320x180 and 640x360. There is no higher
        resolution available over PTP, so the full-frame view is inherently
        soft; magnifying does not resample, it re-renders a smaller part of
        the sensor into the same 640 pixels.
        """
        try:
            return (320, 180) if int(self.session.get_prop(Prop.NIKON_LIVE_VIEW_IMAGE_SIZE)) == 1 else (640, 360)
        except (MtpError, WpdCommandError, ValueError):
            return (640, 360)

    def live_view_prohibit_condition(self) -> int:
        """Nikon's bitmask of reasons live view is unavailable; 0 means ready."""
        try:
            return int(self.session.get_prop(Prop.NIKON_LIVE_VIEW_PROHIBIT_CONDITION))
        except (MtpError, WpdCommandError, ValueError):
            return 0

    # -- zoom --------------------------------------------------------------

    def zoom_level(self) -> int:
        try:
            return int(self.session.get_prop(Prop.NIKON_LIVE_VIEW_IMAGE_ZOOM_RATIO))
        except (MtpError, WpdCommandError, ValueError):
            return 0

    def set_zoom_level(self, level: int) -> None:
        """Set the camera's optical live-view magnification (0 = fit, 7 = maximum)."""
        # Snap to a level the body will accept rather than let it refuse.
        level = min(self.ZOOM_LEVELS, key=lambda valid: abs(valid - int(level)))
        # Immediately after a zoom or a live-view restart the camera rejects a
        # new level as an invalid value until it has finished reconfiguring.
        deadline = time.monotonic() + 3.0
        while True:
            try:
                self.session.set_prop(Prop.NIKON_LIVE_VIEW_IMAGE_ZOOM_RATIO, level)
                return
            except MtpError as exc:
                retryable = exc.response in (
                    Response.DEVICE_BUSY,
                    Response.INVALID_DEVICE_PROP_VALUE,
                    Response.NIKON_INVALID_STATUS,
                )
                if not retryable or time.monotonic() >= deadline:
                    raise CameraError(
                        f"could not set zoom (0x{exc.response:04X})"
                    ) from exc
                time.sleep(0.15)

    # -- autofocus ---------------------------------------------------------

    def set_af_area(self, x: int, y: int) -> None:
        """Move the focus point. Coordinates are in the frame's AF space."""
        try:
            self.session.execute(Op.NIKON_CHANGE_AF_AREA, (int(x), int(y)))
        except MtpError as exc:
            if exc.response == Response.NIKON_NOT_LIVE_VIEW:
                raise CameraError("focus point can only be moved in live view") from exc
            raise CameraError(
                f"could not move focus point (0x{exc.response:04X})"
            ) from exc

    def autofocus(self, timeout: float = 8.0) -> bool:
        """Drive autofocus. Returns False if the camera could not find focus."""
        try:
            self.session.execute(Op.NIKON_AF_DRIVE)
        except MtpError as exc:
            if exc.response == Response.NIKON_OUT_OF_FOCUS:
                return False
            raise CameraError(
                f"autofocus failed (0x{exc.response:04X})"
            ) from exc
        return self.wait_ready(timeout)

    def focus_at(self, x: int, y: int, timeout: float = 8.0) -> bool:
        """Move the focus point and immediately focus there."""
        self.set_af_area(x, y)
        return self.autofocus(timeout)

    def cancel_autofocus(self) -> None:
        self.session.try_execute(Op.NIKON_AF_DRIVE_CANCEL)

    #: The manual-focus increments offered, from finest to coarsest.
    FOCUS_INCREMENTS = ("minimum", "fine", "medium", "coarse")

    #: Default step count for each increment. These are only defaults: how far
    #: a step actually moves focus depends on the lens and the subject
    #: distance, so the interface lets them be set per taste and remembers it.
    #:
    #: Measured on a D750: about 6000 steps covers the whole range of travel,
    #: 800 clearly shifts focus, 400 is slight, and under about 100 is lost in
    #: frame-to-frame noise at full frame -- though a step is far more visible
    #: magnified, which is where manual focus is actually used. A single step
    #: is the finest the body accepts, and it does take it.
    #: it seems minimum value depends on lens - e.g. AF-S 60/2.8 micro reacts to 6, 
    #: while 24-140/4 reacts only from 18
    FOCUS_STEP_DEFAULTS = {"minimum": 6, "fine": 50, "medium": 250, "coarse": 1000}

    #: MfDrive's direction parameter for focusing closer. Two is the other way.
    #: This follows Nikon's usual convention -- it could not be confirmed from
    #: live view here, because at this subject distance both ends of travel
    #: blur the scene equally. Swap the two values if a lens focuses backwards.
    _FOCUS_NEARER, _FOCUS_FURTHER = 1, 2

    def drive_focus(self, steps: int) -> bool:
        """Nudge focus manually. Negative drives nearer, positive further.

        Returns False if the lens is already at that end of its travel.
        """
        direction = self._FOCUS_FURTHER if steps >= 0 else self._FOCUS_NEARER
        # The body answers DEVICE_BUSY while it is still settling from a zoom
        # change or a previous focus move, which is easy to run into when the
        # buttons auto-repeat.
        deadline = time.monotonic() + 3.0
        while True:
            try:
                self.session.execute(Op.NIKON_MF_DRIVE, (direction, abs(int(steps))))
                break
            except MtpError as exc:
                if exc.response in (
                    Response.NIKON_MF_DRIVE_STEP_END,
                    Response.NIKON_MF_DRIVE_STEP_INSUFFICIENT,
                ):
                    return False
                if exc.response == Response.NIKON_NOT_LIVE_VIEW:
                    raise CameraError(
                        "focus can only be driven from the computer in live view"
                    ) from exc
                busy = exc.response in (
                    Response.DEVICE_BUSY,
                    Response.NIKON_INVALID_STATUS,
                )
                if not busy or time.monotonic() >= deadline:
                    raise CameraError(
                        f"could not drive focus (0x{exc.response:04X})"
                    ) from exc
                time.sleep(0.1)
        self.wait_ready(timeout=5.0)
        return True

    # -- exposure settings -------------------------------------------------

    def setting(self, name: str) -> "Setting | None":
        for label, code, formatter in _SETTINGS:
            if label == name:
                return self._read_setting(label, code, formatter)
        return None

    def settings(self) -> "list[Setting]":
        """Every exposure control the body reports, ready for display."""
        supported = set(self.session.device_info().device_properties_supported)
        out = []
        for label, code, formatter in _SETTINGS:
            if code not in supported:
                continue
            found = self._read_setting(label, code, formatter)
            if found is not None:
                out.append(found)
        return out

    def _read_setting(
        self, name: str, code: int, formatter: Callable[[Any], str]
    ) -> "Setting | None":
        try:
            desc = self.session.prop_desc(code)
        except (MtpError, WpdCommandError, ValueError):
            return None
        choices = tuple((value, formatter(value)) for value in desc.allowed_values)
        return Setting(
            code=code,
            name=name,
            value=desc.current,
            label=formatter(desc.current),
            writable=desc.writable,
            choices=choices,
        )

    def set_setting(self, name: str, value: Any) -> None:
        for label, code, _ in _SETTINGS:
            if label == name:
                try:
                    self.session.set_prop(code, value)
                except MtpError as exc:
                    raise CameraError(
                        f"camera refused {name} = {value} (0x{exc.response:04X})"
                    ) from exc
                return
        raise KeyError(name)

    def battery_level(self) -> "int | None":
        try:
            return int(self.session.get_prop(Prop.BATTERY_LEVEL))
        except (MtpError, WpdCommandError, ValueError):
            return None

    # -- capture -----------------------------------------------------------

    def capture(self, autofocus: bool = False, timeout: float = 20.0) -> "list[int]":
        """Take a picture. Returns the handles of any objects the camera created.

        The shot goes to the card, which is what makes it a full-resolution raw
        or JPEG rather than the SDRAM preview. Live view stays running.
        """
        with self._lock:
            if autofocus:
                # Driven as its own step rather than through a capture
                # parameter, so a failure to lock is reported before the
                # shutter is asked to fire.
                self.autofocus()
            self.wait_ready(timeout=5.0)
            opcode = Op.NIKON_INITIATE_CAPTURE_REC_IN_MEDIA
            # Both parameters are fixed: any storage, and a second parameter of
            # zero. A value of one is accepted but then never completes, and
            # leaves a capture pending that has to be torn down.
            params: "tuple[int, ...]" = (0xFFFFFFFF, 0x0000)
            try:
                self.session.execute(opcode, params)
            except MtpError as exc:
                if exc.response in (
                    Response.OPERATION_NOT_SUPPORTED,
                    Response.PARAMETER_NOT_SUPPORTED,
                    Response.INVALID_PARAMETER,
                ):
                    self.session.execute(Op.INITIATE_CAPTURE, (0x00000000, 0x00000000))
                elif exc.response == Response.NIKON_OUT_OF_FOCUS:
                    raise CameraError(
                        "shutter did not fire: autofocus could not lock. Focus "
                        "first, or switch the lens to manual."
                    ) from exc
                else:
                    raise CameraError(
                        f"shutter did not fire (0x{exc.response:04X})"
                    ) from exc
            return self._collect_capture_handles(timeout)

    def _collect_capture_handles(self, timeout: float) -> "list[int]":
        handles: "list[int]" = []
        deadline = time.monotonic() + timeout
        complete = False
        while time.monotonic() < deadline and not complete:
            for code, param in self.poll_events():
                if code in (Event.OBJECT_ADDED, Event.NIKON_OBJECT_ADDED_IN_SDRAM):
                    handles.append(param)
                elif code in (
                    Event.CAPTURE_COMPLETE,
                    Event.NIKON_CAPTURE_COMPLETE_RECV_IN_SDRAM,
                ):
                    complete = True
            if not complete:
                time.sleep(0.05)
        if not complete:
            # Leaving a capture outstanding stops the camera accepting the next
            # one, and can leave live view unable to restart until the body is
            # power cycled.
            self.session.try_execute(Op.NIKON_TERMINATE_CAPTURE)
        return handles

    def poll_events(self) -> "list[tuple[int, int]]":
        """Drain the camera's event queue as (event code, parameter) pairs."""
        try:
            data, _ = self.session.read(Op.NIKON_GET_EVENT)
        except (MtpError, WpdCommandError):
            return []
        if len(data) < 2:
            return []
        reader = Reader(data)
        count = reader.uint16()
        events = []
        for _ in range(count):
            try:
                events.append((reader.uint16(), reader.uint32()))
            except ValueError:
                break
        return events

    def download(self, handle: int) -> "tuple[str, bytes]":
        """Fetch a captured image by handle, as (filename, bytes)."""
        info = self.session.object_info(handle)
        return info.filename, self.session.get_object(handle)
