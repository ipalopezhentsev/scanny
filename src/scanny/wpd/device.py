"""Device enumeration and the MTP-extension transport built on WPD.

Windows binds MTP cameras to its own ``WUDFWpdMtp`` driver, so the libusb route
libgphoto2 takes is unavailable without swapping the driver out from under
every other application on the machine. WPD's MTP extension commands give us a
raw PTP pipe instead: ``SendCommand`` carries an opcode, up to five parameters
and a data phase straight through to the camera.

The one restriction is that the driver only forwards *vendor* opcodes -- those
at 0x9000 and above. Standard PTP operations are reserved to the driver itself.
Everything this project needs from the D750 (live view, autofocus, capture,
event polling) is a Nikon vendor opcode, and the standard device properties are
reachable through :mod:`scanny.wpd.properties` instead.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_ubyte, c_ulong, c_ulonglong, c_wchar_p, create_unicode_buffer
from dataclasses import dataclass
from typing import Sequence

from comtypes import CoCreateInstance, GUID

from . import constants as k
from ._com import (
    CLSID_PortableDeviceFTM,
    CLSID_PortableDeviceManager,
    CLSID_PortableDevicePropVariantCollection,
    CLSID_PortableDeviceValues,
    PROPVARIANT,
    IPortableDevice,
    IPortableDeviceManager,
    IPortableDevicePropVariantCollection,
    IPortableDeviceValues,
    co_task_mem_free,
)

__all__ = [
    "DeviceInfo",
    "enumerate_devices",
    "WpdMtpTransport",
    "MtpError",
    "WpdCommandError",
]


class MtpError(RuntimeError):
    """A PTP/MTP response code other than OK."""

    def __init__(self, opcode: int, response: int, message: str = "") -> None:
        self.opcode = opcode
        self.response = response
        detail = f": {message}" if message else ""
        super().__init__(
            f"MTP operation 0x{opcode:04X} failed with response 0x{response:04X}{detail}"
        )


class WpdCommandError(RuntimeError):
    """The WPD driver rejected the command before it reached the camera.

    This usually means the transport is wedged -- a data phase was abandoned
    part way through, for instance -- and the device needs reopening.
    """

    def __init__(self, hresult: int) -> None:
        self.hresult = hresult
        super().__init__(f"WPD command failed with HRESULT 0x{hresult:08X}")


@dataclass(frozen=True)
class DeviceInfo:
    """A portable device as Windows reports it."""

    pnp_id: str
    friendly_name: str
    description: str
    manufacturer: str

    @property
    def vid_pid(self) -> "tuple[int, int] | None":
        """The USB vendor/product ids parsed out of the PnP id, if present."""
        lowered = self.pnp_id.lower()
        try:
            vid = lowered.split("vid_", 1)[1][:4]
            pid = lowered.split("pid_", 1)[1][:4]
            return int(vid, 16), int(pid, 16)
        except (IndexError, ValueError):
            return None


def _create(clsid: GUID, interface: type) -> object:
    return CoCreateInstance(clsid, interface=interface)


def _query_string(fn, pnp_id: str) -> str:
    length = c_ulong(0)
    try:
        fn(pnp_id, None, byref(length))
    except OSError:
        return ""
    if not length.value:
        return ""
    buf = create_unicode_buffer(length.value)
    try:
        fn(pnp_id, ctypes.cast(buf, c_wchar_p), byref(length))
    except OSError:
        return ""
    return buf.value


def enumerate_devices() -> "list[DeviceInfo]":
    """Every portable device Windows currently reports."""
    mgr = _create(CLSID_PortableDeviceManager, IPortableDeviceManager)
    mgr.RefreshDeviceList()
    count = c_ulong(0)
    mgr.GetDevices(None, byref(count))
    if not count.value:
        return []
    ids = (c_wchar_p * count.value)()
    mgr.GetDevices(ctypes.cast(ids, POINTER(c_wchar_p)), byref(count))
    devices = []
    for i in range(count.value):
        pnp_id = ids[i]
        if not pnp_id:
            continue
        devices.append(
            DeviceInfo(
                pnp_id=pnp_id,
                friendly_name=_query_string(mgr.GetDeviceFriendlyName, pnp_id),
                description=_query_string(mgr.GetDeviceDescription, pnp_id),
                manufacturer=_query_string(mgr.GetDeviceManufacturer, pnp_id),
            )
        )
    return devices


def _new_values() -> IPortableDeviceValues:
    return _create(CLSID_PortableDeviceValues, IPortableDeviceValues)


def _new_params(codes: Sequence[int]) -> IPortableDevicePropVariantCollection:
    coll = _create(
        CLSID_PortableDevicePropVariantCollection, IPortableDevicePropVariantCollection
    )
    for value in codes:
        pv = PROPVARIANT.from_uint32(value)
        coll.Add(byref(pv))
    return coll


def _read_uint32_collection(values: IPortableDeviceValues, key) -> "list[int]":
    coll = POINTER(IPortableDevicePropVariantCollection)()
    try:
        values.GetIPortableDevicePropVariantCollectionValue(byref(key), byref(coll))
    except OSError:
        return []
    if not coll:
        return []
    count = c_ulong(0)
    coll.GetCount(byref(count))
    out = []
    for i in range(count.value):
        pv = PROPVARIANT()
        coll.GetAt(i, byref(pv))
        out.append(pv.as_uint32())
        pv.clear()
    return out


class WpdMtpTransport:
    """A raw PTP pipe to one MTP device, via WPD's vendor-operation commands."""

    #: Chunk size for data-phase reads when the driver reports nothing better.
    DEFAULT_READ_CHUNK = 512 * 1024

    def __init__(self, pnp_id: str, client_name: str = "scanny") -> None:
        self.pnp_id = pnp_id
        self._client_name = client_name
        self._device: "IPortableDevice | None" = None
        self._vendor_opcodes: "frozenset[int]" = frozenset()

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> "WpdMtpTransport":
        info = _new_values()
        info.SetStringValue(byref(k.WPD_CLIENT_NAME), self._client_name)
        info.SetUnsignedIntegerValue(byref(k.WPD_CLIENT_MAJOR_VERSION), 1)
        info.SetUnsignedIntegerValue(byref(k.WPD_CLIENT_MINOR_VERSION), 0)
        info.SetUnsignedIntegerValue(byref(k.WPD_CLIENT_REVISION), 0)
        info.SetUnsignedIntegerValue(
            byref(k.WPD_CLIENT_SECURITY_QUALITY_OF_SERVICE), k.SECURITY_IMPERSONATION
        )
        info.SetUnsignedIntegerValue(
            byref(k.WPD_CLIENT_DESIRED_ACCESS), k.GENERIC_READ | k.GENERIC_WRITE
        )
        device = _create(CLSID_PortableDeviceFTM, IPortableDevice)
        device.Open(self.pnp_id, info)
        self._device = device
        self._vendor_opcodes = frozenset(self._query_vendor_opcodes())
        return self

    def close(self) -> None:
        if self._device is not None:
            try:
                self._device.Close()
            except OSError:
                pass
            self._device = None

    def reopen(self) -> "WpdMtpTransport":
        """Close and reopen the device, clearing any wedged transfer state."""
        self.close()
        return self.open()

    def __enter__(self) -> "WpdMtpTransport":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def device(self) -> IPortableDevice:
        if self._device is None:
            raise RuntimeError("transport is not open")
        return self._device

    @property
    def vendor_opcodes(self) -> "frozenset[int]":
        """Vendor opcodes the driver is willing to forward to this camera."""
        return self._vendor_opcodes

    # -- command plumbing --------------------------------------------------

    def _command(self, command_id: int) -> IPortableDeviceValues:
        params = _new_values()
        category = GUID(k.WPD_CATEGORY_MTP_EXT_VENDOR_OPERATIONS)
        params.SetGuidValue(byref(k.WPD_PROPERTY_COMMON_COMMAND_CATEGORY), byref(category))
        params.SetUnsignedIntegerValue(byref(k.WPD_PROPERTY_COMMON_COMMAND_ID), command_id)
        return params

    def _send(self, params: IPortableDeviceValues) -> IPortableDeviceValues:
        results = POINTER(IPortableDeviceValues)()
        self.device.SendCommand(0, params, byref(results))
        if not results:
            raise RuntimeError("SendCommand returned no results")
        # A failure here is the driver refusing the command, and the results
        # then carry nothing else -- no transfer context, no response code.
        # Raising now turns what would otherwise surface as a bare COMError
        # from some later lookup into the error callers already handle.
        hresult = self._common_hresult(results)
        if hresult:
            raise WpdCommandError(hresult)
        return results

    def _query_vendor_opcodes(self) -> "list[int]":
        results = self._send(self._command(k.CMD_GET_SUPPORTED_VENDOR_OPCODES))
        return _read_uint32_collection(results, k.WPD_PROPERTY_MTP_EXT_VENDOR_OPERATION_CODES)

    def vendor_extension_description(self) -> str:
        results = self._send(self._command(k.CMD_GET_VENDOR_EXTENSION_DESCRIPTION))
        text = c_wchar_p()
        try:
            results.GetStringValue(
                byref(k.WPD_PROPERTY_MTP_EXT_VENDOR_EXTENSION_DESCRIPTION), byref(text)
            )
        except OSError:
            return ""
        value = text.value or ""
        co_task_mem_free(text)
        return value

    @staticmethod
    def _common_hresult(results: IPortableDeviceValues) -> int:
        """The driver-level HRESULT for the command, as opposed to the MTP
        response code. A failure here means the request never reached the
        camera, so no response code will be present in the results."""
        pv = PROPVARIANT()
        try:
            results.GetValue(byref(k.WPD_PROPERTY_COMMON_HRESULT), byref(pv))
        except OSError:
            return 0
        value = int(pv.u.scode) & 0xFFFFFFFF
        pv.clear()
        return value

    @classmethod
    def _response(cls, results: IPortableDeviceValues) -> "tuple[int, list[int]]":
        code = c_ulong(0)
        try:
            results.GetUnsignedIntegerValue(
                byref(k.WPD_PROPERTY_MTP_EXT_RESPONSE_CODE), byref(code)
            )
        except OSError:
            # _send already rejects a failing driver HRESULT, so results
            # without a response code here mean something stranger.
            raise WpdCommandError(cls._common_hresult(results)) from None
        params = _read_uint32_collection(results, k.WPD_PROPERTY_MTP_EXT_RESPONSE_PARAMS)
        return code.value, params

    # -- the three PTP transaction shapes ----------------------------------

    def execute(self, opcode: int, params: Sequence[int] = ()) -> "list[int]":
        """A PTP transaction with no data phase. Returns the response parameters."""
        req = self._command(k.CMD_EXECUTE_WITHOUT_DATA_PHASE)
        req.SetUnsignedIntegerValue(byref(k.WPD_PROPERTY_MTP_EXT_OPERATION_CODE), opcode)
        req.SetIPortableDevicePropVariantCollectionValue(
            byref(k.WPD_PROPERTY_MTP_EXT_OPERATION_PARAMS), _new_params(params)
        )
        code, response_params = self._response(self._send(req))
        if code != 0x2001:
            raise MtpError(opcode, code)
        return response_params

    def read(self, opcode: int, params: Sequence[int] = ()) -> "tuple[bytes, list[int]]":
        """A PTP transaction whose data phase flows device -> host."""
        req = self._command(k.CMD_EXECUTE_WITH_DATA_TO_READ)
        req.SetUnsignedIntegerValue(byref(k.WPD_PROPERTY_MTP_EXT_OPERATION_CODE), opcode)
        req.SetIPortableDevicePropVariantCollectionValue(
            byref(k.WPD_PROPERTY_MTP_EXT_OPERATION_PARAMS), _new_params(params)
        )
        results = self._send(req)

        context = c_wchar_p()
        results.GetStringValue(byref(k.WPD_PROPERTY_MTP_EXT_TRANSFER_CONTEXT), byref(context))
        context_str = context.value
        co_task_mem_free(context)

        total = c_ulonglong(0)
        results.GetUnsignedLargeIntegerValue(
            byref(k.WPD_PROPERTY_MTP_EXT_TRANSFER_TOTAL_DATA_SIZE), byref(total)
        )
        chunk = c_ulong(0)
        try:
            results.GetUnsignedIntegerValue(
                byref(k.WPD_PROPERTY_MTP_EXT_OPTIMAL_TRANSFER_BUFFER_SIZE), byref(chunk)
            )
        except OSError:
            chunk.value = 0
        chunk_size = chunk.value or self.DEFAULT_READ_CHUNK

        try:
            data = self._read_data(context_str, total.value, chunk_size)
            code, response_params = self._end_transfer(context_str)
        except Exception:
            self._end_transfer(context_str, ignore_errors=True)
            raise
        if code != 0x2001:
            raise MtpError(opcode, code)
        return data, response_params

    def _read_data(self, context: str, total: int, chunk_size: int) -> bytes:
        out = bytearray()
        remaining = total
        scratch = (c_ubyte * chunk_size)()
        while remaining > 0:
            want = min(chunk_size, remaining)
            req = self._command(k.CMD_READ_DATA)
            req.SetStringValue(byref(k.WPD_PROPERTY_MTP_EXT_TRANSFER_CONTEXT), context)
            req.SetUnsignedIntegerValue(
                byref(k.WPD_PROPERTY_MTP_EXT_TRANSFER_NUM_BYTES_TO_READ), want
            )
            # The driver fills a buffer the caller supplies; without this the
            # read fails with ERROR_NOT_FOUND and wedges the transfer context.
            req.SetBufferValue(
                byref(k.WPD_PROPERTY_MTP_EXT_TRANSFER_DATA),
                ctypes.cast(scratch, POINTER(c_ubyte)),
                want,
            )
            results = self._send(req)

            buf = POINTER(c_ubyte)()
            size = c_ulong(0)
            results.GetBufferValue(
                byref(k.WPD_PROPERTY_MTP_EXT_TRANSFER_DATA), byref(buf), byref(size)
            )
            if not buf or not size.value:
                if buf:
                    co_task_mem_free(buf)
                break
            out += bytes(ctypes.cast(buf, POINTER(c_ubyte * size.value)).contents)
            co_task_mem_free(buf)
            remaining -= size.value
        return bytes(out)

    def _end_transfer(
        self, context: str, ignore_errors: bool = False
    ) -> "tuple[int, list[int]]":
        req = self._command(k.CMD_END_DATA_TRANSFER)
        req.SetStringValue(byref(k.WPD_PROPERTY_MTP_EXT_TRANSFER_CONTEXT), context)
        try:
            return self._response(self._send(req))
        except (OSError, WpdCommandError):
            if ignore_errors:
                return 0x2001, []
            raise

    def write(
        self, opcode: int, data: bytes, params: Sequence[int] = ()
    ) -> "list[int]":
        """A PTP transaction whose data phase flows host -> device."""
        req = self._command(k.CMD_EXECUTE_WITH_DATA_TO_WRITE)
        req.SetUnsignedIntegerValue(byref(k.WPD_PROPERTY_MTP_EXT_OPERATION_CODE), opcode)
        req.SetIPortableDevicePropVariantCollectionValue(
            byref(k.WPD_PROPERTY_MTP_EXT_OPERATION_PARAMS), _new_params(params)
        )
        req.SetUnsignedLargeIntegerValue(
            byref(k.WPD_PROPERTY_MTP_EXT_TRANSFER_TOTAL_DATA_SIZE), len(data)
        )
        results = self._send(req)

        context = c_wchar_p()
        results.GetStringValue(byref(k.WPD_PROPERTY_MTP_EXT_TRANSFER_CONTEXT), byref(context))
        context_str = context.value
        co_task_mem_free(context)

        chunk = c_ulong(0)
        try:
            results.GetUnsignedIntegerValue(
                byref(k.WPD_PROPERTY_MTP_EXT_OPTIMAL_TRANSFER_BUFFER_SIZE), byref(chunk)
            )
        except OSError:
            chunk.value = 0
        chunk_size = chunk.value or self.DEFAULT_READ_CHUNK

        try:
            for offset in range(0, max(len(data), 1), chunk_size):
                piece = data[offset : offset + chunk_size]
                write_req = self._command(k.CMD_WRITE_DATA)
                write_req.SetStringValue(
                    byref(k.WPD_PROPERTY_MTP_EXT_TRANSFER_CONTEXT), context_str
                )
                buf = (c_ubyte * len(piece)).from_buffer_copy(piece) if piece else (c_ubyte * 0)()
                write_req.SetBufferValue(
                    byref(k.WPD_PROPERTY_MTP_EXT_TRANSFER_DATA),
                    ctypes.cast(buf, POINTER(c_ubyte)),
                    len(piece),
                )
                write_req.SetUnsignedIntegerValue(
                    byref(k.WPD_PROPERTY_MTP_EXT_TRANSFER_NUM_BYTES_TO_WRITE), len(piece)
                )
                self._send(write_req)
            code, response_params = self._end_transfer(context_str)
        except Exception:
            self._end_transfer(context_str, ignore_errors=True)
            raise
        if code != 0x2001:
            raise MtpError(opcode, code)
        return response_params
