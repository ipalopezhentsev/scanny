"""A PTP session layered over a transport.

The transport supplied here is :class:`scanny.wpd.device.WpdMtpTransport`, which
already owns the PTP session Windows opened with the camera -- so there is no
``OpenSession`` call to make. What this class adds is the typed layer: device
info, property descriptors, and get/set of property values with the right
datatype encoding.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol, Sequence

from ..wpd.device import MtpError
from .codes import Op, Response
from .parser import DeviceInfo, ObjectInfo, PropDesc, Writer

__all__ = ["PtpSession", "Transport"]


class Transport(Protocol):
    """The pipe a session needs: three PTP transaction shapes."""

    def execute(self, opcode: int, params: Sequence[int] = ()) -> "list[int]": ...

    def read(self, opcode: int, params: Sequence[int] = ()) -> "tuple[bytes, list[int]]": ...

    def write(self, opcode: int, data: bytes, params: Sequence[int] = ()) -> "list[int]": ...


class PtpSession:
    """Typed PTP operations over a transport.

    All camera traffic is serialised through one lock: PTP is a
    request/response protocol with a single outstanding transaction, and the UI
    drives it from both a live-view thread and the main thread.
    """

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._lock = threading.RLock()
        self._device_info: "DeviceInfo | None" = None
        self._desc_cache: "dict[int, PropDesc]" = {}

    @property
    def lock(self) -> threading.RLock:
        """The transaction lock, for callers composing multi-step sequences."""
        return self._lock

    @property
    def transport(self) -> Transport:
        return self._transport

    # -- raw transactions --------------------------------------------------

    def execute(self, opcode: int, params: Sequence[int] = ()) -> "list[int]":
        with self._lock:
            return self._transport.execute(opcode, params)

    def read(self, opcode: int, params: Sequence[int] = ()) -> "tuple[bytes, list[int]]":
        with self._lock:
            return self._transport.read(opcode, params)

    def write(self, opcode: int, data: bytes, params: Sequence[int] = ()) -> "list[int]":
        with self._lock:
            return self._transport.write(opcode, data, params)

    # -- device info -------------------------------------------------------

    def device_info(self, refresh: bool = False) -> DeviceInfo:
        with self._lock:
            if self._device_info is None or refresh:
                data, _ = self._transport.read(Op.GET_DEVICE_INFO)
                self._device_info = DeviceInfo.parse(data)
            return self._device_info

    # -- device properties -------------------------------------------------

    def prop_desc(self, code: int, refresh: bool = True) -> PropDesc:
        """The full descriptor for a property: datatype, writability, allowed values."""
        with self._lock:
            if not refresh and code in self._desc_cache:
                return self._desc_cache[code]
            data, _ = self._transport.read(Op.GET_DEVICE_PROP_DESC, (code,))
            desc = PropDesc.parse(data)
            self._desc_cache[code] = desc
            return desc

    def get_prop(self, code: int) -> Any:
        """A property's current value, decoded to its declared datatype."""
        with self._lock:
            datatype = self._datatype_for(code)
            data, _ = self._transport.read(Op.GET_DEVICE_PROP_VALUE, (code,))
            from .parser import Reader

            return Reader(data).value(datatype)

    def set_prop(self, code: int, value: Any) -> None:
        """Write a property, encoding to its declared datatype."""
        with self._lock:
            datatype = self._datatype_for(code)
            payload = Writer().value(datatype, value).bytes()
            self._transport.write(Op.SET_DEVICE_PROP_VALUE, payload, (code,))

    def _datatype_for(self, code: int) -> int:
        cached = self._desc_cache.get(code)
        if cached is not None:
            return cached.datatype
        return self.prop_desc(code).datatype

    def supported_properties(self) -> "list[int]":
        return list(self.device_info().device_properties_supported)

    # -- objects -----------------------------------------------------------

    def object_info(self, handle: int) -> ObjectInfo:
        data, _ = self.read(Op.GET_OBJECT_INFO, (handle,))
        return ObjectInfo.parse(data)

    def get_object(self, handle: int) -> bytes:
        data, _ = self.read(Op.GET_OBJECT, (handle,))
        return data

    def delete_object(self, handle: int) -> None:
        self.execute(Op.DELETE_OBJECT, (handle,))

    # -- helpers -----------------------------------------------------------

    def try_execute(self, opcode: int, params: Sequence[int] = ()) -> "int | None":
        """Run an operation, returning the response code instead of raising."""
        try:
            self.execute(opcode, params)
            return Response.OK
        except MtpError as exc:
            return exc.response
