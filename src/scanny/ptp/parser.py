"""Readers and writers for PTP's little-endian dataset encoding."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

from .codes import ARRAY_ELEMENT, SCALAR_FORMATS, DataType, FormFlag

__all__ = ["Reader", "Writer", "DeviceInfo", "PropDesc", "ObjectInfo"]


class Reader:
    """A cursor over a PTP dataset."""

    def __init__(self, data: bytes, offset: int = 0) -> None:
        self._data = data
        self.offset = offset

    @property
    def remaining(self) -> int:
        return len(self._data) - self.offset

    def _unpack(self, fmt: str, size: int) -> Any:
        if self.offset + size > len(self._data):
            raise ValueError(
                f"PTP dataset truncated: need {size} bytes at offset {self.offset}, "
                f"have {len(self._data) - self.offset}"
            )
        value = struct.unpack_from("<" + fmt, self._data, self.offset)[0]
        self.offset += size
        return value

    def uint8(self) -> int:
        return self._unpack("B", 1)

    def int8(self) -> int:
        return self._unpack("b", 1)

    def uint16(self) -> int:
        return self._unpack("H", 2)

    def int16(self) -> int:
        return self._unpack("h", 2)

    def uint32(self) -> int:
        return self._unpack("I", 4)

    def int32(self) -> int:
        return self._unpack("i", 4)

    def uint64(self) -> int:
        return self._unpack("Q", 8)

    def int64(self) -> int:
        return self._unpack("q", 8)

    def string(self) -> str:
        """A PTP string: a character count followed by UCS-2, NUL terminated."""
        count = self.uint8()
        if count == 0:
            return ""
        raw = self._data[self.offset : self.offset + count * 2]
        self.offset += count * 2
        return raw.decode("utf-16-le").rstrip("\x00")

    def uint16_array(self) -> "list[int]":
        return [self.uint16() for _ in range(self.uint32())]

    def uint32_array(self) -> "list[int]":
        return [self.uint32() for _ in range(self.uint32())]

    def value(self, datatype: int) -> Any:
        """One value of the given PTP datatype."""
        if datatype == DataType.STR:
            return self.string()
        if datatype in SCALAR_FORMATS:
            fmt, size = SCALAR_FORMATS[datatype]
            return self._unpack(fmt, size)
        if datatype in (DataType.INT128, DataType.UINT128):
            raw = self._data[self.offset : self.offset + 16]
            self.offset += 16
            return int.from_bytes(raw, "little", signed=datatype == DataType.INT128)
        if datatype in ARRAY_ELEMENT:
            element = ARRAY_ELEMENT[datatype]
            return [self.value(element) for _ in range(self.uint32())]
        raise ValueError(f"unsupported PTP datatype 0x{datatype:04X}")


class Writer:
    """Builds a PTP dataset."""

    def __init__(self) -> None:
        self._parts: "list[bytes]" = []

    def _pack(self, fmt: str, value: Any) -> "Writer":
        self._parts.append(struct.pack("<" + fmt, value))
        return self

    def uint8(self, value: int) -> "Writer":
        return self._pack("B", value)

    def uint16(self, value: int) -> "Writer":
        return self._pack("H", value)

    def uint32(self, value: int) -> "Writer":
        return self._pack("I", value)

    def string(self, value: str) -> "Writer":
        if not value:
            return self.uint8(0)
        encoded = (value + "\x00").encode("utf-16-le")
        self.uint8(len(value) + 1)
        self._parts.append(encoded)
        return self

    def value(self, datatype: int, value: Any) -> "Writer":
        if datatype == DataType.STR:
            return self.string(value)
        if datatype in SCALAR_FORMATS:
            fmt, _ = SCALAR_FORMATS[datatype]
            return self._pack(fmt, value)
        if datatype in ARRAY_ELEMENT:
            element = ARRAY_ELEMENT[datatype]
            self.uint32(len(value))
            for item in value:
                self.value(element, item)
            return self
        raise ValueError(f"unsupported PTP datatype 0x{datatype:04X}")

    def bytes(self) -> bytes:
        return b"".join(self._parts)


@dataclass
class DeviceInfo:
    """The PTP ``DeviceInfo`` dataset."""

    standard_version: int
    vendor_extension_id: int
    vendor_extension_version: int
    vendor_extension_desc: str
    functional_mode: int
    operations_supported: "list[int]"
    events_supported: "list[int]"
    device_properties_supported: "list[int]"
    capture_formats: "list[int]"
    image_formats: "list[int]"
    manufacturer: str
    model: str
    device_version: str
    serial_number: str

    @classmethod
    def parse(cls, data: bytes) -> "DeviceInfo":
        r = Reader(data)
        return cls(
            standard_version=r.uint16(),
            vendor_extension_id=r.uint32(),
            vendor_extension_version=r.uint16(),
            vendor_extension_desc=r.string(),
            functional_mode=r.uint16(),
            operations_supported=r.uint16_array(),
            events_supported=r.uint16_array(),
            device_properties_supported=r.uint16_array(),
            capture_formats=r.uint16_array(),
            image_formats=r.uint16_array(),
            manufacturer=r.string(),
            model=r.string(),
            device_version=r.string(),
            serial_number=r.string(),
        )

    def supports(self, opcode: int) -> bool:
        return opcode in self.operations_supported

    def has_property(self, code: int) -> bool:
        return code in self.device_properties_supported


@dataclass
class PropDesc:
    """The PTP ``DevicePropDesc`` dataset: type, writability and allowed values."""

    code: int
    datatype: int
    writable: bool
    factory_default: Any
    current: Any
    form: int = FormFlag.NONE
    minimum: Any = None
    maximum: Any = None
    step: Any = None
    enumeration: "list[Any]" = field(default_factory=list)

    @classmethod
    def parse(cls, data: bytes) -> "PropDesc":
        r = Reader(data)
        code = r.uint16()
        datatype = r.uint16()
        writable = r.uint8() == 1
        factory_default = r.value(datatype)
        current = r.value(datatype)
        desc = cls(
            code=code,
            datatype=datatype,
            writable=writable,
            factory_default=factory_default,
            current=current,
        )
        # The form field is optional; some properties end after the current value.
        if r.remaining == 0:
            return desc
        desc.form = r.uint8()
        if desc.form == FormFlag.RANGE:
            desc.minimum = r.value(datatype)
            desc.maximum = r.value(datatype)
            desc.step = r.value(datatype)
        elif desc.form == FormFlag.ENUMERATION:
            desc.enumeration = [r.value(datatype) for _ in range(r.uint16())]
        return desc

    @property
    def allowed_values(self) -> "list[Any]":
        """Every value the camera will accept, when it says so explicitly."""
        if self.form == FormFlag.ENUMERATION:
            return list(self.enumeration)
        if self.form == FormFlag.RANGE and all(
            isinstance(v, int) for v in (self.minimum, self.maximum, self.step)
        ):
            if self.step <= 0:
                return []
            return list(range(self.minimum, self.maximum + 1, self.step))
        return []


@dataclass
class ObjectInfo:
    """The PTP ``ObjectInfo`` dataset, trimmed to the fields we use."""

    storage_id: int
    object_format: int
    protection_status: int
    compressed_size: int
    thumb_format: int
    thumb_compressed_size: int
    thumb_width: int
    thumb_height: int
    image_width: int
    image_height: int
    image_bit_depth: int
    parent_object: int
    association_type: int
    association_desc: int
    sequence_number: int
    filename: str
    capture_date: str
    modification_date: str
    keywords: str

    @classmethod
    def parse(cls, data: bytes) -> "ObjectInfo":
        r = Reader(data)
        return cls(
            storage_id=r.uint32(),
            object_format=r.uint16(),
            protection_status=r.uint16(),
            compressed_size=r.uint32(),
            thumb_format=r.uint16(),
            thumb_compressed_size=r.uint32(),
            thumb_width=r.uint32(),
            thumb_height=r.uint32(),
            image_width=r.uint32(),
            image_height=r.uint32(),
            image_bit_depth=r.uint32(),
            parent_object=r.uint32(),
            association_type=r.uint16(),
            association_desc=r.uint32(),
            sequence_number=r.uint32(),
            filename=r.string(),
            capture_date=r.string(),
            modification_date=r.string(),
            keywords=r.string(),
        )
