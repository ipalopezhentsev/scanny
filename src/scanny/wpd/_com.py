"""Hand-written ctypes/comtypes bindings for the Windows Portable Devices API.

The typelibs embedded in ``portabledeviceapi.dll`` / ``portabledevicetypes.dll``
are defective for our purposes: ``comtypes.client.GetModule`` drops the out
parameter of ``GetIPortableDevicePropVariantCollectionValue`` and both GUID
parameters of ``Set/GetGuidValue``, and it mis-marshals the ``[in, out]``
parameters of ``IPortableDeviceManager.GetDevices`` (which silently yields a
device count of zero). Declaring the vtables by hand costs a page of code and
makes the marshalling exact.

Every out parameter below is declared ``["in"]`` and takes an explicit
``byref()`` from the caller, so comtypes never rewrites a signature on us.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    Structure,
    Union,
    c_byte,
    c_double,
    c_float,
    c_int,
    c_long,
    c_longlong,
    c_short,
    c_ubyte,
    c_ulong,
    c_ulonglong,
    c_ushort,
    c_void_p,
    c_wchar_p,
)

from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

_ole32 = ctypes.oledll.ole32

VT_EMPTY = 0
VT_I2 = 2
VT_I4 = 3
VT_ERROR = 10
VT_BOOL = 11
VT_UNKNOWN = 13
VT_I1 = 16
VT_UI1 = 17
VT_UI2 = 18
VT_UI4 = 19
VT_I8 = 20
VT_UI8 = 21
VT_LPWSTR = 31
VT_CLSID = 72


class PROPERTYKEY(Structure):
    """The ``PROPERTYKEY`` struct: a format GUID plus a property id."""

    _fields_ = [("fmtid", GUID), ("pid", c_ulong)]

    def __init__(self, fmtid: "str | GUID | None" = None, pid: int = 0) -> None:
        if fmtid is None:
            super().__init__()
        else:
            super().__init__(GUID(fmtid) if isinstance(fmtid, str) else fmtid, pid)

    def __repr__(self) -> str:
        return f"PROPERTYKEY({self.fmtid}, {self.pid})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PROPERTYKEY):
            return NotImplemented
        return self.fmtid == other.fmtid and self.pid == other.pid

    def __hash__(self) -> int:
        return hash((str(self.fmtid), self.pid))


class _CAUB(Structure):
    _fields_ = [("cElems", c_ulong), ("pElems", POINTER(c_ubyte))]


class _PROPVARIANT_UNION(Union):
    _fields_ = [
        ("cVal", c_byte),
        ("bVal", c_ubyte),
        ("iVal", c_short),
        ("uiVal", c_ushort),
        ("lVal", c_long),
        ("ulVal", c_ulong),
        ("intVal", c_int),
        ("hVal", c_longlong),
        ("uhVal", c_ulonglong),
        ("fltVal", c_float),
        ("dblVal", c_double),
        ("boolVal", c_short),
        ("scode", c_long),
        ("pwszVal", c_wchar_p),
        ("punkVal", POINTER(IUnknown)),
        ("puuid", POINTER(GUID)),
        ("caub", _CAUB),
    ]


class PROPVARIANT(Structure):
    """A ``PROPVARIANT`` covering the handful of types WPD actually uses."""

    _fields_ = [
        ("vt", c_ushort),
        ("wReserved1", c_ushort),
        ("wReserved2", c_ushort),
        ("wReserved3", c_ushort),
        ("u", _PROPVARIANT_UNION),
    ]

    @classmethod
    def from_uint32(cls, value: int) -> "PROPVARIANT":
        pv = cls()
        pv.vt = VT_UI4
        pv.u.ulVal = value & 0xFFFFFFFF
        return pv

    def as_uint32(self) -> int:
        return int(self.u.ulVal)

    def clear(self) -> None:
        prop_variant_clear(self)


def prop_variant_clear(pv: PROPVARIANT) -> None:
    _ole32.PropVariantClear(ctypes.byref(pv))


def co_task_mem_free(ptr: object) -> None:
    ctypes.windll.ole32.CoTaskMemFree(ctypes.cast(ptr, c_void_p))


CLSID_PortableDeviceManager = GUID("{0AF10CEC-2ECD-4B92-9581-34F6AE0637F3}")
CLSID_PortableDeviceFTM = GUID("{F7C0039A-4762-488A-B4B3-760EF9A1BA9B}")
CLSID_PortableDeviceValues = GUID("{0C15D503-D017-47CE-9016-7B3F978721CC}")
CLSID_PortableDevicePropVariantCollection = GUID("{08A99E2F-6D6D-4B80-AF5A-BAF2BCBE4CB9}")
CLSID_PortableDeviceKeyCollection = GUID("{DE2D022D-2480-43BE-97F0-D1FA2CF98F4F}")

_PK = POINTER(PROPERTYKEY)
_PV = POINTER(PROPVARIANT)


class IPortableDeviceValues(IUnknown):
    _iid_ = GUID("{6848F6F2-3155-4F86-B6F5-263EEEAB3143}")


class IPortableDevicePropVariantCollection(IUnknown):
    _iid_ = GUID("{89B2E422-4F1B-4316-BCEF-A44AFEA83EB3}")


class IPortableDeviceKeyCollection(IUnknown):
    _iid_ = GUID("{DADA2357-E0AD-492E-98DB-DD61C53BA353}")


class IPortableDeviceValuesCollection(IUnknown):
    _iid_ = GUID("{6E3F2D79-4E07-48C4-8208-D8C2E5AF4A99}")


IPortableDevicePropVariantCollection._methods_ = [
    COMMETHOD([], HRESULT, "GetCount", (["in"], POINTER(c_ulong), "pcElems")),
    COMMETHOD([], HRESULT, "GetAt", (["in"], c_ulong, "dwIndex"), (["in"], _PV, "pValue")),
    COMMETHOD([], HRESULT, "Add", (["in"], _PV, "pValue")),
    COMMETHOD([], HRESULT, "GetType", (["in"], POINTER(c_ushort), "pvt")),
    COMMETHOD([], HRESULT, "ChangeType", (["in"], c_ushort, "vt")),
    COMMETHOD([], HRESULT, "Clear"),
    COMMETHOD([], HRESULT, "RemoveAt", (["in"], c_ulong, "dwIndex")),
]

IPortableDeviceKeyCollection._methods_ = [
    COMMETHOD([], HRESULT, "GetCount", (["in"], POINTER(c_ulong), "pcElems")),
    COMMETHOD([], HRESULT, "GetAt", (["in"], c_ulong, "dwIndex"), (["in"], _PK, "pKey")),
    COMMETHOD([], HRESULT, "Add", (["in"], _PK, "Key")),
    COMMETHOD([], HRESULT, "Clear"),
    COMMETHOD([], HRESULT, "RemoveAt", (["in"], c_ulong, "dwIndex")),
]

IPortableDeviceValues._methods_ = [
    COMMETHOD([], HRESULT, "GetCount", (["in"], POINTER(c_ulong), "pcelt")),
    COMMETHOD([], HRESULT, "GetAt", (["in"], c_ulong, "index"), (["in"], _PK, "pKey"), (["in"], _PV, "pValue")),
    COMMETHOD([], HRESULT, "SetValue", (["in"], _PK, "key"), (["in"], _PV, "pValue")),
    COMMETHOD([], HRESULT, "GetValue", (["in"], _PK, "key"), (["in"], _PV, "pValue")),
    COMMETHOD([], HRESULT, "SetStringValue", (["in"], _PK, "key"), (["in"], c_wchar_p, "Value")),
    COMMETHOD([], HRESULT, "GetStringValue", (["in"], _PK, "key"), (["in"], POINTER(c_wchar_p), "pValue")),
    COMMETHOD([], HRESULT, "SetUnsignedIntegerValue", (["in"], _PK, "key"), (["in"], c_ulong, "Value")),
    COMMETHOD([], HRESULT, "GetUnsignedIntegerValue", (["in"], _PK, "key"), (["in"], POINTER(c_ulong), "pValue")),
    COMMETHOD([], HRESULT, "SetSignedIntegerValue", (["in"], _PK, "key"), (["in"], c_int, "Value")),
    COMMETHOD([], HRESULT, "GetSignedIntegerValue", (["in"], _PK, "key"), (["in"], POINTER(c_int), "pValue")),
    COMMETHOD([], HRESULT, "SetUnsignedLargeIntegerValue", (["in"], _PK, "key"), (["in"], c_ulonglong, "Value")),
    COMMETHOD([], HRESULT, "GetUnsignedLargeIntegerValue", (["in"], _PK, "key"), (["in"], POINTER(c_ulonglong), "pValue")),
    COMMETHOD([], HRESULT, "SetSignedLargeIntegerValue", (["in"], _PK, "key"), (["in"], c_longlong, "Value")),
    COMMETHOD([], HRESULT, "GetSignedLargeIntegerValue", (["in"], _PK, "key"), (["in"], POINTER(c_longlong), "pValue")),
    COMMETHOD([], HRESULT, "SetFloatValue", (["in"], _PK, "key"), (["in"], c_float, "Value")),
    COMMETHOD([], HRESULT, "GetFloatValue", (["in"], _PK, "key"), (["in"], POINTER(c_float), "pValue")),
    COMMETHOD([], HRESULT, "SetErrorValue", (["in"], _PK, "key"), (["in"], HRESULT, "Value")),
    COMMETHOD([], HRESULT, "GetErrorValue", (["in"], _PK, "key"), (["in"], POINTER(HRESULT), "pValue")),
    COMMETHOD([], HRESULT, "SetKeyValue", (["in"], _PK, "key"), (["in"], _PK, "Value")),
    COMMETHOD([], HRESULT, "GetKeyValue", (["in"], _PK, "key"), (["in"], _PK, "pValue")),
    COMMETHOD([], HRESULT, "SetBoolValue", (["in"], _PK, "key"), (["in"], c_int, "Value")),
    COMMETHOD([], HRESULT, "GetBoolValue", (["in"], _PK, "key"), (["in"], POINTER(c_int), "pValue")),
    COMMETHOD([], HRESULT, "SetIUnknownValue", (["in"], _PK, "key"), (["in"], POINTER(IUnknown), "pValue")),
    COMMETHOD([], HRESULT, "GetIUnknownValue", (["in"], _PK, "key"), (["in"], POINTER(POINTER(IUnknown)), "ppValue")),
    COMMETHOD([], HRESULT, "SetGuidValue", (["in"], _PK, "key"), (["in"], POINTER(GUID), "Value")),
    COMMETHOD([], HRESULT, "GetGuidValue", (["in"], _PK, "key"), (["in"], POINTER(GUID), "pValue")),
    COMMETHOD([], HRESULT, "SetBufferValue", (["in"], _PK, "key"), (["in"], POINTER(c_ubyte), "pValue"), (["in"], c_ulong, "cbValue")),
    COMMETHOD([], HRESULT, "GetBufferValue", (["in"], _PK, "key"), (["in"], POINTER(POINTER(c_ubyte)), "ppValue"), (["in"], POINTER(c_ulong), "pcbValue")),
    COMMETHOD([], HRESULT, "SetIPortableDeviceValuesValue", (["in"], _PK, "key"), (["in"], POINTER(IPortableDeviceValues), "pValue")),
    COMMETHOD([], HRESULT, "GetIPortableDeviceValuesValue", (["in"], _PK, "key"), (["in"], POINTER(POINTER(IPortableDeviceValues)), "ppValue")),
    COMMETHOD([], HRESULT, "SetIPortableDevicePropVariantCollectionValue", (["in"], _PK, "key"), (["in"], POINTER(IPortableDevicePropVariantCollection), "pValue")),
    COMMETHOD([], HRESULT, "GetIPortableDevicePropVariantCollectionValue", (["in"], _PK, "key"), (["in"], POINTER(POINTER(IPortableDevicePropVariantCollection)), "ppValue")),
    COMMETHOD([], HRESULT, "SetIPortableDeviceKeyCollectionValue", (["in"], _PK, "key"), (["in"], POINTER(IPortableDeviceKeyCollection), "pValue")),
    COMMETHOD([], HRESULT, "GetIPortableDeviceKeyCollectionValue", (["in"], _PK, "key"), (["in"], POINTER(POINTER(IPortableDeviceKeyCollection)), "ppValue")),
    COMMETHOD([], HRESULT, "SetIPortableDeviceValuesCollectionValue", (["in"], _PK, "key"), (["in"], POINTER(IPortableDeviceValuesCollection), "pValue")),
    COMMETHOD([], HRESULT, "GetIPortableDeviceValuesCollectionValue", (["in"], _PK, "key"), (["in"], POINTER(POINTER(IPortableDeviceValuesCollection)), "ppValue")),
    COMMETHOD([], HRESULT, "RemoveValue", (["in"], _PK, "key")),
    COMMETHOD([], HRESULT, "CopyValuesFromPropertyStore", (["in"], c_void_p, "pStore")),
    COMMETHOD([], HRESULT, "CopyValuesToPropertyStore", (["in"], c_void_p, "pStore")),
    COMMETHOD([], HRESULT, "Clear"),
]


class IPortableDeviceManager(IUnknown):
    _iid_ = GUID("{A1567595-4C2F-4574-A6FA-ECEF917B9A40}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetDevices", (["in"], POINTER(c_wchar_p), "pPnPDeviceIDs"), (["in"], POINTER(c_ulong), "pcPnPDeviceIDs")),
        COMMETHOD([], HRESULT, "RefreshDeviceList"),
        COMMETHOD([], HRESULT, "GetDeviceFriendlyName", (["in"], c_wchar_p, "pszPnPDeviceID"), (["in"], c_wchar_p, "pDeviceFriendlyName"), (["in"], POINTER(c_ulong), "pcchDeviceFriendlyName")),
        COMMETHOD([], HRESULT, "GetDeviceDescription", (["in"], c_wchar_p, "pszPnPDeviceID"), (["in"], c_wchar_p, "pDeviceDescription"), (["in"], POINTER(c_ulong), "pcchDeviceDescription")),
        COMMETHOD([], HRESULT, "GetDeviceManufacturer", (["in"], c_wchar_p, "pszPnPDeviceID"), (["in"], c_wchar_p, "pDeviceManufacturer"), (["in"], POINTER(c_ulong), "pcchDeviceManufacturer")),
        COMMETHOD([], HRESULT, "GetDeviceProperty", (["in"], c_wchar_p, "pszPnPDeviceID"), (["in"], c_wchar_p, "pszDevicePropertyName"), (["in"], POINTER(c_ubyte), "pData"), (["in"], POINTER(c_ulong), "pcbData"), (["in"], POINTER(c_ulong), "pdwType")),
        COMMETHOD([], HRESULT, "GetPrivateDevices", (["in"], POINTER(c_wchar_p), "pPnPDeviceIDs"), (["in"], POINTER(c_ulong), "pcPnPDeviceIDs")),
    ]


class IPortableDeviceProperties(IUnknown):
    _iid_ = GUID("{7F6D695C-03DF-4439-A809-59266BEEE3A6}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetSupportedProperties", (["in"], c_wchar_p, "pszObjectID"), (["in"], POINTER(POINTER(IPortableDeviceKeyCollection)), "ppKeys")),
        COMMETHOD([], HRESULT, "GetPropertyAttributes", (["in"], c_wchar_p, "pszObjectID"), (["in"], _PK, "Key"), (["in"], POINTER(POINTER(IPortableDeviceValues)), "ppAttributes")),
        COMMETHOD([], HRESULT, "GetValues", (["in"], c_wchar_p, "pszObjectID"), (["in"], POINTER(IPortableDeviceKeyCollection), "pKeys"), (["in"], POINTER(POINTER(IPortableDeviceValues)), "ppValues")),
        COMMETHOD([], HRESULT, "SetValues", (["in"], c_wchar_p, "pszObjectID"), (["in"], POINTER(IPortableDeviceValues), "pValues"), (["in"], POINTER(POINTER(IPortableDeviceValues)), "ppResults")),
        COMMETHOD([], HRESULT, "Delete", (["in"], c_wchar_p, "pszObjectID"), (["in"], POINTER(IPortableDeviceKeyCollection), "pKeys")),
    ]


class IPortableDeviceContent(IUnknown):
    _iid_ = GUID("{6A96ED84-7C73-4480-9938-BF5AF477D426}")
    _methods_ = [
        COMMETHOD([], HRESULT, "EnumObjects", (["in"], c_ulong, "dwFlags"), (["in"], c_wchar_p, "pszParentObjectID"), (["in"], c_void_p, "pFilter"), (["in"], POINTER(c_void_p), "ppEnum")),
        COMMETHOD([], HRESULT, "Properties", (["in"], POINTER(POINTER(IPortableDeviceProperties)), "ppProperties")),
    ]


class IPortableDevice(IUnknown):
    _iid_ = GUID("{625E2DF8-6392-4CF0-9AD1-3CFA5F17775C}")
    _methods_ = [
        COMMETHOD([], HRESULT, "Open", (["in"], c_wchar_p, "pszPnPDeviceID"), (["in"], POINTER(IPortableDeviceValues), "pClientInfo")),
        COMMETHOD([], HRESULT, "SendCommand", (["in"], c_ulong, "dwFlags"), (["in"], POINTER(IPortableDeviceValues), "pParameters"), (["in"], POINTER(POINTER(IPortableDeviceValues)), "ppResults")),
        COMMETHOD([], HRESULT, "Content", (["in"], POINTER(POINTER(IPortableDeviceContent)), "ppContent")),
        COMMETHOD([], HRESULT, "Capabilities", (["in"], POINTER(c_void_p), "ppCapabilities")),
        COMMETHOD([], HRESULT, "Cancel"),
        COMMETHOD([], HRESULT, "Close"),
        COMMETHOD([], HRESULT, "Advise", (["in"], c_ulong, "dwFlags"), (["in"], c_void_p, "pCallback"), (["in"], POINTER(IPortableDeviceValues), "pParameters"), (["in"], POINTER(c_wchar_p), "ppszCookie")),
        COMMETHOD([], HRESULT, "Unadvise", (["in"], c_wchar_p, "pszCookie")),
        COMMETHOD([], HRESULT, "GetPnPDeviceID", (["in"], POINTER(c_wchar_p), "ppszPnPDeviceID")),
    ]
