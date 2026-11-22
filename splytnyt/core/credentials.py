"""Secure local credential storage backed by Windows Credential Manager.

Stores optional user API keys (such as setlist.fm) in the OS credential vault
so personal keys are never written to project files, Git, source code, or
packaged executables.

On Windows this uses ``advapi32.dll`` (``CredReadW``, ``CredWriteW``,
``CredDeleteW``) directly via ``ctypes`` — zero extra third-party dependencies,
works identically from source and from PyInstaller builds.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Optional

SERVICE_PREFIX = "SplytNyt"

KEY_SETLISTFM = "setlistfm_api_key"
KEY_ACOUSTID = "acoustid_api_key"


def mask_key(key: str) -> str:
    """Return a safe, masked representation of an API key for display or logs."""
    cleaned = (key or "").strip()
    if not cleaned:
        return "(not set)"
    if len(cleaned) <= 6:
        return "*" * len(cleaned)
    return f"{cleaned[:3]}...{cleaned[-3:]}"


def _target_name(name: str) -> str:
    return f"{SERVICE_PREFIX}/{name}"


# ----------------------------------------------------------------------
# Windows Credential Manager (advapi32.dll)
# ----------------------------------------------------------------------

if sys.platform == "win32":
    CRED_TYPE_GENERIC = 1
    CRED_PERSIST_LOCAL_MACHINE = 2

    class _FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", wintypes.DWORD),
            ("dwHighDateTime", wintypes.DWORD),
        ]

    class _CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", _FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    _PCREDENTIALW = ctypes.POINTER(_CREDENTIALW)
    _advapi32 = ctypes.windll.advapi32

    _advapi32.CredReadW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_PCREDENTIALW),
    ]
    _advapi32.CredReadW.restype = wintypes.BOOL

    _advapi32.CredWriteW.argtypes = [_PCREDENTIALW, wintypes.DWORD]
    _advapi32.CredWriteW.restype = wintypes.BOOL

    _advapi32.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    _advapi32.CredDeleteW.restype = wintypes.BOOL

    _advapi32.CredFree.argtypes = [ctypes.c_void_p]
    _advapi32.CredFree.restype = None


def backend_name() -> str:
    """Human-readable name of the active credential store."""
    if sys.platform == "win32":
        return "Windows Credential Manager"
    return "Local process memory"


_FALLBACK_STORE: dict[str, str] = {}


def get_credential(name: str) -> str:
    """Read a stored credential by logical key name, or return ``""``."""
    target = _target_name(name)
    if sys.platform != "win32":
        return _FALLBACK_STORE.get(target, "")

    pcred = _PCREDENTIALW()
    ok = _advapi32.CredReadW(target, CRED_TYPE_GENERIC, 0, ctypes.byref(pcred))
    if not ok:
        return ""
    try:
        cred = pcred.contents
        size = int(cred.CredentialBlobSize)
        if size <= 0 or not cred.CredentialBlob:
            return ""
        raw = ctypes.string_at(cred.CredentialBlob, size)
        try:
            return raw.decode("utf-16-le").strip()
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="ignore").strip()
    finally:
        _advapi32.CredFree(pcred)


def set_credential(name: str, secret: str, username: str = "SplytNyt") -> bool:
    """Save or update a credential in Windows Credential Manager.

    Passing an empty string deletes any stored credential for ``name``.
    """
    cleaned = (secret or "").strip()
    if not cleaned:
        return delete_credential(name)

    target = _target_name(name)
    if sys.platform != "win32":
        _FALLBACK_STORE[target] = cleaned
        return True

    blob = cleaned.encode("utf-16-le")
    buf = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
    cred = _CREDENTIALW()
    cred.Flags = 0
    cred.Type = CRED_TYPE_GENERIC
    cred.TargetName = target
    cred.Comment = "SplytNyt music service API key"
    cred.CredentialBlobSize = len(blob)
    cred.CredentialBlob = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte))
    cred.Persist = CRED_PERSIST_LOCAL_MACHINE
    cred.AttributeCount = 0
    cred.Attributes = None
    cred.TargetAlias = None
    cred.UserName = username

    return bool(_advapi32.CredWriteW(ctypes.byref(cred), 0))


def delete_credential(name: str) -> bool:
    """Remove a stored credential if it exists."""
    target = _target_name(name)
    if sys.platform != "win32":
        _FALLBACK_STORE.pop(target, None)
        return True

    ok = _advapi32.CredDeleteW(target, CRED_TYPE_GENERIC, 0)
    if ok:
        return True
    # ERROR_NOT_FOUND (1168) is not an error when clearing a key.
    err = ctypes.GetLastError()
    return err == 1168


def has_credential(name: str) -> bool:
    """Return True if a non-empty credential is stored for ``name``."""
    return bool(get_credential(name))
