from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path

from loom.contracts import LoomStorageError


def user_sid() -> str:
    if sys.platform != "win32":
        raise LoomStorageError("unsupported_security_platform")
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    advapi.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    token = wintypes.HANDLE()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
        raise LoomStorageError("file_security_unavailable")
    try:
        size = wintypes.DWORD()
        advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not advapi.GetTokenInformation(token, 1, buffer, size, ctypes.byref(size)):
            raise LoomStorageError("file_security_unavailable")
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        text = wintypes.LPWSTR()
        if not advapi.ConvertSidToStringSidW(sid, ctypes.byref(text)):
            raise LoomStorageError("file_security_unavailable")
        try:
            if text.value is None:
                raise LoomStorageError("file_security_unavailable")
            return text.value
        finally:
            kernel.LocalFree(text)
    finally:
        kernel.CloseHandle(token)


@contextmanager
def descriptor() -> Iterator[ctypes.c_void_p]:
    if sys.platform != "win32":
        raise LoomStorageError("unsupported_security_platform")
    sid = user_sid()
    sddl = f"O:{sid}D:P(A;;FA;;;SY)(A;;FA;;;{sid})"
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    value = ctypes.c_void_p()
    if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl, 1, ctypes.byref(value), None
    ):
        raise LoomStorageError("file_security_unavailable")
    try:
        yield value
    finally:
        kernel.LocalFree(value)


class SecurityAttributes(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", wintypes.BOOL),
    ]


def create(path: Path, *, directory: bool = False, exclusive: bool = False) -> int | None:
    if sys.platform != "win32":
        raise LoomStorageError("unsupported_security_platform")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateDirectoryW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(SecurityAttributes)]
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(SecurityAttributes),
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    with descriptor() as security:
        attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), security, False)
        if directory:
            if not kernel.CreateDirectoryW(str(path), ctypes.byref(attributes)):
                if ctypes.get_last_error() == 183:
                    raise FileExistsError()
                raise LoomStorageError("file_security_unavailable")
            return None
        handle = kernel.CreateFileW(
            str(path),
            0x40000000,
            7,
            ctypes.byref(attributes),
            1 if exclusive else 4,
            0x00200080,
            None,
        )
        if handle == ctypes.c_void_p(-1).value:
            if ctypes.get_last_error() in (80, 183):
                raise FileExistsError()
            raise LoomStorageError("file_security_unavailable")
    import msvcrt

    try:
        return msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_APPEND | os.O_BINARY)
    except OSError:
        kernel.CloseHandle(handle)
        raise LoomStorageError("file_security_unavailable") from None


def check(path: Path) -> None:
    if sys.platform != "win32":
        raise LoomStorageError("unsupported_security_platform")
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    advapi.GetFileSecurityW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(wintypes.DWORD),
    ]
    size = wintypes.DWORD()
    advapi.GetFileSecurityW(str(path), 5, None, 0, ctypes.byref(size))
    actual = ctypes.create_string_buffer(size.value)
    if not advapi.GetFileSecurityW(str(path), 5, actual, size, ctypes.byref(size)):
        raise LoomStorageError("file_security_unavailable")

    def serialize(value: object) -> str:
        text = wintypes.LPWSTR()
        if not advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW(
            value, 1, 5, ctypes.byref(text), None
        ):
            raise LoomStorageError("file_security_unavailable")
        try:
            if text.value is None:
                raise LoomStorageError("file_security_unavailable")
            return text.value
        finally:
            kernel.LocalFree(text)

    with descriptor() as expected:
        if serialize(actual) != serialize(expected):
            raise LoomStorageError("insecure_permissions")
