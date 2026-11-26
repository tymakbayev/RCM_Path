"""Host memory and disk counters recorded by the runners (A4 peak RSS, A5-A8 `rss`, A8 memory and device snapshots).

Windows: the Win32 calls of the reference runs, moved here from a5_run.py and a8_run.py without changes (the A8 device
counter now opens the volume that holds the PostgreSQL data directory instead of a hard-coded C:).
Linux: /proc equivalents. Other systems: None. A counter that cannot be read is recorded as None.
"""

from __future__ import annotations

import os
import sys

IS_WIN = os.name == "nt"

if IS_WIN:
    import ctypes
    import ctypes.wintypes as wt

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    class _MemStat(ctypes.Structure):
        _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    class _PerfInfo(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                    ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                    ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                    ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                    ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                    ("HandleCount", wt.DWORD), ("ProcessCount", wt.DWORD), ("ThreadCount", wt.DWORD)]

    class _DiskPerf(ctypes.Structure):
        _fields_ = [("BytesRead", ctypes.c_longlong), ("BytesWritten", ctypes.c_longlong),
                    ("ReadTime", ctypes.c_longlong), ("WriteTime", ctypes.c_longlong), ("IdleTime", ctypes.c_longlong),
                    ("ReadCount", wt.DWORD), ("WriteCount", wt.DWORD), ("QueueDepth", wt.DWORD),
                    ("SplitCount", wt.DWORD), ("QueryTime", ctypes.c_longlong), ("StorageDeviceNumber", wt.DWORD),
                    ("StorageManagerName", wt.WCHAR * 8)]

    _K32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _K32.CreateFileW.restype = wt.HANDLE
    _K32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, wt.LPVOID, wt.DWORD, wt.DWORD, wt.HANDLE]
    _K32.DeviceIoControl.argtypes = [wt.HANDLE, wt.DWORD, wt.LPVOID, wt.DWORD, wt.LPVOID, wt.DWORD,
                                     ctypes.POINTER(wt.DWORD), wt.LPVOID]
    _K32.CloseHandle.argtypes = [wt.HANDLE]


# ------------------------------------------------------------------ process memory
def rss_mb() -> dict | None:
    """Peak / current resident set of this process in MiB (Windows: working set via K32GetProcessMemoryInfo)."""
    if IS_WIN:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wt.HANDLE
        f = k32.K32GetProcessMemoryInfo
        f.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
        f.restype = wt.BOOL
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        if not f(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return {"peak_mb": c.PeakWorkingSetSize / 2**20, "current_mb": c.WorkingSetSize / 2**20}
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            kv = dict(line.split(":", 1) for line in fh if ":" in line)
        return {"peak_mb": int(kv["VmHWM"].split()[0]) / 1024, "current_mb": int(kv["VmRSS"].split()[0]) / 1024}
    except (OSError, KeyError, ValueError):
        pass
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"peak_mb": peak / (2**20 if sys.platform == "darwin" else 1024), "current_mb": None}
    except (ImportError, OSError):
        return None


# ------------------------------------------------------------------ system memory
def memory() -> dict:
    """Available / total physical memory and the OS file cache, GiB (None where the OS does not report it)."""
    if IS_WIN:
        m = _MemStat()
        m.dwLength = ctypes.sizeof(_MemStat)
        p = _PerfInfo()
        p.cb = ctypes.sizeof(_PerfInfo)
        ok_m = bool(_K32.GlobalMemoryStatusEx(ctypes.byref(m)))
        ok_p = bool(_K32.K32GetPerformanceInfo(ctypes.byref(p), p.cb))
        return {"avail_phys_gb": m.ullAvailPhys / 2**30 if ok_m else None,
                "total_phys_gb": m.ullTotalPhys / 2**30 if ok_m else None,
                "system_cache_gb": p.SystemCache * p.PageSize / 2**30 if ok_p else None}
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            kb = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in fh if ":" in line)}
        return {"avail_phys_gb": kb["MemAvailable"] / 2**20, "total_phys_gb": kb["MemTotal"] / 2**20,
                "system_cache_gb": kb["Cached"] / 2**20}
    except (OSError, KeyError, ValueError):
        return {"avail_phys_gb": None, "total_phys_gb": None, "system_cache_gb": None}


# ------------------------------------------------------------------ device counters of the volume holding `path`
def disk_counters(path: str) -> dict | None:
    """Cumulative reads of the volume that holds `path`: {read_count, bytes_read, read_time_100ns}; None if unavailable.

    Windows: IOCTL_DISK_PERFORMANCE on the volume (no admin rights needed on the reference stand).
    Linux: /proc/diskstats line of the device that holds `path` (sectors of 512 bytes, read time in ms)."""
    if IS_WIN:
        drive = os.path.splitdrive(os.path.abspath(path))[0] or "C:"
        h = _K32.CreateFileW("\\\\.\\" + drive, 0, 3, None, 3, 0, None)
        if h is None or h == wt.HANDLE(-1).value:
            return None
        try:
            d = _DiskPerf()
            ret = wt.DWORD()
            if not _K32.DeviceIoControl(h, 0x70020, None, 0, ctypes.byref(d), ctypes.sizeof(d), ctypes.byref(ret),
                                        None):
                return None
            return {"read_count": int(d.ReadCount), "bytes_read": int(d.BytesRead), "read_time_100ns": int(d.ReadTime)}
        finally:
            _K32.CloseHandle(h)
    try:
        dev = os.stat(path).st_dev
        major, minor = os.major(dev), os.minor(dev)
        with open("/proc/diskstats", encoding="ascii") as fh:
            for line in fh:
                f = line.split()
                if int(f[0]) == major and int(f[1]) == minor:
                    return {"read_count": int(f[3]), "bytes_read": int(f[5]) * 512, "read_time_100ns": int(f[6]) * 10_000}
    except (OSError, ValueError, IndexError):
        pass
    return None
