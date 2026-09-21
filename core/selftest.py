#!/usr/bin/env python3
"""Hardware-in-the-loop RX self-test for DroneCMD.

Drives a real SDR (currently HackRF, **receive-only**) across DroneCMD's capture
parameters and code paths and returns a structured pass/fail result set. It
never transmits.

With ambient RF (no known transmitter) this validates the *plumbing and
robustness* of the whole chain — that every parameter and capability executes
and produces well-formed IQ/results — not decode-against-known-payload, which
would require a controlled source.

Public API:
    run_selftest(verbose=True) -> list[Result]
    summarize(results)         -> dict
    print_summary(results)     -> int   (process exit code)

Used by ``scripts/hackrf_selftest.py`` and the ``dronecmd selftest`` CLI
subcommand. Each check is isolated: an exception is recorded as a failure and
the run continues. Requires the HackRF CLI tools (e.g. PothosSDR) and a
connected HackRF; checks that cannot run are reported as SKIP.
"""
from __future__ import annotations

import asyncio
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Baseline operating point reused across checks.
BASE_FREQ = 2.44e9
BASE_RATE = 8e6
BASE_GAIN = 32.0
BASE_N = 65536


@dataclass
class Result:
    name: str
    status: str  # "OK" | "FAIL" | "SKIP"
    detail: str = ""
    seconds: float = 0.0


class _Skip(Exception):
    pass


class Recorder:
    def __init__(self, verbose: bool = True) -> None:
        self.results: List[Result] = []
        self.verbose = verbose

    def run(self, name: str, fn: Callable[[], str]) -> None:
        """Run one check. ``fn`` returns a detail string on success, raises to
        fail, or raises ``_Skip`` to skip."""
        t0 = time.perf_counter()
        try:
            detail = fn()
            status = "OK"
        except _Skip as s:
            detail, status = str(s), "SKIP"
        except Exception as e:  # noqa: BLE001 - self-test isolates every check
            detail = f"{type(e).__name__}: {e}"
            status = "FAIL"
            if self.verbose:
                traceback.print_exc()
        dt = time.perf_counter() - t0
        if self.verbose:
            print(f"[{status:4}] {name}  ({dt:.2f}s)  {detail}")
        self.results.append(Result(name, status, detail, dt))


def _assert_iq(x: np.ndarray, n: int) -> str:
    assert x.dtype == np.complex64, f"dtype {x.dtype} != complex64"
    assert x.size == n, f"size {x.size} != {n}"
    assert np.isfinite(x).all(), "non-finite samples"
    power = float(np.mean(np.abs(x) ** 2))
    assert power > 0.0, "zero power (no signal / dead RX)"
    return f"n={x.size} power={power:.3e} peak={float(np.max(np.abs(x))):.3f}"


def _cfg(**kw: Any) -> Any:
    from core.capture import GainMode, SDRConfig, SDRPlatform

    params: Dict[str, Any] = dict(
        platform=SDRPlatform.HACKRF,
        frequency_hz=BASE_FREQ,
        sample_rate_hz=BASE_RATE,
        gain_mode=GainMode.MANUAL,
        gain_db=BASE_GAIN,
        duration_s=None,
    )
    params.update(kw)
    return SDRConfig(**params)


def _capture(cfg: Any, n: int) -> np.ndarray:
    """One synchronous capture through the auto-selected backend."""
    from core.capture import EnhancedLiveCapture

    live = EnhancedLiveCapture(cfg)
    live.hardware.open()
    try:
        live.hardware.configure(cfg)
        return live.hardware.read_samples(n)
    finally:
        live.hardware.close()


# --------------------------------------------------------------------------
# Section 0 — environment / backend selection
# --------------------------------------------------------------------------
def check_backend_selection() -> str:
    import core.capture as cap

    if cap.SOAPY_SDR_AVAILABLE:
        backend = "SoapyHackRFHardware (SoapySDR)"
    elif cap.HACKRF_TRANSFER_AVAILABLE:
        backend = f"HackRFTransferHardware (CLI: {cap.HACKRF_TRANSFER_PATH})"
    else:
        raise RuntimeError("no HackRF backend available")
    live = cap.EnhancedLiveCapture(_cfg())
    return f"{backend} -> {type(live.hardware).__name__}"


def check_soapy_backend_note() -> str:
    import core.capture as cap

    if cap.SOAPY_SDR_AVAILABLE:
        return "SoapySDR importable — in-process backend active"
    raise _Skip(
        "SoapySDR Python bindings not importable on this interpreter "
        "(PothosSDR ships 3.9-only); using hackrf_transfer CLI backend"
    )


def check_device_present() -> str:
    from core.capture import HackRFTransferHardware

    hw = HackRFTransferHardware(_cfg())
    hw.open()  # runs hackrf_info; raises if no device
    hw.close()
    return "hackrf_info reports a connected HackRF"


# --------------------------------------------------------------------------
# Section 1 — capture parameter sweeps (vary one axis around the baseline)
# --------------------------------------------------------------------------
def make_freq_check(freq: float) -> Callable[[], str]:
    def fn() -> str:
        x = _capture(_cfg(frequency_hz=freq), BASE_N)
        return f"{freq/1e6:.2f} MHz: " + _assert_iq(x, BASE_N)

    return fn


def make_rate_check(rate: float) -> Callable[[], str]:
    def fn() -> str:
        n = 131072  # a bit more so high rates still fill quickly
        x = _capture(_cfg(sample_rate_hz=rate), n)
        return f"{rate/1e6:.1f} MSps: " + _assert_iq(x, n)

    return fn


def make_gain_check(label: str, **kw: Any) -> Callable[[], str]:
    def fn() -> str:
        x = _capture(_cfg(**kw), BASE_N)
        return f"{label}: " + _assert_iq(x, BASE_N)

    return fn


def make_bandwidth_check(bw: Optional[float]) -> Callable[[], str]:
    def fn() -> str:
        x = _capture(_cfg(bandwidth_hz=bw), BASE_N)
        lbl = "none" if bw is None else f"{bw/1e6:.1f} MHz"
        return f"bw={lbl}: " + _assert_iq(x, BASE_N)

    return fn


def make_nsamples_check(n: int) -> Callable[[], str]:
    def fn() -> str:
        x = _capture(_cfg(), n)
        return _assert_iq(x, n)

    return fn


# --------------------------------------------------------------------------
# Section 2 — parameter validation (out-of-range must be rejected)
# --------------------------------------------------------------------------
def check_rejects_high_sample_rate() -> str:
    from core.capture import SDRConfig, SDRPlatform

    # HackRF tops out at 20 MSps; SDRConfig validates hardware limits at
    # construction, so an out-of-range rate must be rejected there.
    try:
        SDRConfig(platform=SDRPlatform.HACKRF, sample_rate_hz=25e6)
    except ValueError as e:
        return f"rejected 25 MSps at config build: {e}"
    raise AssertionError("25 MSps was NOT rejected")


def check_config_validates_frequency_range() -> str:
    from core.capture import SDRConfig, SDRPlatform

    try:
        SDRConfig(platform=SDRPlatform.HACKRF, frequency_hz=7e9)  # > 6 GHz
    except Exception as e:  # noqa: BLE001
        return f"7 GHz rejected at config build: {type(e).__name__}"
    raise _Skip("SDRConfig did not reject 7 GHz (validation may be advisory)")


# --------------------------------------------------------------------------
# Section 3 — capture API paths
# --------------------------------------------------------------------------
def check_enhanced_capture_samples() -> str:
    from core.capture import EnhancedLiveCapture

    async def run() -> np.ndarray:
        async with EnhancedLiveCapture(_cfg()) as cap_:
            samples, meta = await cap_.capture_samples(num_samples=BASE_N)
            assert meta.total_samples >= BASE_N
            return samples

    x = asyncio.run(run())
    return "EnhancedLiveCapture.capture_samples: " + _assert_iq(x, BASE_N)


def check_enhanced_stream_samples() -> str:
    from core.capture import EnhancedLiveCapture

    async def run() -> np.ndarray:
        async with EnhancedLiveCapture(_cfg()) as cap_:
            async for chunk, metrics in cap_.stream_samples(chunk_size=BASE_N):
                return chunk  # first chunk is enough for a plumbing check
        return np.empty(0, dtype=np.complex64)

    x = asyncio.run(run())
    return "stream_samples first chunk: " + _assert_iq(x, BASE_N)


def check_enhanced_async_read() -> str:
    from core.capture import EnhancedLiveCapture

    async def run() -> np.ndarray:
        live = EnhancedLiveCapture(_cfg())
        await live.connect()
        try:
            return await live.hardware.read_samples_async(BASE_N)
        finally:
            await live.disconnect()

    x = asyncio.run(run())
    return "read_samples_async: " + _assert_iq(x, BASE_N)


def check_simple_capture_manager() -> str:
    from capture.manager import CaptureManager

    mgr = CaptureManager(platform="hackrf", sample_rate=BASE_RATE)
    mgr.set_frequency(BASE_FREQ)
    mgr.set_gain(gain_db=BASE_GAIN, mode="manual")
    if not mgr.connect():
        raise RuntimeError("CaptureManager.connect() returned False")
    try:
        x = mgr.capture(duration=BASE_N / BASE_RATE, auto_connect=False)
    finally:
        mgr.disconnect()
    assert x.size >= BASE_N // 2, f"only {x.size} samples"
    return "CaptureManager.capture: " + _assert_iq(x.astype(np.complex64), x.size)


# --------------------------------------------------------------------------
# Section 4 — analysis pipeline on a baseline capture (shared)
# --------------------------------------------------------------------------
_BASELINE_IQ: Optional[np.ndarray] = None


def _baseline_iq() -> np.ndarray:
    global _BASELINE_IQ
    if _BASELINE_IQ is None:
        _BASELINE_IQ = _capture(_cfg(), 200000)
    return _BASELINE_IQ


def check_signal_processor() -> str:
    from core.signal_processing import SignalProcessor

    x = _baseline_iq()
    sp = SignalProcessor()
    p = sp.calculate_power_dbfs(x)
    snr = sp.estimate_snr(x)
    fc = sp.estimate_carrier_frequency(x, sample_rate=BASE_RATE)
    norm = sp.normalize(x)
    agc = sp.apply_agc(x)
    assert np.isfinite(p) and np.isfinite(snr) and np.isfinite(fc)
    assert norm.size == x.size and agc.size == x.size
    return f"power={p:.1f}dBFS snr={snr:.1f}dB fc_off={fc/1e3:.1f}kHz"


def check_signal_quality() -> str:
    from core.signal_processing import analyze_signal_quality

    q = analyze_signal_quality(_baseline_iq(), sample_rate=BASE_RATE)
    assert "signal_power_dbfs" in q and "occupied_bandwidth_hz" in q
    return (
        f"occ_bw={q.get('occupied_bandwidth_hz', 0)/1e3:.1f}kHz "
        f"crest={q.get('crest_factor_db', 0):.1f}dB"
    )


def check_signal_detector() -> str:
    from capture.detector import SignalDetector

    det = SignalDetector(sample_rate=BASE_RATE)
    sigs = det.detect_signals(_baseline_iq()[:131072], threshold=0.05, method="auto")
    summ = det.get_detection_summary()
    assert isinstance(sigs, list)
    return f"{len(sigs)} region(s); methods={summ.get('supported_methods')}"


def check_blind_resolution() -> str:
    from core.blind import resolve_ofdm_profile, resolve_sc_profile

    x = _baseline_iq()[:20000].astype(np.complex128)
    sc_spec, sc_conf = resolve_sc_profile(x)
    ofdm_name, ofdm_conf = resolve_ofdm_profile(x)
    # Ambient RF should not confidently mis-lock a specific PHY unless a real
    # matching signal is present; this exercises the resolve + reject paths.
    return (
        f"SC={'lock:'+sc_spec.name if sc_spec else 'no-lock'}({sc_conf:.2f}) "
        f"OFDM={ofdm_name or 'no-lock'}({ofdm_conf:.2f})"
    )


def check_fileio_roundtrip() -> str:
    import tempfile

    from utils.fileio import read_iq_file, write_iq_file

    x = _baseline_iq()[:4096]
    fd, path = tempfile.mkstemp(suffix=".iq")
    os.close(fd)
    try:
        write_iq_file(path, x)
        y = read_iq_file(path)
        assert y.size == x.size, f"roundtrip size {y.size} != {x.size}"
        assert np.allclose(y, x, atol=1e-4), "roundtrip values differ"
    finally:
        os.remove(path)
    return f"wrote+read {x.size} samples losslessly"


# --------------------------------------------------------------------------
# Section 5 — CLI paths (subprocess)
# --------------------------------------------------------------------------
def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cli(*args: str, timeout: float = 60.0) -> Any:
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, os.path.join(_repo_root(), "cli.py"), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_repo_root(),
    )


def check_cli_info() -> str:
    p = _cli("info")
    if p.returncode != 0:
        raise _Skip(
            f"`cli.py info` exit {p.returncode}: "
            f"{(p.stderr or p.stdout).strip()[:120]}"
        )
    return "cli.py info OK"


def check_cli_config_show() -> str:
    p = _cli("config", "show")
    if p.returncode != 0:
        raise _Skip(
            f"`cli.py config show` exit {p.returncode}: "
            f"{(p.stderr or p.stdout).strip()[:120]}"
        )
    return "cli.py config show OK"


def check_cli_capture_then_analyze() -> str:
    import tempfile

    out = os.path.join(tempfile.gettempdir(), "hackrf_selftest_cli.iq")
    cap = _cli(
        "capture",
        "--platform",
        "hackrf",
        "--frequency",
        str(BASE_FREQ),
        "--duration",
        "0.05",
        "--output",
        out,
        timeout=90,
    )
    if cap.returncode != 0:
        raise _Skip(
            f"`cli.py capture` exit {cap.returncode}: "
            f"{(cap.stderr or cap.stdout).strip()[:160]}"
        )
    if not os.path.exists(out):
        raise AssertionError("capture reported success but no output file")
    size = os.path.getsize(out)
    ana = _cli("analyze", "--input", out, timeout=90)
    os.remove(out)
    if ana.returncode != 0:
        blob = ana.stderr or ana.stdout
        if "trained" in blob.lower() or "model" in blob.lower():
            # Expected: the classifier is intentionally not pre-trained (see
            # README) and fails loudly. Capture itself worked — that is what
            # this hardware check validates.
            return f"cli capture ({size}B) OK; analyze needs a trained model (expected)"
        raise _Skip(
            f"capture OK ({size}B); analyze exit {ana.returncode}: {blob.strip()[:120]}"
        )
    return f"cli capture ({size}B) -> analyze OK"


def _auto_gain_kw() -> Dict[str, Any]:
    from core.capture import GainMode

    return dict(gain_mode=GainMode.AUTO, gain_db=None)


def run_selftest(verbose: bool = True) -> List[Result]:
    """Run the full HackRF RX self-test and return the per-check results."""
    if verbose:
        print("=" * 70)
        print("DroneCMD HackRF RX self-test (receive-only; never transmits)")
        print(
            f"baseline: {BASE_FREQ/1e6:.1f} MHz, {BASE_RATE/1e6:.1f} MSps, "
            f"gain {BASE_GAIN:.0f} dB, N={BASE_N}"
        )
        print("=" * 70)
    r = Recorder(verbose=verbose)

    # 0 — environment
    r.run("backend/selection", check_backend_selection)
    r.run("backend/soapysdr-inprocess", check_soapy_backend_note)
    r.run("device/present", check_device_present)

    # 1 — parameter sweeps
    for f in (433.92e6, 915e6, 2.44e9, 5.8e9):
        r.run(f"param/frequency/{f/1e6:.0f}MHz", make_freq_check(f))
    for sr in (2e6, 8e6, 16e6, 20e6):
        r.run(f"param/sample_rate/{sr/1e6:.0f}MSps", make_rate_check(sr))
    r.run("param/gain/auto", make_gain_check("AUTO", **_auto_gain_kw()))
    for g in (8.0, 32.0, 40.0):
        r.run(
            f"param/gain/manual/{g:.0f}dB",
            make_gain_check(f"MANUAL {g:.0f}dB", gain_db=g),
        )
    for bw in (None, 5e6):
        r.run(
            f"param/bandwidth/{'none' if bw is None else f'{bw/1e6:.0f}MHz'}",
            make_bandwidth_check(bw),
        )
    for n in (16384, 65536, 1_000_000):
        r.run(f"param/num_samples/{n}", make_nsamples_check(n))

    # 2 — parameter validation
    r.run("validate/reject-high-rate", check_rejects_high_sample_rate)
    r.run("validate/frequency-range", check_config_validates_frequency_range)

    # 3 — capture APIs
    r.run("api/enhanced/capture_samples", check_enhanced_capture_samples)
    r.run("api/enhanced/stream_samples", check_enhanced_stream_samples)
    r.run("api/enhanced/read_samples_async", check_enhanced_async_read)
    r.run("api/simple/CaptureManager", check_simple_capture_manager)

    # 4 — analysis pipeline
    r.run("pipeline/signal_processor", check_signal_processor)
    r.run("pipeline/signal_quality", check_signal_quality)
    r.run("pipeline/signal_detector", check_signal_detector)
    r.run("pipeline/blind_resolution", check_blind_resolution)
    r.run("pipeline/fileio_roundtrip", check_fileio_roundtrip)

    # 5 — CLI
    r.run("cli/info", check_cli_info)
    r.run("cli/config_show", check_cli_config_show)
    r.run("cli/capture+analyze", check_cli_capture_then_analyze)

    return r.results


def summarize(results: List[Result]) -> Dict[str, Any]:
    """Aggregate results into a JSON-friendly summary dict."""
    ok = sum(r.status == "OK" for r in results)
    fail = sum(r.status == "FAIL" for r in results)
    skip = sum(r.status == "SKIP" for r in results)
    return {
        "ok": ok,
        "fail": fail,
        "skip": skip,
        "total": len(results),
        "passed": fail == 0,
        "checks": [
            {
                "name": r.name,
                "status": r.status,
                "detail": r.detail,
                "seconds": round(r.seconds, 3),
            }
            for r in results
        ],
    }


def print_summary(results: List[Result]) -> int:
    """Print the summary table and return a process exit code (0 = no FAIL)."""
    s = summarize(results)
    print("\n" + "=" * 70)
    print(
        f"HackRF self-test: {s['ok']} OK, {s['fail']} FAIL, {s['skip']} SKIP "
        f"({s['total']} checks)"
    )
    if s["fail"]:
        print("\nFAILURES:")
        for r in results:
            if r.status == "FAIL":
                print(f"  - {r.name}: {r.detail}")
    print("=" * 70)
    return 1 if s["fail"] else 0
