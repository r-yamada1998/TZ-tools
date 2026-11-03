"""
Stand-ins for the instrument drivers, for --dry-run.

install() puts fake pyinterface and ogameasure modules into sys.modules before
tz_controller imports them, so the TZ-controller classes run unchanged while
nothing reaches the receiver. Every driver call is logged, and time.sleep
advances a simulated clock instead of waiting.

The fakes answer with numbers of the right shape (an IV-like monitor reading,
a spectrum trace that depends on the bias, LO Att., IF band and load), so the
scripts and the analysis run end to end. The numbers mean nothing physically.

    import tz_fake_hw
    tz_fake_hw.install(log_path="calls.log")   # before importing tz_controller
    tz_fake_hw.STATE.load = "hot"              # what the operator put in front of the feed
"""

from __future__ import annotations

import sys
import time
import types

import numpy as np

# Wiring of the receiver, as in device_config.toml and measure_config.toml
DA_MIXER = {5: "v1", 6: "h1", 7: "h2", 8: "v2"}
AD_MONITOR = {9: ("v1", "i"), 10: ("v1", "v"), 11: ("h1", "i"), 12: ("h1", "v"),
              13: ("h2", "i"), 14: ("h2", "v"), 15: ("v2", "i"), 16: ("v2", "v")}
PATH_BAND = {105: "HU", 106: "HL", 107: "VU", 108: "VL"}
PORT_POL = {4: "H", 5: "V"}
BOX_GAIN = 3 * 0.94       # junction mV per DA volt
GAP_MV = {"v": 6.85, "h": 8.1}
TRACE_POINTS = 461


class _State:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.da_volts = {}          # DA ch -> output [V]
        self.band = None            # last IF path opened
        self.loatt = {}             # "H"/"V" -> mA
        self.irr = {"freq": None, "output": False}
        self.lo1 = {"freq_hz": None, "output": False}
        self.load = None            # "hot", "cold" or None (chopper)
        self.sweep_time = 0.3
        self.rng = np.random.default_rng(0)

    def bias_mv(self, mixer: str) -> float:
        for ch, name in DA_MIXER.items():
            if name == mixer:
                return self.da_volts.get(ch, 0.0) * BOX_GAIN
        return 0.0


STATE = _State()
LOG: list[str] = []
_log_file = None


class _Clock:
    t = 1.0e6

    @classmethod
    def sleep(cls, s) -> None:
        log(f"sleep({_fmt(s)})")
        cls.t += max(float(s), 0.0)

    @classmethod
    def now(cls) -> float:
        return cls.t


def _fmt(x) -> str:
    if isinstance(x, (float, np.floating)):
        return repr(round(float(x), 6))
    if isinstance(x, (list, tuple)):
        inner = ", ".join(_fmt(v) for v in x)
        return f"[{inner}]" if isinstance(x, list) else f"({inner})"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {_fmt(v)}" for k, v in x.items()) + "}"
    return repr(x)


def log(line: str) -> None:
    LOG.append(line)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()


def _call(dev: str, method: str, args, kwargs) -> None:
    parts = [_fmt(a) for a in args] + [f"{k}={_fmt(v)}" for k, v in kwargs.items()]
    log(f"{dev}.{method}({', '.join(parts)})")


class _Fake:
    """Logs every method call; methods not defined below return None."""

    _name = "device"

    def __getattr__(self, method):
        if method.startswith("__"):
            raise AttributeError(method)

        def f(*args, **kwargs):
            _call(self._name, method, args, kwargs)
            return 0.0 if method.endswith("_query") else None
        return f

    def __setattr__(self, key, value):
        if not key.startswith("_") and not callable(value):
            log(f"{self._name}.{key} = {_fmt(value)}")
        object.__setattr__(self, key, value)

    def _log(self, method, *args, **kwargs):
        _call(self._name, method, args, kwargs)


# --- pyinterface ---------------------------------------------------------

class _DA(_Fake):
    def __init__(self, rsw_id):
        object.__setattr__(self, "_name", f"DA3346[{rsw_id}]")

    def output_da(self, smpl_ch_req, data):
        self._log("output_da", [r["ch_no"] for r in smpl_ch_req],
                  [r["range"] for r in smpl_ch_req], list(data))
        for r, v in zip(smpl_ch_req, data):
            STATE.da_volts[int(r["ch_no"])] = float(v)


def _iv_current(v_mv: float, gap: float) -> float:
    a = abs(v_mv)
    i = 4.0 * a if a < gap else 4.0 * gap + 120.0 * (a - gap)
    return float(np.sign(v_mv) * i)


class _AD(_Fake):
    def __init__(self, rsw_id):
        object.__setattr__(self, "_name", f"AD3177[{rsw_id}]")
        object.__setattr__(self, "_chs", [])
        object.__setattr__(self, "_count", 0)

    def set_sampling_config(self, smpl_ch_req, smpl_num, smpl_freq, single_diff, trig_mode):
        self._log("set_sampling_config", [r["ch_no"] for r in smpl_ch_req],
                  [r["range"] for r in smpl_ch_req], smpl_num, smpl_freq, single_diff, trig_mode)
        object.__setattr__(self, "_chs", [int(r["ch_no"]) for r in smpl_ch_req])
        object.__setattr__(self, "_count", int(smpl_num))

    def get_status(self):
        self._log("get_status")
        _Clock.t += 0.015     # smpl_num samples at smpl_freq
        return {"smpl_count": self._count}

    def _raw(self, ch: int) -> float:
        noise = float(STATE.rng.normal(0, 2e-4))
        if ch not in AD_MONITOR:
            return noise
        mixer, kind = AD_MONITOR[ch]
        v = STATE.bias_mv(mixer)
        if kind == "v":
            return v / -5.0 + noise
        return _iv_current(v, GAP_MV[mixer[0]]) / -500.0 + noise

    def read_sampling_buffer(self, num, offset):
        self._log("read_sampling_buffer", num, offset)
        return [[self._raw(ch) for ch in self._chs] for _ in range(num)]


def _pyinterface_open(model, rsw_id):
    log(f"pyinterface.open({model}, {rsw_id})")
    if int(model) == 3346:
        return _DA(rsw_id)
    if int(model) == 3177:
        return _AD(rsw_id)
    raise ValueError(f"no fake for board {model}")


# --- ogameasure ------------------------------------------------------------

class _Com(_Fake):
    def __init__(self, kind, host, port, timeout=3):
        object.__setattr__(self, "_name", f"{kind}[{host}:{port}]")
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "connection", False)
        log(f"{self._name} created (timeout={timeout})")

    def open(self):
        self._log("open")

    def close(self):
        self._log("close")

    def send(self, msg):
        self._log("send", msg)

    def readline(self):
        self._log("readline")
        return "1\n"


def _ethernet(host, port, timeout=3, **kwargs):
    return _Com("eth", host, port, timeout)


def _gpib(host, port, **kwargs):
    return _Com("gpib", host, port)


class _N9343C(_Fake):
    def __init__(self, com):
        object.__setattr__(self, "_name", "N9343C")
        object.__setattr__(self, "com", com)

    def sweep_time_set(self, s):
        self._log("sweep_time_set", s)
        STATE.sweep_time = float(s)

    def sweep_time_query(self):
        self._log("sweep_time_query")
        return STATE.sweep_time

    def scaletype_query(self):
        self._log("scaletype_query")
        return "LOG"

    def trace_data_query(self):
        self._log("trace_data_query")
        _Clock.t += 0.11      # one read of the trace takes about this long
        return [f"{x:.4f}" for x in _trace()]


class _Switch(_Fake):
    def __init__(self, com):
        object.__setattr__(self, "_name", "11713B")
        object.__setattr__(self, "com", com)

    def switch_open(self, ch):
        self._log("switch_open", ch)
        if ch in PATH_BAND:
            STATE.band = PATH_BAND[ch]

    def switch_close(self, ch):
        self._log("switch_close", ch)


class _LoAtt(_Fake):
    def __init__(self, com):
        port = com._name.rsplit(":", 1)[1].rstrip("]")
        object.__setattr__(self, "_name", f"GPDVC15[{port}]")
        object.__setattr__(self, "_port", int(port))
        object.__setattr__(self, "com", com)

    def output_set(self, ma):
        self._log("output_set", ma)
        STATE.loatt[PORT_POL.get(self._port, str(self._port))] = float(ma)


class _APSYN420(_Fake):
    def __init__(self, com):
        object.__setattr__(self, "_name", "APSYN420")
        object.__setattr__(self, "com", com)

    def freq_set(self, f, unit="GHz"):
        self._log("freq_set", f, unit)
        scale = {"GHz": 1e9, "MHz": 1e6, "kHz": 1e3, "Hz": 1.0}[unit]
        STATE.lo1["freq_hz"] = round(float(f) * scale, 3)

    def output_on(self):
        self._log("output_on")
        STATE.lo1["output"] = True

    def output_off(self):
        self._log("output_off")
        STATE.lo1["output"] = False

    def freq_query(self):
        self._log("freq_query")
        return STATE.lo1["freq_hz"] or 0.0

    def output_query(self):
        self._log("output_query")
        return 1 if STATE.lo1["output"] else 0

    def ref_source_query(self):
        self._log("ref_source_query")
        return "EXT"

    def ref_output_query(self):
        self._log("ref_output_query")
        return 1

    def ref_locked_query(self):
        self._log("ref_locked_query")
        return 1

    def close(self):
        self._log("close")


class _FSW0020(_Fake):
    def __init__(self, com):
        object.__setattr__(self, "_name", "FSW0020")
        object.__setattr__(self, "com", com)

    def freq_set(self, f, unit="GHz"):
        self._log("freq_set", f, unit)
        STATE.irr["freq"] = float(f)

    def output_set(self, on):
        self._log("output_set", on)
        STATE.irr["output"] = bool(on)


def _trace() -> np.ndarray:
    """A 0-span or spectrum trace in dBm, from the fake receiver state."""
    pol = (STATE.band or "HL")[0]
    pair = ("v1", "v2") if pol == "V" else ("h1", "h2")
    x1, x2 = STATE.bias_mv(pair[0]), STATE.bias_mv(pair[1])
    att = STATE.loatt.get(pol, 23.0)
    trx = 80 + 60 * ((x1 - 6.5) ** 2 + (x2 - 7.0) ** 2) + 0.5 * (att - 23) ** 2
    noise = STATE.rng.normal(0, 0.05, TRACE_POINTS)

    def level(t_load):
        return 10 * np.log10(t_load + trx) - 100 + noise

    if STATE.irr["output"]:
        tr = level(300.0)
        band = STATE.band or "HU"
        signal_usb = (STATE.irr["freq"] or 0) > 15.609
        tr[TRACE_POINTS // 2] += 40 if (band[1] == "U") == signal_usb else 25
        return tr
    if STATE.load == "hot":
        return level(300.0)
    if STATE.load == "cold":
        return level(77.0)
    phase = (np.arange(TRACE_POINTS) // 40) % 2 == 0
    return np.where(phase, level(300.0), level(77.0))


def _module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    m.__fake__ = True
    return m


def install(log_path=None, fast_clock: bool = True) -> None:
    """Replace the drivers. Must run before tz_controller is imported."""
    global _log_file
    for real in ("pyinterface", "ogameasure", "tz_controller"):
        mod = sys.modules.get(real)
        if mod is not None and not getattr(mod, "__fake__", False) and real != "tz_controller":
            raise RuntimeError(f"{real} is already imported; the fake must be installed first")
        if real == "tz_controller" and any(k.startswith("tz_controller.") for k in sys.modules):
            raise RuntimeError("tz_controller is already imported; the fake must be installed first")

    sys.modules["pyinterface"] = _module("pyinterface", open=_pyinterface_open)
    sys.modules["ogameasure"] = _module(
        "ogameasure",
        ethernet=_ethernet,
        gpib_prologix=_gpib,
        ELVA1=types.SimpleNamespace(GPDVC15=types.SimpleNamespace(GPDVC15_100=_LoAtt)),
        Keysight=types.SimpleNamespace(N9343C=_N9343C),
        Agilent=types.SimpleNamespace(agilent_11713B=_Switch),
        AnaPico=types.SimpleNamespace(APSYN420=_APSYN420),
        Phasematrix=types.SimpleNamespace(FSW0020=_FSW0020),
    )
    # tz_controller reads its config with tomli; on Python 3.11+ tomllib is the same
    if "tomli" not in sys.modules:
        try:
            import tomli  # noqa: F401
        except ImportError:
            import tomllib
            sys.modules["tomli"] = tomllib

    if log_path is not None:
        _log_file = open(log_path, "w")
    if fast_clock:
        time.sleep = _Clock.sleep
        time.monotonic = _Clock.now
        time.perf_counter = _Clock.now
        time.time = _Clock.now

