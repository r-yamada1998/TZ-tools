"""
Shared parts of the TZ-tools scripts.

    start()     read the two configs, check them, open the instruments
    Devices     the TZ-controller instances a script uses, and the steps made
                of them (SIS bias, IF band, LO Att., spectrum analyzer traces)
    Recorder    one run directory: meta.json and traces.csv
    ask_yes     wait for the operator

Configs:
    device_config.toml   the instruments, in the format of TZ-controller (the
                         file initialize.py reads). Which file: see
                         device_config in measure_config.toml, or --device-config.
    measure_config.toml  settings only these scripts use

--dry-run puts tz_fake_hw in place of the instrument drivers, so the scripts
run through the TZ-controller classes without touching the receiver.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import pathlib
import time
import tomllib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MEASURE_CONFIG = HERE / "measure_config.toml"

# TZ-controller logs every step at INFO, which buries the prompts. DeviceBase
# configures logging only when nothing else has.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

BIAS_NAMES = ["v1", "h1", "h2", "v2"]
BANDS = ["HU", "HL", "VU", "VL"]
FACTOR_TOLERANCE = 0.01

# Set by --yes: answer every prompt with y. Meant for --dry-run tests.
AUTO_YES = False
DRY_RUN = False


def ask_yes(message: str) -> None:
    """Wait until the operator types y."""
    if AUTO_YES:
        print(f"{message} [y/n]: y")
        return
    while input(f"{message} [y/n]: ").strip().lower() != "y":
        pass


def set_load(load: str) -> None:
    """Note which load the operator put in front of the feed ("hot" or "cold").
    Only the dry-run receiver uses it."""
    if DRY_RUN:
        import tz_fake_hw
        tz_fake_hw.STATE.load = load


def add_common_args(parser) -> None:
    parser.add_argument("--dry-run", action="store_true",
                        help="fake instrument drivers; nothing is sent to the receiver")
    parser.add_argument("--yes", action="store_true",
                        help="answer y to every prompt (for --dry-run tests)")
    parser.add_argument("--keep-sa", action="store_true",
                        help="leave the spectrum analyzer settings as they are "
                             "(averaging is then assumed off)")
    parser.add_argument("--device-config", metavar="PATH",
                        help="device_config.toml to use (default: see measure_config.toml)")


# ----------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------

def load_measure_config() -> dict:
    with open(MEASURE_CONFIG, "rb") as f:
        return tomllib.load(f)


def device_config_path(meas: dict, given: str | None = None) -> pathlib.Path:
    """The device config: --device-config, else the first existing file of the list."""
    if given:
        return pathlib.Path(given).expanduser().resolve()
    for name in meas["device_config"]:
        path = (HERE / pathlib.Path(name).expanduser()).resolve()
        if path.exists():
            return path
    raise SystemExit(f"No device config found: {meas['device_config']} (from {HERE})")


def load_configs(given: str | None = None) -> tuple[pathlib.Path, dict, dict]:
    """(device config path, device config, measure config)"""
    meas = load_measure_config()
    path = device_config_path(meas, given)
    with open(path, "rb") as f:
        return path, tomllib.load(f), meas


def preflight(path: pathlib.Path, cfg: dict, meas: dict) -> None:
    """Refuse to run on a config that would bias the junctions wrongly."""
    setter = cfg["sis_bias_setter"]
    factor = setter["conversion_factor"]
    expected = meas["expected_setter_factor"]
    if abs(factor - expected) / expected > FACTOR_TOLERANCE:
        raise SystemExit(
            f"Aborted: sis_bias_setter.conversion_factor in {path} is {factor}, "
            f"expected {expected:.6f} = 1/(3*0.94). The junctions would get "
            f"{factor / expected:.2f} times the setting. Use a corrected "
            f"device config (--device-config)."
        )
    if sorted(setter["channel"]) != sorted(BIAS_NAMES):
        raise SystemExit(f"Aborted: sis_bias_setter.channel must define {BIAS_NAMES}.")
    lim = meas["bias"]
    for name, v in setter["tuned"].items():
        if not lim["min"] <= v <= lim["max"]:
            raise SystemExit(f"Aborted: tuned {name} = {v} mV is outside the bias limits.")


def check_tz_controller() -> None:
    """
    Refuse to run on a tz_controller without the parts these scripts use:
    per-channel conversion of SISReader (TZ-controller PR #24) and the analyzer,
    1st LO, IF switch and IRR test signal operations (TZ-controller issue #25).
    """
    from tz_controller.if_switch import IFSwitch
    from tz_controller.signal_generator import SignalGenerator
    from tz_controller.sis_reader import SISReader
    from tz_controller.speana_controller import SpectrumAnalyzer
    missing = [name for cls, name in (
        (SISReader, "_resolve_conversion_factors"),
        (SpectrumAnalyzer, "fresh_trace"),
        (SignalGenerator, "reconnect"),
        (IFSwitch, "query"),
    ) if not hasattr(cls, name)]
    try:
        import tz_controller.irr_signal_generator  # noqa: F401
    except ImportError:
        missing.append("irr_signal_generator")
    if missing:
        raise SystemExit(
            f"Aborted: the installed tz_controller lacks {', '.join(missing)}. "
            "Update it (uv sync) after TZ-controller PR #24 and the PR of issue #25 are merged.")


# Operating values are read from the device config, as initialize.py reads them.

def tuned_bias(cfg: dict) -> dict:
    return {name: float(cfg["sis_bias_setter"]["tuned"][name]) for name in BIAS_NAMES}


def tuned_loatt(cfg: dict, meas: dict) -> dict:
    """Operating LO Att. currents [mA] by polarization."""
    channels = cfg["local_attenuator"]["channels"]
    return {pol: float(channels[str(port)]["att_current"])
            for pol, port in meas["local_attenuator"]["port"].items()}


FREQ_UNIT_GHZ = {"GHz": 1.0, "MHz": 1e-3, "kHz": 1e-6, "Hz": 1e-9}


def tuned_lo1_ghz(cfg: dict) -> float:
    """1st LO frequency [GHz]."""
    d = cfg["signal_generator"]["defaults"]
    return float(d["freq"]) * FREQ_UNIT_GHZ[d.get("freq_unit", "GHz")]


# The LO Att. cannot be asked for its current, so the last value set from
# TZ-tools is kept in a file for tz_status.py. Values set by other programs
# (initialize.py) are not in it.
LOATT_STATE = HERE / ".loatt_last.json"


def remember_loatt(pol: str, current_ma: float) -> None:
    try:
        state = json.loads(LOATT_STATE.read_text())
    except (OSError, ValueError):
        state = {}
    state[pol] = {"mA": current_ma, "time": datetime.datetime.now().isoformat(timespec="seconds")}
    LOATT_STATE.write_text(json.dumps(state))


def last_loatt() -> dict:
    try:
        return json.loads(LOATT_STATE.read_text())
    except (OSError, ValueError):
        return {}


# ----------------------------------------------------------------------------
# Start and finish
# ----------------------------------------------------------------------------

def start(args, use: set[str]) -> "Devices":
    """Read and check the configs, then open the instruments in `use`."""
    global AUTO_YES, DRY_RUN
    AUTO_YES = args.yes
    DRY_RUN = args.dry_run
    path, cfg, meas = load_configs(args.device_config)
    preflight(path, cfg, meas)
    if args.dry_run:
        import tz_fake_hw
        tz_fake_hw.install()
    check_tz_controller()
    print(f"device config: {path}" + ("  (dry run: fake instruments)" if args.dry_run else ""))
    return Devices(path, cfg, meas, use)


def finish(recorder: "Recorder") -> None:
    """Run the analysis; a failure there must not lose the raw data."""
    recorder.meta["finished"] = datetime.datetime.now().isoformat()
    recorder.save_meta()
    print(f'\nRaw data: "{recorder.dir}"')
    try:
        import rx_performance_analysis as analysis
        analysis.analyze(recorder.dir)
    except Exception as e:
        print(f"Analysis failed ({e!r}). Run: python rx_performance_analysis.py {recorder.dir}")


# ----------------------------------------------------------------------------
# Instruments
# ----------------------------------------------------------------------------

class Devices:
    """
    The TZ-controller instances named in `use`:

        sis      SISSetter          "sis_bias_setter"
        reader   SISReader          "sis_bias_reader"
        sa       SpectrumAnalyzer   "spectrum_analyzer"
        switch   IFSwitch           "if_switch"
        loatt    LoAttenuator       "local_attenuator"
        lo1      SignalGenerator    "signal_generator"
        irr      IrrSignalGenerator "irr_signal_generator"

    The methods below are the steps the scripts repeat.
    """

    def __init__(self, path: pathlib.Path, cfg: dict, meas: dict, use: set[str]) -> None:
        self.config_file = str(path)
        self.cfg = cfg
        self.m = meas
        self.open(use)

    def open(self, use: set[str]) -> None:
        """Open the instruments in `use`, in the order below."""
        if "sis" in use:
            from tz_controller.sis_setter import SISSetter
            self.sis = SISSetter(device_name="sis_bias_setter", config_file=self.config_file)
        if "reader" in use:
            from tz_controller.sis_reader import SISReader
            self.reader = SISReader(device_name="sis_bias_reader", config_file=self.config_file)
        if "sa" in use:
            from tz_controller.speana_controller import SpectrumAnalyzer
            self.sa = SpectrumAnalyzer(device_name="spectrum_analyzer", config_file=self.config_file)
            if not self.sa.keep_open:
                raise SystemExit(
                    f"Aborted: [spectrum_analyzer] in {self.config_file} needs keep_open = true "
                    "and timeout = 15. Without them every call opens a new connection, "
                    "which the analyzer answers only after about 4 s.")
        if "switch" in use:
            from tz_controller.if_switch import IFSwitch
            self.switch = IFSwitch(device_name="if_switch", config_file=self.config_file)
        if "loatt" in use:
            from tz_controller.lo_att import LoAttenuator
            self.loatt = LoAttenuator(device_name="local_attenuator", config_file=self.config_file)
        if "lo1" in use:
            from tz_controller.signal_generator import SignalGenerator
            self.lo1 = SignalGenerator(device_name="signal_generator", config_file=self.config_file)
        if "irr" in use:
            from tz_controller.irr_signal_generator import IrrSignalGenerator
            self.irr = IrrSignalGenerator(device_name="irr_signal_generator", config_file=self.config_file)

    # --- SIS bias -------------------------------------------------------

    def set_bias(self, **values: float) -> None:
        """set_bias(v1=6.1, h1=7.2) [mV], within the limits of measure_config.toml."""
        lim = self.m["bias"]
        for name, v in values.items():
            if name not in BIAS_NAMES:
                raise KeyError(f"Unknown bias name {name!r}")
            if not lim["min"] <= v <= lim["max"]:
                raise SystemExit(f"Aborted: {name} = {v} mV is outside [{lim['min']}, {lim['max']}] mV.")
        self.sis.setup(**values)
        self.sis.run()

    def zero_bias(self) -> None:
        self.set_bias(**{name: 0.0 for name in BIAS_NAMES})

    # --- SIS monitors -----------------------------------------------------

    def setup_monitors(self, names: list[str]) -> list[str]:
        """reader.setup() for these channels, read in this order."""
        self.reader.setup(**{n: True for n in names})
        return list(names)

    def read_monitors(self, names: list[str]) -> dict:
        """reader.run(): [mV] for *_v, [uA] for the others."""
        r = self.reader.run()
        return {n: r[n] for n in names}

    def monitor_factors(self) -> dict:
        """The conversion factor of every monitor channel in the device config."""
        table = self.reader.device_config.conversion_factor
        names = list(self.reader.device_config.channel)
        if isinstance(table, dict):
            return {n: float(table[n]) for n in names}
        return {n: float(table) for n in names}

    # --- IF band ----------------------------------------------------------

    def select_band(self, band: str) -> None:
        """
        Open the path of one IF band and close it right away. The IF Switch
        Controller keeps the switch where the pulse put it; leaving the open
        drive on for one band after another switches several paths at once.
        """
        path = self.m["if_switch"]["path"][band]
        self.switch.setup(oc="open", channel=path)
        self.switch.run()
        time.sleep(0.1)
        self.switch.setup(oc="close", channel=path)
        self.switch.run()

    def close_band(self, band: str) -> None:
        self.switch.setup(oc="close", channel=self.m["if_switch"]["path"][band])
        self.switch.run()

    # --- LO Att. -----------------------------------------------------------

    def set_loatt(self, pol: str, current_ma: float) -> None:
        """pol is "H" or "V"."""
        port = self.m["local_attenuator"]["port"][pol]
        self.loatt.setup(gpib_port=port, att_current=current_ma)
        self.loatt.run()
        if not DRY_RUN:
            remember_loatt(pol, current_ma)

    # --- 1st LO ------------------------------------------------------------

    def lo1_on(self, freq_ghz: float) -> None:
        """Frequency, reference and output on; the power is not sent."""
        self.lo1.setup(freq=freq_ghz, freq_unit="GHz", power=None, output=True)
        self.lo1.run()

    def lo1_off(self) -> None:
        self.lo1.set_output_only(False)

    def disconnect_lo1(self) -> None:
        """Close the connection and leave the output as it is. The generator
        takes one connection at a time, so a script that waits must not keep it."""
        self.lo1.disconnect()

    def reconnect_lo1(self) -> None:
        self.lo1.reconnect()

    # --- IRR test signal -------------------------------------------------

    def set_irr(self, freq_ghz: float | None = None, output: bool | None = None) -> None:
        self.irr.setup(freq=freq_ghz, output=output)
        self.irr.run()

    # --- Spectrum analyzer -----------------------------------------------

    def setup_sa(self, profile: str, average_off: bool = False) -> None:
        """
        profile is "chopper", "slider" or "irr" (measure_config.toml).
        average_off turns the analyzer's own averaging off, for new_sweeps().
        """
        params = dict(self.m["spectrum_analyzer"][profile])
        if average_off:
            params["average_onoff"] = "OFF"
        self._averaging = str(params.get("average_onoff", "OFF")).upper() == "ON"
        self._average_num = int(params.get("average_num", 1))
        self.sa.setup(**params)

    def fresh_trace(self) -> np.ndarray:
        """
        A trace with nothing from before the last change (IF band, load, test
        signal). With averaging on, the average is restarted and the trace
        read after average_num sweeps (TZ manual: confirm the averaging has
        completed). Call it after the settling wait.
        """
        if getattr(self, "_averaging", False):
            print(f"waiting for {self._average_num} averaged sweeps")
            return np.array([float(x) for x in self.sa.fresh_trace(self._average_num)])
        return self.trace()

    def new_sweeps(self, n: int, warmup: float = 2.0, min_gap: float = 1.0, poll: float = 0.05) -> list:
        """
        n traces from n separate sweeps, none of which began before this call,
        for averaging on the PC (irr_measure --soft-average). Needs the
        analyzer's averaging off (setup_sa(..., average_off=True)).

        The trace is read over and over and a change in its content marks a
        finished sweep. While traces are read back to back the content
        changes every 1.3-1.6 s, and in the first second after a pause faster
        than one sweep (timing_test, 2026-10-22). So everything read during
        `warmup` is dropped, and a change counts only `min_gap` after the
        previous one.
        Returns [(seconds since the call, trace), ...].
        """
        sweep = self.sa.sweep_time()
        timeout = 3 * max(sweep, min_gap) + 5.0
        t0 = time.monotonic()
        last = self.trace()
        while time.monotonic() - t0 < warmup:
            time.sleep(poll)
            last = self.trace()
        t_change = time.monotonic()
        out = []
        while len(out) < n:
            time.sleep(poll)
            tr = self.trace()
            now = time.monotonic()
            if np.array_equal(tr, last):
                if now - t_change > timeout:
                    raise RuntimeError(
                        f"No new sweep for {timeout:.0f} s (sweep time {sweep} s). "
                        "Is the analyzer sweeping continuously?")
                continue
            last = tr
            if now - t_change < min_gap:
                continue
            t_change = now
            out.append((round(now - t0, 3), tr))
        return out

    def set_continuous(self, on: bool) -> None:
        """Leave it on at the end of a run (the usual state)."""
        self.sa.set_continuous(on)

    def single_trace(self) -> np.ndarray:
        """One sweep started by this call. Needs set_continuous(False)."""
        return np.array([float(x) for x in self.sa.single_trace()])

    def sa_params(self) -> dict:
        try:
            return self.sa.get_current_param()
        except Exception as e:
            return {"error": repr(e)}

    def trace(self) -> np.ndarray:
        return np.array([float(x) for x in self.sa.run()])


# ----------------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------------

class Recorder:
    """
    One run directory per execution:
        <kind>_<YYMMDD_HHMMSS>/meta.json    conditions and settings
        <kind>_<YYMMDD_HHMMSS>/traces.csv   one row per trace; columns are
                                            the row's conditions, then
                                            p000, p001, ... the trace in dBm
    """

    def __init__(self, kind: str, meta: dict, columns: list[str], out_root=".") -> None:
        stamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.dir = pathlib.Path(out_root) / f"{kind}_{stamp}"
        self.dir.mkdir(parents=True)
        self.meta = {"kind": kind, "started": datetime.datetime.now().isoformat(), **meta}
        self.columns = ["timestamp"] + columns
        self._header_written = False
        self.save_meta()

    def save_meta(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False))

    def add(self, trace: np.ndarray, **info) -> None:
        path = self.dir / "traces.csv"
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if not self._header_written:
                w.writerow(self.columns + [f"p{i:03d}" for i in range(len(trace))])
                self._header_written = True
            row = [datetime.datetime.now().isoformat()] + [info[c] for c in self.columns[1:]]
            w.writerow(row + [f"{x:.4f}" for x in trace])


def add_rsky_arg(parser) -> None:
    """For the chopper scripts: the cold side of the chopper is the sky."""
    parser.add_argument("--rsky", action="store_true",
                        help="cold load is the sky (R/sky): report Tsys(R/sky) = Thot/(Y-1) "
                             "instead of Trx, which needs a 77 K load")


def base_meta(dev: Devices, args=None) -> dict:
    """Conditions common to every run, written to meta.json."""
    setter = dev.cfg["sis_bias_setter"]
    return {
        "arguments": vars(args) if args is not None else None,
        # Read by the analysis: "sky" gives Tsys(R/sky), "ln2" gives Trx
        "cold_load": "sky" if getattr(args, "rsky", False) else "ln2",
        "device_config": dev.config_file,
        "measure_config": str(MEASURE_CONFIG),
        "dry_run": DRY_RUN,
        "thot": dev.m["loads"]["thot"],
        "tcold": dev.m["loads"]["tcold"],
        "sis_channel": dict(setter["channel"]),
        "sis_conversion_factor": setter["conversion_factor"],
        "if_switch_path": dict(dev.m["if_switch"]["path"]),
        "loatt_port": dict(dev.m["local_attenuator"]["port"]),
        # Last LO Att. currents set from TZ-tools (not read back; see last_loatt)
        "loatt_last": last_loatt(),
    }
