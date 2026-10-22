"""
Shared parts of the measurement scripts ported from the ROS system.

    Rig        the instruments, opened from device_config_ros.toml
    Recorder   writes a run directory: meta.json and traces.csv
    ask_yes    the y/n prompt the ROS scripts used before each step

The ROS system recorded every topic into necstdb through its logger. Here only
what the analysis needs is kept, in plain CSV: one row per spectrum analyzer
trace, with the conditions of that trace in the leading columns.

--dry-run replaces the instruments with fakes, so that the scripts and the
analysis can be exercised without the receiver.
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
CONFIG_FILE = str(HERE / "device_config_ros.toml")

# Per-step INFO lines from tz_controller bury the prompts. Calling basicConfig
# first keeps them quiet, since DeviceBase skips its own call once a handler
# exists.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

BIAS_NAMES = ["v1", "h1", "h2", "v2"]
BANDS = ["HU", "HL", "VU", "VL"]

# The ROS writer: DA output = command / (3 * 0.94). See device_config_ros.toml.
EXPECTED_SETTER_FACTOR = 1 / (3 * 0.94)
FACTOR_TOLERANCE = 0.01


# Set by --yes: answer every prompt with y. Meant for --dry-run tests.
AUTO_YES = False


def ask_yes(message: str) -> None:
    """Wait until the operator types y, as the ROS scripts did."""
    if AUTO_YES:
        print(f"{message} [y/n]: y")
        return
    while input(f"{message} [y/n]: ").strip().lower() != "y":
        pass


def add_common_args(parser) -> None:
    parser.add_argument("--dry-run", action="store_true",
                        help="use fake instruments; nothing is sent to the receiver")
    parser.add_argument("--yes", action="store_true",
                        help="answer y to every prompt (for --dry-run tests)")
    parser.add_argument("--keep-sa", action="store_true",
                        help="leave the spectrum analyzer settings as they are "
                             "(averaging is then assumed off)")


def start(args, use: set[str]) -> "Rig":
    """Parse the common options, check the config, open the instruments."""
    global AUTO_YES
    AUTO_YES = args.yes
    cfg = load_config()
    preflight(cfg)
    return Rig(cfg, use, dry_run=args.dry_run)


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


def load_config() -> dict:
    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


def preflight(cfg: dict) -> None:
    """Refuse to run on a config that would bias the junctions wrongly."""
    setter = cfg["sis_bias_setter"]
    factor = setter["conversion_factor"]
    rel = abs(factor - EXPECTED_SETTER_FACTOR) / EXPECTED_SETTER_FACTOR
    if rel > FACTOR_TOLERANCE:
        raise SystemExit(
            f"Aborted: sis_bias_setter.conversion_factor is {factor}, expected "
            f"{EXPECTED_SETTER_FACTOR:.6f} (the ROS writer). The bias would be "
            f"{factor / EXPECTED_SETTER_FACTOR:.2f}x the intended value."
        )
    if sorted(setter["channel"]) != sorted(BIAS_NAMES):
        raise SystemExit(f"Aborted: sis_bias_setter.channel must define {BIAS_NAMES}.")
    m = cfg["measurement"]
    for name, v in setter["tuned"].items():
        if not m["bias_min"] <= v <= m["bias_max"]:
            raise SystemExit(f"Aborted: tuned {name} = {v} mV is outside the bias limits.")


# ----------------------------------------------------------------------------
# Instruments
# ----------------------------------------------------------------------------

class Rig:
    """
    The instruments a ported script needs, behind the calls the ROS scripts
    made (sis.set_v1_v, switch.set_open_ch, loatt.set_cur, irr.set_sg_freq ...).

    Only the instruments named in `use` are opened.
    """

    def __init__(self, cfg: dict, use: set[str], dry_run: bool = False) -> None:
        self.cfg = cfg
        self.m = cfg["measurement"]
        self.dry_run = dry_run
        self.bias = {name: None for name in BIAS_NAMES}
        self.att = {"H": None, "V": None}
        self.irr_on = False
        self.irr_freq = None
        self.band = None
        # Which load is in front of the feed, as told by the operator.
        # Only the dry-run fake reads it.
        self.load = None

        if dry_run:
            self._fake = _FakeReceiver(self)
            return

        if "sis" in use:
            from tz_controller.sis_setter import SISSetter
            self.sis = SISSetter(device_name="sis_bias_setter", config_file=CONFIG_FILE)
        if "sa" in use:
            from tz_controller.speana_controller import SpectrumAnalyzer
            self.sa = SpectrumAnalyzer(device_name="spectrum_analyzer", config_file=CONFIG_FILE)
            if not dry_run:
                self._keep_sa_open()
        if "switch" in use:
            from tz_controller.if_switch import IFSwitch
            self.switch = IFSwitch(device_name="if_switch", config_file=CONFIG_FILE)
        if "loatt" in use:
            from tz_controller.lo_att import LoAttenuator
            self.loatt = LoAttenuator(device_name="local_attenuator", config_file=CONFIG_FILE)
        if "irr" in use:
            import ogameasure
            c = cfg["irr_signal_generator"]
            self._irr_com = ogameasure.ethernet(c["host"], c["port"])
            self._irr_com.open()
            self.irr = ogameasure.Phasematrix.FSW0020(self._irr_com)
            self.irr.use_external_reference_source()

    # --- SIS bias -------------------------------------------------------

    def set_bias(self, **values: float) -> None:
        """set_bias(v1=6.4, h1=7.3): names as in device_config_ros.toml."""
        for name, v in values.items():
            if name not in BIAS_NAMES:
                raise KeyError(f"Unknown bias name {name!r}")
            if not self.m["bias_min"] <= v <= self.m["bias_max"]:
                raise SystemExit(
                    f"Aborted: {name} = {v} mV is outside "
                    f"[{self.m['bias_min']}, {self.m['bias_max']}] mV."
                )
        if not self.dry_run:
            self.sis.setup(**values)
            self.sis.run()
        self.bias.update(values)

    def zero_bias(self) -> None:
        self.set_bias(**{name: 0.0 for name in BIAS_NAMES})

    # --- IF switch ------------------------------------------------------

    def open_band(self, band: str) -> None:
        """ROS switch.set_open_ch(band)."""
        if not self.dry_run:
            # IFSwitch.open() is fine; called through .sw to mirror close_band.
            self.switch.sw.switch_open(self.m["if_switch_path"][band])
        self.band = band

    def close_band(self, band: str) -> None:
        """
        ROS switch.set_close_ch(band). IFSwitch.close() in tz_controller sends
        switch_open by mistake, so the driver is called directly.
        """
        if not self.dry_run:
            self.switch.sw.switch_close(self.m["if_switch_path"][band])

    # --- LO attenuator --------------------------------------------------

    def set_loatt(self, pol: str, current_ma: float) -> None:
        """pol is "H" or "V"; the port comes from measurement.loatt_port."""
        if not self.dry_run:
            port = self.m["loatt_port"][pol]
            self.loatt.setup(gpib_port=port, att_current=current_ma)
            self.loatt.run()
        self.att[pol] = current_ma

    # --- IRR test signal --------------------------------------------------

    def set_irr_freq(self, freq_ghz: float) -> None:
        if not self.dry_run:
            self.irr.freq_set(freq_ghz, "GHz")
        self.irr_freq = freq_ghz

    def set_irr_output(self, on: bool) -> None:
        if not self.dry_run:
            self.irr.output_set(1 if on else 0)
        self.irr_on = on

    def close_irr(self) -> None:
        if not self.dry_run and hasattr(self, "_irr_com"):
            self._irr_com.close()

    # --- Spectrum analyzer -----------------------------------------------

    def _keep_sa_open(self) -> None:
        """
        On 2026-10-22 (sa_check.py) the analyzer answered *IDN? at once, but
        other queries sent on a new connection only after about 4 s, longer
        than ogameasure's 3 s timeout. tz_controller opens and closes the
        connection around every call; the ROS node kept one open and read a
        trace every 0.11 s. So one connection is opened here and kept for the
        whole run (close() does nothing), with a longer timeout as a margin.
        """
        com = self.sa.com
        com.timeout = 15
        com.open()
        com.close = lambda: None

    def setup_sa(self, profile: str, average_off: bool = False) -> None:
        """
        profile is "sa_chopper", "sa_slider" or "sa_irr" from [measurement].
        average_off turns the analyzer's own averaging off, for new_sweeps().
        """
        params = dict(self.m[profile])
        if average_off:
            params["average_onoff"] = "OFF"
        self._averaging = str(params.get("average_onoff", "OFF")).upper() == "ON"
        self._average_num = int(params.get("average_num", 1))
        if self.dry_run:
            self._fake.sa_params = params
            return
        self.sa.setup(**params)

    def fresh_trace(self) -> np.ndarray:
        """
        A trace that contains nothing from before the last change (IF path,
        load, test signal). With averaging on, the analyzer keeps averaging
        across such a change, so the average is restarted and the trace is
        read only after average_num sweeps. The TZ manual asks for the same
        by hand: confirm the averaging has completed before recording.
        Call it after the settling wait.
        """
        if getattr(self, "_averaging", False) and not self.dry_run:
            self.sa.com.open()
            try:
                self.sa.sa.average_restart()
                sweep = float(self.sa.sa.sweep_time_query())
            finally:
                self.sa.com.close()
            wait = self._average_num * sweep * 1.2 + 1.0
            print(f"waiting {wait:.0f} s for {self._average_num} averaged sweeps")
            time.sleep(wait)
        return self.trace()

    def new_sweeps(self, n: int, warmup: float = 2.0, min_gap: float = 1.0, poll: float = 0.05) -> list:
        """
        n traces from n separate sweeps, none of which began before this call.
        For averaging on the PC side instead of in the analyzer: the result
        shows that no sweep from before a change is mixed in, and the spread
        between sweeps gives the uncertainty of every measurement.

        Needs the analyzer's averaging off (setup_sa(..., average_off=True)).
        The trace is read over and over; a change in its content marks a
        finished sweep.

        Measured on 2026-10-22 (timing_test): while traces are read back to
        back, the content changes every 1.3 s (chopper settings) or 1.6 s
        (IRR settings), not every sweep time (0.3 or 0.47 s). In the first
        second after a pause in reading, it changed every 0.2 s, faster than
        one sweep, so those changes are not whole new sweeps; the cause is not
        known. Hence everything read during `warmup` seconds is dropped, and a
        change counts only if it comes at least `min_gap` seconds after the
        previous one.
        Returns [(seconds since the call, trace), ...].
        """
        if self.dry_run:
            return [(0.0, self._fake.trace()) for _ in range(n)]
        self.sa.com.open()
        try:
            sweep = float(self.sa.sa.sweep_time_query())
        finally:
            self.sa.com.close()
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
        """INIT:CONT ON / OFF. Leave it ON at the end of a run (the usual state)."""
        if not self.dry_run:
            self.sa.com.send("INIT:CONT " + ("ON" if on else "OFF"))

    def single_trace(self) -> np.ndarray:
        """
        Start one sweep, wait until the analyzer reports it finished, then read
        the trace. Needs set_continuous(False). Unlike new_sweeps(), this does
        not guess the end of a sweep from the trace content, so the trace
        surely comes from a sweep that began after the call.
        """
        if self.dry_run:
            return self._fake.trace()
        self.sa.com.send("INIT:IMM;*OPC?")
        self.sa.com.readline()
        return self.trace()

    def sa_params(self) -> dict:
        if self.dry_run:
            return dict(self._fake.sa_params)
        try:
            return self.sa.get_current_param()
        except Exception as e:
            return {"error": repr(e)}

    def trace(self) -> np.ndarray:
        if self.dry_run:
            return self._fake.trace()
        return np.array([float(x) for x in self.sa.run()])


def sleep(rig: Rig, seconds: float) -> None:
    """time.sleep, skipped in dry-run."""
    if not rig.dry_run:
        time.sleep(seconds)


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


def base_meta(rig: Rig, args=None) -> dict:
    """Conditions common to every run, written to meta.json."""
    setter = rig.cfg["sis_bias_setter"]
    return {
        "arguments": vars(args) if args is not None else None,
        "config_file": CONFIG_FILE,
        "dry_run": rig.dry_run,
        "thot": rig.m["thot"],
        "tcold": rig.m["tcold"],
        "sis_channel": dict(setter["channel"]),
        "sis_conversion_factor": setter["conversion_factor"],
        "if_switch_path": dict(rig.m["if_switch_path"]),
        "loatt_port": dict(rig.m["loatt_port"]),
    }


# ----------------------------------------------------------------------------
# Fake receiver for --dry-run
# ----------------------------------------------------------------------------

class _FakeReceiver:
    """
    Produces traces of the right shape so that the scripts and the analysis
    can be run end to end. The numbers mean nothing physically.
    """

    N = 461

    def __init__(self, rig: Rig) -> None:
        self.rig = rig
        self.sa_params = {}
        self.rng = np.random.default_rng(0)

    def _trx(self) -> float:
        b = self.rig.bias
        pol = (self.rig.band or "HL")[0]
        pair = ("v1", "v2") if pol == "V" else ("h1", "h2")
        x1, x2 = (b[pair[0]] or 0.0), (b[pair[1]] or 0.0)
        att = self.rig.att.get(pol) or 23.0
        return 80 + 60 * ((x1 - 6.5) ** 2 + (x2 - 7.0) ** 2) + 0.5 * (att - 23) ** 2

    def trace(self) -> np.ndarray:
        trx = self._trx()
        thot, tcold = self.rig.m["thot"], self.rig.m["tcold"]
        noise = self.rng.normal(0, 0.05, self.N)

        def level(t_load):
            return 10 * np.log10(t_load + trx) - 100 + noise

        if self.rig.irr_on:
            # Test tone on top of the hot level. The tone is stronger in the
            # IF port of its own sideband (LO 15.609 GHz x 6).
            tr = level(thot)
            band = self.rig.band or "HU"
            signal_usb = (self.rig.irr_freq or 0) > 15.609
            tr[self.N // 2] += 40 if (band[1] == "U") == signal_usb else 25
            return tr
        if self.rig.load == "hot":
            return level(thot)
        if self.rig.load == "cold":
            return level(tcold)
        # Chopper rotating: hot and cold alternate along the 0-span trace.
        phase = (np.arange(self.N) // 40) % 2 == 0
        return np.where(phase, level(thot), level(tcold))
