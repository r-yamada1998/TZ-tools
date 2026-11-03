#!/usr/bin/env python3
"""
Ask the spectrum analyzer what state it is in, without changing any setting.

Sends queries on a plain socket with a long timeout and prints how long each
answer took, how many bytes came, and the first bytes, together with the
settings that decide the answer (data format, continuous sweep, error queue).
On purpose it does not go through SpectrumAnalyzer: it checks the
communication that class relies on.

Part 1 opens a new connection for every query, as SpectrumAnalyzer does;
part 2 keeps one connection open. On 2026-10-22 only *IDN? was answered at
once on a new connection; the trace came after about 4 s, beyond the 3 s
timeout of ogameasure.

Usage (in the TZ-tools folder):
    uv run python checks/sa_check.py [--timeout 10]

Output: on screen, and sa_check_<YYMMDD_HHMMSS>.txt.
"""

import argparse
import datetime
import pathlib
import socket
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import rx_performance_common as tm  # noqa: E402


class Out:
    def __init__(self):
        self.path = f"sa_check_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.txt"
        self.lines = []

    def say(self, text: str) -> None:
        print(text, flush=True)
        self.lines.append(text)
        with open(self.path, "w") as f:
            f.write("\n".join(self.lines) + "\n")


def ask(host: str, port: int, cmd: str, timeout: float) -> tuple:
    """Send one command on a fresh connection. Returns (seconds, raw bytes, note)."""
    s = socket.create_connection((host, port), timeout=5)
    try:
        return ask_on(s, cmd, timeout)
    finally:
        s.close()


def ask_on(s: socket.socket, cmd: str, timeout: float) -> tuple:
    """Send one command on an open connection. Returns (seconds, raw bytes, note)."""
    s.settimeout(timeout)
    data = b""
    t0 = time.monotonic()
    note = "line end received"
    try:
        s.sendall((cmd + "\n").encode())
        while not data.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                note = "connection closed by the analyzer"
                break
            data += chunk
    except socket.timeout:
        note = f"no line end within {timeout} s"
    return time.monotonic() - t0, data, note


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--timeout", type=float, default=10.0, help="wait for each answer [s]")
    p.add_argument("--device-config", metavar="PATH",
                   help="device_config.toml to use (default: see measure_config.toml)")
    args = p.parse_args()

    cfg = tm.load_configs(args.device_config)[1]["spectrum_analyzer"]
    host, port = cfg["host"], int(cfg["port"])
    out = Out()
    out.say(f"sa_check {datetime.datetime.now().isoformat()}  analyzer {host}:{port}")

    queries = ["*IDN?", "SYST:ERR?", "FORM?", "INIT:CONT?", "SWE:TIME?", "FREQ:SPAN?",
               "BAND?", "BAND:VID?", "AVER:TRAC1?"]

    def trace(dt, data, note, label):
        n_values = data.count(b",") + 1 if data.strip() else 0
        out.say(f"  {label:<26} {dt:6.3f} s  {len(data)} bytes, about {n_values} values ({note})")

    out.say("== 1. a new connection for every query (as SpectrumAnalyzer does) ==")
    for q in queries:
        dt, data, note = ask(host, port, q, args.timeout)
        out.say(f"  {q:<26} {dt:6.3f} s  {data.strip()[:60]!r}  ({note})")
    for k in range(3):
        trace(*ask(host, port, "TRACE:DATA? TRACE1", args.timeout), f"TRACE:DATA? TRACE1 #{k + 1}")

    out.say("== 2. one connection kept open ==")
    s = socket.create_connection((host, port), timeout=5)
    try:
        for q in queries:
            dt, data, note = ask_on(s, q, args.timeout)
            out.say(f"  {q:<26} {dt:6.3f} s  {data.strip()[:60]!r}  ({note})")
        for k in range(10):
            dt, data, note = ask_on(s, "TRACE:DATA? TRACE1", args.timeout)
            trace(dt, data, note, f"TRACE:DATA? TRACE1 #{k + 1}")
            if k == 0:
                out.say(f"    first bytes: {data[:80]!r}")
    finally:
        s.close()
    out.say(f"Saved: {out.path}")


if __name__ == "__main__":
    main()
