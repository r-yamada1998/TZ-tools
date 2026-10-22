#!/usr/bin/env python3
"""
Ask the spectrum analyzer what state it is in, without changing any setting.

Written after timing_test.py stopped on 2026-10-22: SWE:TIME? was answered,
but TRACE:DATA? TRACE1 gave nothing within ogameasure's 3 s timeout. This
script sends the same trace query on a plain socket with a long timeout and
prints what comes back (how long it took, how many bytes, the first bytes),
together with the settings that decide the answer (data format, continuous
sweep, error queue).

Usage (in the folder that holds this script):
    uv run python sa_check.py [--timeout 20]

Output: on screen, and sa_check_<YYMMDD_HHMMSS>.txt.
"""

import argparse
import datetime
import socket
import time

import rx_performance_common as tm


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
    finally:
        s.close()
    return time.monotonic() - t0, data, note


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--timeout", type=float, default=20.0, help="wait for the trace answer [s]")
    args = p.parse_args()

    cfg = tm.load_config()["spectrum_analyzer"]
    host, port = cfg["host"], int(cfg["port"])
    out = Out()
    out.say(f"sa_check {datetime.datetime.now().isoformat()}  analyzer {host}:{port}")

    queries = ["*IDN?", "SYST:ERR?", "SYST:ERR?", "SYST:ERR?", "FORM?", "FORM:BORD?",
               "INIT:CONT?", "SWE:TIME?", "FREQ:CENT?", "FREQ:SPAN?", "BAND?", "BAND:VID?",
               "AVER:TRAC1?", "AVER:TRAC1:COUN?"]
    for q in queries:
        dt, data, note = ask(host, port, q, 3.0)
        out.say(f"  {q:<18} {dt:6.3f} s  {data.strip()[:80]!r}  ({note})")

    for k in range(3):
        dt, data, note = ask(host, port, "TRACE:DATA? TRACE1", args.timeout)
        n_values = data.count(b",") + 1 if data.strip() else 0
        out.say(f"TRACE:DATA? TRACE1 #{k + 1}: {dt:.3f} s, {len(data)} bytes, "
                f"about {n_values} values ({note})")
        out.say(f"    first bytes: {data[:120]!r}")
        out.say(f"    last bytes:  {data[-40:]!r}")

    for _ in range(3):
        dt, data, note = ask(host, port, "SYST:ERR?", 3.0)
        out.say(f"  SYST:ERR? after   {dt:6.3f} s  {data.strip()[:80]!r}  ({note})")
    out.say(f"Saved: {out.path}")


if __name__ == "__main__":
    main()
