#!/usr/bin/env python3
"""
Analysis for the measurement scripts ported from the ROS system.

The formulas are taken from the ROS-era Jupyter notebooks so that numbers stay
comparable with ROS-era results:

    chopper Trx   trx_map-cmd_ver5.ipynb, yfactor_chopper.ipynb,
                  LoAtt_vs_Trx.ipynb (evaluate_trx_from_rotating_chopper_data)
    hot/cold Trx  irr_cal_template.ipynb (cal_trx)
    IRR           irr_cal_template.ipynb (irr_cal_linear)

Usage:
    python tz_analysis.py <run_directory>

The run directory is what a measurement script wrote. Its meta.json says which
kind of measurement it holds; the result is written next to the raw data as
result.csv, plus a PNG when matplotlib is available.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

BANDS = ["HU", "HL", "VU", "VL"]


# ----------------------------------------------------------------------------
# Formulas
# ----------------------------------------------------------------------------

def dbm_to_mw(dbm):
    return 10.0 ** (np.asarray(dbm, dtype=float) / 10.0)


def trx_from_y(y, thot: float, tcold: float):
    """Y-factor method. y is P_hot / P_cold in linear units."""
    return (thot - tcold * y) / (y - 1.0)


def chopper_hot_cold(trace, smooth: int = 3, nsig: float = 2.0, mask_edges: bool = True):
    """
    Split a 0-span trace taken while the chopper rotates into hot and cold
    levels, as the ROS notebooks did.

    Samples on the rising and falling edges are found from the smoothed
    derivative and dropped (ver5 behavior; the older yfactor_chopper notebook
    kept them, which is mask_edges=False). The rest is split at its mean:
    above is hot, below is cold. Returns the two mean levels in dBm.
    """
    d = np.array(trace, dtype=float)
    kernel = np.ones(smooth) / float(smooth)
    sd = np.convolve(d, kernel, mode="same")
    sdd = sd[1:] - sd[:-1]
    # The ROS code compares the derivative with its variance, not its
    # standard deviation. Kept as is so that results match.
    sig = np.var(sdd)
    edges = np.concatenate([np.where(sdd > sig * nsig)[0], np.where(sdd < -sig * nsig)[0]])
    if mask_edges:
        d[edges] = np.nan
    av = np.nanmean(d)
    hot = np.nanmean(d[d > av])
    cold = np.nanmean(d[d < av])
    return float(hot), float(cold)


def chopper_trx(trace, thot: float, tcold: float, **kwargs) -> float:
    hot, cold = chopper_hot_cold(trace, **kwargs)
    return float(trx_from_y(10.0 ** ((hot - cold) / 10.0), thot, tcold))


def hotcold_trx(hot_trace, cold_trace, thot: float, tcold: float) -> float:
    """Separate HOT and COLD traces. Y is the mean of the dB difference."""
    diff = np.asarray(hot_trace, dtype=float) - np.asarray(cold_trace, dtype=float)
    return float(trx_from_y(10.0 ** (diff.mean() / 10.0), thot, tcold))


def irr_linear(up_h, up_c, low_h, low_c, gu_u, gl_u, gu_l, gl_l):
    """
    Image rejection ratio of a 2SB receiver, all inputs in linear units.

    up_h, up_c    USB IF power with HOT and COLD loads
    low_h, low_c  LSB IF power with HOT and COLD loads
    gu_u          USB IF <- USB test signal
    gl_u          LSB IF <- USB test signal
    gu_l          USB IF <- LSB test signal
    gl_l          LSB IF <- LSB test signal

    Returns (R_usb, R_lsb) in linear units.
    """
    m_dsb = (up_h - up_c) / (low_h - low_c)
    m_u = gu_u / gl_u
    m_l = gl_l / gu_l
    r_u = m_u * (m_l * m_dsb - 1.0) / (m_u - m_dsb)
    r_l = m_l * (m_u - m_dsb) / (m_l * m_dsb - 1.0)
    return float(r_u), float(r_l)


# ----------------------------------------------------------------------------
# Run directory I/O
# ----------------------------------------------------------------------------

def load_run(run_dir: pathlib.Path):
    meta = json.loads((run_dir / "meta.json").read_text())
    rows = []
    with (run_dir / "traces.csv").open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_info = header.index("p000")
        for r in reader:
            info = dict(zip(header[:n_info], r[:n_info]))
            info["trace"] = np.array([float(x) for x in r[n_info:]])
            rows.append(info)
    return meta, rows


def write_result(run_dir: pathlib.Path, header, rows) -> pathlib.Path:
    path = run_dir / "result.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# ----------------------------------------------------------------------------
# Per-measurement analysis
# ----------------------------------------------------------------------------

def analyze_chopper(run_dir, meta, rows):
    """yfactor_chopper: one 0-span trace per band."""
    thot, tcold = meta["thot"], meta["tcold"]
    out = []
    for r in rows:
        trx = chopper_trx(r["trace"], thot, tcold)
        out.append([r["band"], round(trx, 2)])
        print(f"{r['band']}: Trx = {trx:.1f} K")
    write_result(run_dir, ["band", "trx_K"], out)


def analyze_hotcold(run_dir, meta, rows):
    """yfactor_hotcold: separate HOT and COLD traces per band."""
    thot, tcold = meta["thot"], meta["tcold"]
    by = {(r["load"], r["band"]): r["trace"] for r in rows}
    out = []
    for band in BANDS:
        if ("hot", band) in by and ("cold", band) in by:
            trx = hotcold_trx(by[("hot", band)], by[("cold", band)], thot, tcold)
            out.append([band, round(trx, 2)])
            print(f"{band}: Trx = {trx:.1f} K")
    write_result(run_dir, ["band", "trx_K"], out)


def analyze_trx_map(run_dir, meta, rows):
    """trx_map: chopper Trx on a (bias1, bias2) grid for one band."""
    thot, tcold = meta["thot"], meta["tcold"]
    b1 = np.array([float(r["bias1_mV"]) for r in rows])
    b2 = np.array([float(r["bias2_mV"]) for r in rows])
    trx = np.array([chopper_trx(r["trace"], thot, tcold) for r in rows])
    write_result(run_dir, ["bias1_mV", "bias2_mV", "trx_K"],
                 [[a, b, round(t, 2)] for a, b, t in zip(b1, b2, trx)])
    i = int(np.nanargmin(trx))
    print(f"{meta['band']}: minimum Trx = {trx[i]:.1f} K at bias1 = {b1[i]:.3f}, bias2 = {b2[i]:.3f} mV")

    plt = _pyplot()
    if plt is None:
        return
    u1, u2 = np.unique(b1.round(4)), np.unique(b2.round(4))
    grid = np.full((len(u2), len(u1)), np.nan)
    for a, b, t in zip(b1.round(4), b2.round(4), trx):
        grid[np.searchsorted(u2, b), np.searchsorted(u1, a)] = t
    fig, ax = plt.subplots(figsize=(8, 6.5))
    d1 = (u1[1] - u1[0]) / 2 if len(u1) > 1 else 0.05
    d2 = (u2[1] - u2[0]) / 2 if len(u2) > 1 else 0.05
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="jet", vmin=50, vmax=200,
                   extent=[u1[0] - d1, u1[-1] + d1, u2[0] - d2, u2[-1] + d2])
    fig.colorbar(im, ax=ax, label="Trx [K]")
    ax.set_xlabel("SIS Bias1 [mV]  (" + "+".join(meta["bias1_names"]) + ")")
    ax.set_ylabel("SIS Bias2 [mV]  (" + "+".join(meta["bias2_names"]) + ")")
    ax.set_title(f"Trx map {meta['band']}")
    fig.savefig(run_dir / "trx_map.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def analyze_loatt(run_dir, meta, rows):
    """loatt_search: chopper Trx versus LO attenuator current, per band."""
    thot, tcold = meta["thot"], meta["tcold"]
    out = []
    series = {}
    for r in rows:
        att = float(r["att_mA"])
        trx = chopper_trx(r["trace"], thot, tcold)
        out.append([r["band"], att, round(trx, 2)])
        series.setdefault(r["band"], []).append((att, trx))
    write_result(run_dir, ["band", "att_mA", "trx_K"], out)
    for band, pts in series.items():
        att, trx = np.array(pts).T
        i = int(np.nanargmin(trx))
        print(f"{band}: minimum Trx = {trx[i]:.1f} K at att = {att[i]:g} mA")

    plt = _pyplot()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for band, pts in series.items():
        att, trx = np.array(pts).T
        ax.plot(att, trx, "o-", label=band)
    ax.set_xlabel("LO attenuator current [mA]")
    ax.set_ylabel("Trx [K]")
    ax.grid(True, linestyle=":")
    ax.legend()
    fig.savefig(run_dir / "trx_vs_loatt.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def analyze_irr(run_dir, meta, rows):
    """irr_measure: HOT, COLD, USB and LSB test signal, four bands each."""
    thot, tcold = meta["thot"], meta["tcold"]
    by = {(r["load"], r["band"]): r["trace"] for r in rows}
    out = []
    for pol in ["H", "V"]:
        u, l = pol + "U", pol + "L"
        hot_u, cold_u = by[("hot", u)], by[("cold", u)]
        hot_l, cold_l = by[("hot", l)], by[("cold", l)]
        # Test signal level is read at the peak of (signal - hot), as in the
        # ROS notebook. It used scipy argrelmax(order=300) on 461 points,
        # which in practice is the global maximum; argmax gives the same.
        def peak(band, test):
            tr = by[(test, band)]
            return tr[int(np.argmax(tr - by[("hot", band)]))]
        gu_u = peak(u, "usbtest")
        gl_u = peak(l, "usbtest")
        gu_l = peak(u, "lsbtest")
        gl_l = peak(l, "lsbtest")
        params_db = [np.mean(hot_u), np.mean(cold_u), np.mean(hot_l), np.mean(cold_l),
                     gu_u, gl_u, gu_l, gl_l]
        r_u, r_l = irr_linear(*dbm_to_mw(params_db))
        trx_u = hotcold_trx(hot_u, cold_u, thot, tcold)
        trx_l = hotcold_trx(hot_l, cold_l, thot, tcold)
        print(f"{pol}-pol: IRR_USB = {10 * np.log10(r_u):.2f} dB, IRR_LSB = {10 * np.log10(r_l):.2f} dB, "
              f"Trx {u} = {trx_u:.1f} K, Trx {l} = {trx_l:.1f} K")
        out.append([pol, round(10 * np.log10(r_u), 3), round(10 * np.log10(r_l), 3),
                    round(trx_u, 2), round(trx_l, 2)])
    write_result(run_dir, ["pol", "irr_usb_dB", "irr_lsb_dB", "trx_usb_K", "trx_lsb_K"], out)


ANALYZERS = {
    "yfactor_chopper": analyze_chopper,
    "yfactor_hotcold": analyze_hotcold,
    "trx_map": analyze_trx_map,
    "loatt_search": analyze_loatt,
    "irr_measure": analyze_irr,
}


def analyze(run_dir) -> None:
    run_dir = pathlib.Path(run_dir)
    meta, rows = load_run(run_dir)
    ANALYZERS[meta["kind"]](run_dir, meta, rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    analyze(sys.argv[1])
