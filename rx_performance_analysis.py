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
    python rx_performance_analysis.py <run_directory>
    python rx_performance_analysis.py irr-map <output_directory> <irr_run_dir> [<irr_run_dir> ...]

The run directory is what a measurement script wrote. Its meta.json says which
kind of measurement it holds; the result is written next to the raw data as
result.csv, plus a PNG when matplotlib is available.

irr-map combines several irr_measure runs taken at different SIS biases into
IRR and Trx maps per polarization (ROS: irr_map_multi_data_integration.ipynb).

In every map the color and the number written in a cell come from the same
array element. Two ROS notebooks got this wrong in different ways: the Trx map
wrote transposed numbers, and the IRR map could shift its colors by a column.
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


def to_grid(x, y, z, decimals: int = 4):
    """
    Put scattered (x, y, z) on a grid. Axes are the distinct values actually
    measured, so a float step can never create an extra row or column (the
    ROS IRR notebook built its axes with numpy.arange and duplicated one).
    Returns (xs, ys, grid) with grid[iy, ix].
    """
    x = np.round(np.asarray(x, dtype=float), decimals)
    y = np.round(np.asarray(y, dtype=float), decimals)
    xs, ys = np.unique(x), np.unique(y)
    grid = np.full((len(ys), len(xs)), np.nan)
    grid[np.searchsorted(ys, y), np.searchsorted(xs, x)] = z
    return xs, ys, grid


def _edges(c):
    """Cell edges around sorted centers c (spacing may be uneven)."""
    if len(c) == 1:
        return np.array([c[0] - 0.05, c[0] + 0.05])
    mid = (c[1:] + c[:-1]) / 2
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def draw_map(ax, xs, ys, grid, vmin, vmax, label, fmt="{:.0f}", numbers=True):
    """
    Color map with the value written in each cell.

    Colors and numbers are drawn from the same grid[iy, ix], cell by cell, so
    they cannot disagree. (The ROS Trx notebook wrote the numbers from the
    transposed array: each number belonged to the cell mirrored across the
    diagonal.)
    """
    im = ax.pcolormesh(_edges(xs), _edges(ys), grid, cmap="jet", vmin=vmin, vmax=vmax)
    ax.figure.colorbar(im, ax=ax, label=label)
    if numbers:
        size = 10 if grid.size <= 150 else 6
        for iy, yv in enumerate(ys):
            for ix, xv in enumerate(xs):
                v = grid[iy, ix]
                if np.isfinite(v):
                    ax.text(xv, yv, fmt.format(v), ha="center", va="center", fontsize=size,
                            color="white", fontweight="bold",
                            path_effects=_outline())
    return im


def _outline():
    from matplotlib.patheffects import withStroke
    return [withStroke(linewidth=1.5, foreground="black")]


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


def _hz(v) -> float:
    """'6GHz', '500kHz', 6e9 -> Hz."""
    if isinstance(v, (int, float)):
        return float(v)
    v = str(v).strip()
    for unit, f in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3), ("Hz", 1.0)):
        if v.endswith(unit):
            return float(v[: -len(unit)]) * f
    return float(v)


def freq_axis(meta, n: int):
    """IF frequency of each trace point [GHz], from the recorded analyzer settings."""
    sa = meta.get("sa_params") or {}
    try:
        center, span = _hz(sa["center"]), _hz(sa["span"])
    except (KeyError, ValueError, TypeError):
        return None
    return np.linspace(center - span / 2, center + span / 2, n) / 1e9


def analyze_hotcold(run_dir, meta, rows):
    """
    yfactor_hotcold: separate HOT and COLD traces per band, taken with the
    slider. Gives the band Trx and, point by point, Trx versus IF frequency.
    """
    thot, tcold = meta["thot"], meta["tcold"]
    by = {(r["load"], r["band"]): r["trace"] for r in rows}
    out, spectra = [], {}
    for band in BANDS:
        if ("hot", band) in by and ("cold", band) in by:
            hot, cold = by[("hot", band)], by[("cold", band)]
            trx = hotcold_trx(hot, cold, thot, tcold)
            spectra[band] = trx_from_y(10.0 ** ((hot - cold) / 10.0), thot, tcold)
            out.append([band, round(trx, 2)])
            print(f"{band}: Trx = {trx:.1f} K")
    write_result(run_dir, ["band", "trx_K"], out)

    n = len(next(iter(spectra.values()))) if spectra else 0
    f = freq_axis(meta, n)
    with (run_dir / "trx_spectrum.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["if_freq_GHz" if f is not None else "point"] + list(spectra))
        for i in range(n):
            x = round(float(f[i]), 6) if f is not None else i
            w.writerow([x] + [round(float(spectra[b][i]), 2) for b in spectra])

    plt = _pyplot()
    if plt is None or not spectra:
        return
    x = f if f is not None else np.arange(n)
    fig, ax = plt.subplots(figsize=(9, 5))
    for band, t in spectra.items():
        ax.plot(x, t, label=f"{band} (band Trx {dict(out)[band]:.0f} K)")
    ax.set_xlabel("IF frequency [GHz]" if f is not None else "trace point")
    ax.set_ylabel("Trx [K]")
    ax.set_ylim(0, 300)
    ax.grid(True, linestyle=":")
    ax.legend()
    fig.savefig(run_dir / "trx_spectrum.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


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
    xs, ys, grid = to_grid(b1, b2, trx)
    fig, ax = plt.subplots(figsize=(12, 9.5))
    draw_map(ax, xs, ys, grid, 50, 200, "Trx [K]")
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


def irr_values(meta, rows):
    """
    IRR and Trx of both polarizations from one irr_measure run.
    Returns {pol: dict(irr_usb_dB, irr_lsb_dB, trx_usb_K, trx_lsb_K)}.
    """
    thot, tcold = meta["thot"], meta["tcold"]
    by = {(r["load"], r["band"]): r["trace"] for r in rows}
    result = {}
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

        params_db = [np.mean(hot_u), np.mean(cold_u), np.mean(hot_l), np.mean(cold_l),
                     peak(u, "usbtest"), peak(l, "usbtest"), peak(u, "lsbtest"), peak(l, "lsbtest")]
        r_u, r_l = irr_linear(*dbm_to_mw(params_db))
        result[pol] = {
            "irr_usb_dB": 10 * np.log10(r_u),
            "irr_lsb_dB": 10 * np.log10(r_l),
            "trx_usb_K": hotcold_trx(hot_u, cold_u, thot, tcold),
            "trx_lsb_K": hotcold_trx(hot_l, cold_l, thot, tcold),
        }
    return result


def analyze_irr(run_dir, meta, rows):
    """irr_measure: HOT, COLD, USB and LSB test signal, four bands each."""
    res = irr_values(meta, rows)
    out = []
    for pol, r in res.items():
        print(f"{pol}-pol: IRR_USB = {r['irr_usb_dB']:.2f} dB, IRR_LSB = {r['irr_lsb_dB']:.2f} dB, "
              f"Trx {pol}U = {r['trx_usb_K']:.1f} K, Trx {pol}L = {r['trx_lsb_K']:.1f} K")
        out.append([pol] + [round(r[k], 3) for k in ("irr_usb_dB", "irr_lsb_dB", "trx_usb_K", "trx_lsb_K")])
    write_result(run_dir, ["pol", "irr_usb_dB", "irr_lsb_dB", "trx_usb_K", "trx_lsb_K"], out)


def irr_map(run_dirs, out_dir) -> None:
    """
    Combine several irr_measure runs, taken at different SIS biases, into
    maps: for each polarization, IRR and Trx of USB and LSB over
    (bias1, bias2) = (v1, v2) or (h1, h2). Port of the ROS notebook
    irr_map_multi_data_integration.ipynb.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for d in run_dirs:
        meta, rows = load_run(pathlib.Path(d))
        b = meta["bias_mV"]
        for pol, r in irr_values(meta, rows).items():
            n1, n2 = ("v1", "v2") if pol == "V" else ("h1", "h2")
            table.append([str(d), pol, b[n1], b[n2]] + [r[k] for k in
                         ("irr_usb_dB", "irr_lsb_dB", "trx_usb_K", "trx_lsb_K")])
    header = ["run", "pol", "bias1_mV", "bias2_mV", "irr_usb_dB", "irr_lsb_dB", "trx_usb_K", "trx_lsb_K"]
    with (out_dir / "irr_map.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    print(f'Table: "{out_dir / "irr_map.csv"}"')

    plt = _pyplot()
    if plt is None:
        return
    for pol in ["H", "V"]:
        rows = [t for t in table if t[1] == pol]
        if not rows:
            continue
        x = [t[2] for t in rows]
        y = [t[3] for t in rows]
        fig, axes = plt.subplots(2, 2, figsize=(16, 13))
        for ax, col, title, lim, unit, fmt in [
            (axes[0, 0], 5, "LSB IRR", (0, 25), "IRR [dB]", "{:.1f}"),
            (axes[0, 1], 4, "USB IRR", (0, 25), "IRR [dB]", "{:.1f}"),
            (axes[1, 0], 7, "LSB Trx", (50, 200), "Trx [K]", "{:.0f}"),
            (axes[1, 1], 6, "USB Trx", (50, 200), "Trx [K]", "{:.0f}"),
        ]:
            xs, ys, grid = to_grid(x, y, [t[col] for t in rows])
            draw_map(ax, xs, ys, grid, lim[0], lim[1], unit, fmt=fmt)
            n1, n2 = ("v1", "v2") if pol == "V" else ("h1", "h2")
            ax.set_xlabel(f"SIS Bias1 [mV]  ({n1})")
            ax.set_ylabel(f"SIS Bias2 [mV]  ({n2})")
            ax.set_title(f"{pol}-pol. {title}")
        fig.savefig(out_dir / f"irr_map_{pol}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def analyze_settle(run_dir, meta, rows, tol: float = 0.05):
    """settle_test: chopper Trx against time since each bias change."""
    thot, tcold = meta["thot"], meta["tcold"]
    watch = float(meta["arguments"]["watch"])
    out, curves = [], []
    for c in sorted({int(r["change"]) for r in rows}):
        rs = [r for r in rows if int(r["change"]) == c]
        t = np.array([float(r["t_read_start_s"]) for r in rs])
        t_end = np.array([float(r["t_read_end_s"]) for r in rs])
        trx = np.array([chopper_trx(r["trace"], thot, tcold) for r in rs])
        final = float(np.nanmedian(trx[t >= watch - 3.0])) if np.any(t >= watch - 3.0) else float(trx[-1])
        off = np.abs(trx - final) > tol * abs(final)
        # The first read after which every read stays within tol of the final value
        idx = np.nonzero(off)[0]
        if len(idx) == 0:
            settled = 0.0
        elif idx[-1] + 1 < len(t):
            settled = float(t[idx[-1] + 1])
        else:
            settled = float("nan")
        read_s = float(np.median(t_end - t))
        out.append([c, rs[0]["target"], round(final, 2), settled, round(read_s, 3), len(rs)])
        curves.append((c, rs[0]["target"], t, trx))
        print(f"change {c} -> {rs[0]['target']}: final Trx {final:.1f} K, "
              f"within {tol:.0%} from {settled:.2f} s, one read takes {read_s:.2f} s")
    write_result(run_dir, ["change", "target", "final_trx_K", "settled_after_s", "read_time_s", "n_traces"], out)

    plt = _pyplot()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for c, target, t, trx in curves:
        ax.plot(t, trx, ".-", color="C0" if target == "B" else "C1", alpha=0.7, lw=0.8)
    ax.plot([], [], ".-", color="C0", label="A -> B")
    ax.plot([], [], ".-", color="C1", label="B -> A")
    ax.set_xlabel("start of trace read since bias change [s]")
    ax.set_ylabel("chopper Trx [K]")
    ax.set_title(f"Settling after SIS bias change, {meta['band']}")
    ax.legend()
    fig.savefig(run_dir / "settle_test.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


ANALYZERS = {
    "settle_test": analyze_settle,
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
    if len(sys.argv) >= 4 and sys.argv[1] == "irr-map":
        irr_map(sys.argv[3:], sys.argv[2])
    elif len(sys.argv) == 2:
        analyze(sys.argv[1])
    else:
        raise SystemExit(__doc__)
