#!/usr/bin/env python3
"""
Plot the CSV files off the receiver, on a laptop.

    python3 plot_iv.py <directory or csv file> [...]

The two kinds of CSV are told apart by name.

    iv_*.csv     from iv_curve.py: IV curves
    probe_*.csv  from probe_channels.py: raw volts of every AD channel, and
                 the same data converted into IV curves

The PNG goes next to the input. With no arguments the current directory is
searched.
"""

import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MIXER_ORDER = ["v1", "v2", "h1", "h2"]

# Shared range so the four panels can be compared. The sweep runs -5 to 10 mV.
XLIM = (-7.0, 12.0)

# Channel map and conversion factors for turning probe_*.csv into IV curves.
# Determined by the sweep on 2026-09-04. Odd = current, even = voltage monitor.
PROBE_MAP = {"v1": ("c9", "c10"), "h1": ("c11", "c12"),
             "h2": ("c13", "c14"), "v2": ("c15", "c16")}
I_FACTOR = -500.0   # uA/V
V_FACTOR = -2.80    # mV/V


def read_csv(path: pathlib.Path) -> tuple[list[str], dict[str, list[float]]]:
    with open(path) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    cols = {name: [] for name in header}
    for r in rows[1:]:
        for name, value in zip(header, r):
            cols[name].append(float(value))
    return header, cols


def plot_iv(paths: list[pathlib.Path], out: pathlib.Path) -> None:
    """Draw the iv_<mixer>.csv files as a 2x2 grid of IV curves."""
    by_mixer = {p.stem.replace("iv_", ""): p for p in paths}
    order = [m for m in MIXER_ORDER if m in by_mixer]
    order += [m for m in by_mixer if m not in order]

    n = len(order)
    ncol = 2 if n > 1 else 1
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.2 * nrow), squeeze=False)

    ilim = []
    for ax, mixer in zip(axes.flatten(), order):
        _, c = read_csv(by_mixer[mixer])
        # Keep the sample points. The line is a thin guide between them.
        ax.plot(c["v_mV"], c["i_uA"], lw=0.7, marker=".", ms=3.5, mew=0)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_title(mixer)
        ax.set_xlabel("Junction voltage [mV]")
        ax.set_ylabel("Current [uA]")
        ax.grid(alpha=0.3)
        ilim.append((min(c["i_uA"]), max(c["i_uA"])))

    for ax in axes.flatten()[n:]:
        ax.axis("off")

    lo = min(a for a, _ in ilim)
    hi = max(b for _, b in ilim)
    pad = 0.05 * (hi - lo)
    for ax in axes.flatten()[:n]:
        ax.set_xlim(*XLIM)
        ax.set_ylim(lo - pad, hi + pad)

    fig.suptitle(f"SIS IV curves  {out.parent.name}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote: {out}")


def plot_probe(paths: list[pathlib.Path], out: pathlib.Path) -> None:
    """Draw every AD channel from probe_<mixer>.csv, colouring the ones that moved."""
    by_mixer = {p.stem.replace("probe_", ""): p for p in paths}
    order = [m for m in MIXER_ORDER if m in by_mixer]
    order += [m for m in by_mixer if m not in order]

    n = len(order)
    ncol = 2 if n > 1 else 1
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.2 * nrow), squeeze=False)

    for ax, mixer in zip(axes.flatten(), order):
        header, c = read_csv(by_mixer[mixer])
        x = c[header[0]]
        for name in header[1:]:
            y = c[name]
            span = max(y) - min(y)
            if span < 0.05:
                ax.plot(x, y, color="0.8", lw=0.5, marker=".", ms=2,
                        mew=0, zorder=1)
            else:
                ax.plot(x, y, lw=0.7, marker=".", ms=3.5, mew=0, zorder=2,
                        label=f"{name} ({span:.2f} V)")
        ax.set_title(f"{mixer} sweep")
        ax.set_xlabel("Bias command [mV]")
        ax.set_ylabel("Raw AD voltage [V]")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=8)

    for ax in axes.flatten()[n:]:
        ax.axis("off")

    fig.suptitle(f"AD channel probe  {out.parent.name}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote: {out}")


def plot_probe_as_iv(paths: list[pathlib.Path], out: pathlib.Path) -> None:
    """
    Convert probe_*.csv with the known channel map and draw real IV curves.

    probe_channels.py records raw volts, so plotting it directly puts volts on
    both axes. Multiplying the voltage column by -2.80 mV/V and the current
    column by -500 uA/V is what turns it into junction voltage and current.
    """
    by_mixer = {p.stem.replace("probe_", ""): p for p in paths}
    order = [m for m in MIXER_ORDER if m in by_mixer and m in PROBE_MAP]
    if not order:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), squeeze=False)
    ilim = []
    for ax, mixer in zip(axes.flatten(), order):
        header, c = read_csv(by_mixer[mixer])
        i_col, v_col = PROBE_MAP[mixer]
        if i_col not in c or v_col not in c:
            ax.axis("off")
            continue

        cmd = c[header[0]]
        v_mV = [y * V_FACTOR for y in c[v_col]]
        i_uA = [y * I_FACTOR for y in c[i_col]]

        ax.plot(v_mV, i_uA, lw=0.7, marker=".", ms=3.5, mew=0)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_title(f"{mixer}   I: ch{i_col[1:]},  V: ch{v_col[1:]}")
        ax.set_xlabel("Junction voltage [mV]")
        ax.set_ylabel("Current [uA]")
        ax.grid(alpha=0.3)
        ilim.append((min(i_uA), max(i_uA)))

    for ax in axes.flatten()[len(order):]:
        ax.axis("off")

    # Share the axes across panels, so the difference in current between the
    # H and V mixers is visible at a glance.
    lo = min(a for a, _ in ilim)
    hi = max(b for _, b in ilim)
    pad = 0.05 * (hi - lo)
    for ax in axes.flatten()[:len(order)]:
        ax.set_xlim(*XLIM)
        ax.set_ylim(lo - pad, hi + pad)

    fig.suptitle(f"SIS IV curves from channel probe  {out.parent.name}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote: {out}")


def main() -> None:
    args = sys.argv[1:] or ["."]

    files: list[pathlib.Path] = []
    for a in args:
        p = pathlib.Path(a)
        files.extend(sorted(p.glob("*.csv")) if p.is_dir() else [p])

    # Group by directory, so several inputs do not end up on one figure.
    made = False
    for parent in sorted({p.parent for p in files}):
        here = [p for p in files if p.parent == parent]
        iv = [p for p in here if p.name.startswith("iv_")]
        probe = [p for p in here if p.name.startswith("probe_")]
        if iv:
            plot_iv(iv, parent / "iv_curves.png")
            made = True
        if probe:
            plot_probe(probe, parent / "probe_channels.png")
            plot_probe_as_iv(probe, parent / "probe_iv.png")
            made = True

    if not made:
        raise SystemExit(
            f"No iv_*.csv or probe_*.csv found (searched: {', '.join(args)})"
        )


if __name__ == "__main__":
    main()
