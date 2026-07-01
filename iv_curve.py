#!/usr/bin/env python3
"""
SIS IV curve measurement script.
Sweeps bias voltage and records actual junction voltage and current.

Usage:
    uv run python iv_curve.py
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import datetime

sys.path.insert(0, "/root/tz_workspace/TZ-controller")

from tz_controller.sis_setter import SISSetter
from tz_controller.sis_reader import SISReader

CONFIG_FILE = "/root/tz_workspace/TZ-controller/tz_controller/device_config.toml"

MIXERS = ["v1", "v2", "h1", "h2"]

# WARNING: Asymmetric range per hardware constraint.
# Bias box is designed for positive voltage; negative side is limited
# to -5 mV for origin confirmation (per receiver developer Tak Nakajima's advice).
V_START = -5.0   # mV
V_STOP  = 10.0   # mV
V_STEP  =  0.1   # mV
SETTLE_TIME = 0.05  # seconds

def measure_iv(mixer: str) -> tuple[list[float], list[float]]:
    """
    Measure the IV curve of the specified mixer channel.

    Parameters
    ----------
    mixer : str
        Channel name (v1, v2, h1, h2)

    Returns
    -------
    tuple[list[float], list[float]]
        (Actual Voltage List [V], Current Monitor Voltage List [V])
    """
    setter = SISSetter(device_name="sis_bias_setter", config_file=CONFIG_FILE)
    reader = SISReader(device_name="sis_bias_reader", config_file=CONFIG_FILE)

    v_ch = f"{mixer}_v"
    i_ch = mixer

    reader.setup(**{v_ch: True, i_ch: True})

    voltages = []
    currents = []

    try:
        for v_cmd in np.arange(V_START, V_STOP + V_STEP, V_STEP):
            setter.setup(**{mixer: float(v_cmd)})
            setter.run()
            time.sleep(SETTLE_TIME)

            result = reader.run()
            voltages.append(result[v_ch])
            currents.append(result[i_ch])

    finally:
        setter.teardown()
        reader.teardown()

    return voltages, currents


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, mixer in zip(axes, MIXERS):
        print(f"Measuring {mixer}...")
        voltages, currents = measure_iv(mixer)

        ax.plot(voltages, currents)
        ax.set_title(mixer)
        ax.set_xlabel("Junction Voltage [V]")
        ax.set_ylabel("Current monitor [V]")
        ax.grid(True)

    fig.suptitle(f"SIS IV Curves - {timestamp}")
    plt.tight_layout()

    save_path = f"iv_curve_{timestamp}.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
