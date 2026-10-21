#!/usr/bin/env python3
"""
SIS IVカーブ測定。

本番環境には触らない。使うのは同じディレクトリに置いた次の2つだけ。

    device_config_iv.toml   IV測定専用の config
    sis_reader_multi.py     ch別の換算係数に対応した reader（クラス名も別）

site-packages の tz_controller と ~/tz_workspace/device_config.toml は書き換えない。

使い方（このファイルのあるフォルダで）:
    uv run python iv_curve_measure.py          9月と同じ換算（device_config_iv.toml）
    uv run python iv_curve_measure.py --ros    旧システムの換算（device_config_iv_ros.toml）

出力:
    iv_<timestamp>[_ros]/iv_<mixer>.csv   ミクサごとの生データ
    iv_<timestamp>/iv_curves.png    2x2 のIVカーブ（matplotlib があれば）

掃引範囲について:
    負側は -5.0 mV まで。原点確認のためで、これ以上は振らない（開発者 中島拓氏の助言）。
    正側は 10.0 mV。2026-09-04 の掃引で V偏波のギャップは 7.5-7.6 mV に見えたが、
    H偏波は 8.0 mV までに立ち上がらなかった。接合3個なら理論ギャップは 8.4 mV なので、
    そこを十分越えるところまで伸ばす。既存 sis_iv_measure.py が常用していた上限と同じ値で、
    ハードリミット V_STOP_HARD_LIMIT と一致する（これ以上は preflight が弾く）。
"""

import datetime
import logging
import pathlib
import sys
import time
import tomllib

import numpy as np

# tz_controller は1測定点ごとに INFO ログを出す。146点×4本で600行近く流れて
# 要約が埋もれるので先に basicConfig を済ませて黙らせる（DeviceBase は
# ハンドラが既にあれば basicConfig を呼ばない）。
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from tz_controller.sis_setter import SISSetter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sis_reader_multi import SISReaderMulti  # noqa: E402

# --ros: 旧システムの換算（DA ÷(3*0.94)、AD 5V、電圧 -5 mV/V）で測る。
ROS = "--ros" in sys.argv[1:]
CONFIG_FILE = str(HERE / ("device_config_iv_ros.toml" if ROS else "device_config_iv.toml"))

MIXERS = ["v1", "v2", "h1", "h2"]

V_START = -5.0   # mV
V_STOP = 10.0    # mV
V_STEP = 0.1     # mV
SETTLE_TIME = 0.05  # s

# 安全ガード
EXPECTED_SETTER_FACTOR = 1 / (3 * 0.94) if ROS else 0.94 / 3
FACTOR_TOLERANCE = 0.01
V_START_HARD_LIMIT = -5.0   # これより負には絶対に振らない
V_STOP_HARD_LIMIT = 10.0


def preflight() -> dict:
    """バイアスをかける前に config の数値だけを検証する。"""
    with open(CONFIG_FILE, "rb") as f:
        cfg = tomllib.load(f)

    setter_cfg = cfg["sis_bias_setter"]
    reader_cfg = cfg["sis_bias_reader"]
    factor = setter_cfg["conversion_factor"]

    rel = abs(factor - EXPECTED_SETTER_FACTOR) / EXPECTED_SETTER_FACTOR
    if rel > FACTOR_TOLERANCE:
        raise SystemExit(
            f"中止: setter の conversion_factor が {factor} です。期待値は "
            f"{EXPECTED_SETTER_FACTOR}。このまま掃引すると意図の "
            f"{factor / EXPECTED_SETTER_FACTOR:.2f} 倍のバイアスがかかります。"
        )

    if V_START < V_START_HARD_LIMIT:
        raise SystemExit(
            f"中止: V_START={V_START} mV は負側のハードリミット "
            f"{V_START_HARD_LIMIT} mV を超えています。"
        )

    if V_STOP > V_STOP_HARD_LIMIT:
        raise SystemExit(
            f"中止: V_STOP={V_STOP} mV はハードリミット {V_STOP_HARD_LIMIT} mV を"
            "超えています。"
        )

    # 電圧chと電流chが両方登録されているなら、係数はテーブルでなければならない。
    if not isinstance(reader_cfg["conversion_factor"], dict):
        raise SystemExit(
            "中止: reader の conversion_factor がスカラーです。電圧chと電流chは"
            "モニタゲインが違うので、テーブル形式でなければ必ずどちらかが誤変換されます。"
        )

    # 登録順の点検。電圧chは 2.86 V 振れるので、末尾に置くと先頭の電流chに漏れる。
    names = list(reader_cfg["channel"].keys())
    if names and names[-1].endswith("_v"):
        print(
            f"警告: 登録の末尾が電圧ch ({names[-1]}) です。末尾の大信号は先頭ch "
            f"({names[0]}) に 0.81 倍で漏れます。順序の入れ替えを検討してください。"
        )

    n_points = int(round((V_STOP - V_START) / V_STEP)) + 1
    print("--- 測定条件 ---")
    print(f"  config        : {CONFIG_FILE}")
    print(f"  掃引           : {V_START} -> {V_STOP} mV, {V_STEP} mV 刻み ({n_points}点)")
    print(f"  負側の下限     : {V_START_HARD_LIMIT} mV (中島氏の助言。これ以上振らない)")
    print(f"  setter 係数    : {factor}")
    print(f"  DA 出力の範囲  : {V_START * factor:+.4f} 〜 {V_STOP * factor:+.4f} V (レンジ 10V)")
    print(f"  常用の tuned   : {setter_cfg['tuned']}")
    print(f"  読み出し順     : {' -> '.join(names)}")
    print(f"  換算係数       : {reader_cfg['conversion_factor']}")
    print(f"  掃引するミクサ : {', '.join(MIXERS)} (1本ずつ、同時には出力しない)")
    print()
    if input("この条件で測定しますか [yes/no]: ").strip().lower() != "yes":
        raise SystemExit("中止しました。DA には何も出力していません。")

    return cfg


def measure(mixer: str) -> dict:
    """1本のミクサのIVカーブを測る。"""
    v_ch = f"{mixer}_v"
    i_ch = mixer

    setter = SISSetter(device_name="sis_bias_setter", config_file=CONFIG_FILE)
    reader = SISReaderMulti(device_name="sis_bias_reader", config_file=CONFIG_FILE)

    # 電圧chを先、電流chを後。この順序が読み出し順になる。逆にすると電流chが汚れる。
    reader.setup(**{v_ch: True, i_ch: True})

    # 試し読み。バイアスをかける前にch指定が通ることを確認する。
    reader.run()
    print(f"  0 V での試し読み OK ({v_ch}, {i_ch})")

    f_v = reader.conversion_factors[v_ch]
    f_i = reader.conversion_factors[i_ch]

    rows = []
    try:
        for v_cmd in np.arange(V_START, V_STOP + V_STEP / 2, V_STEP):
            setter.setup(**{mixer: float(v_cmd)})
            setter.run()
            time.sleep(SETTLE_TIME)

            result = reader.run()
            v_mV = result[v_ch]     # 換算済み [mV]
            i_uA = result[i_ch]     # 換算済み [uA]
            rows.append(
                (float(v_cmd), v_mV / f_v, i_uA / f_i, v_mV, i_uA)
            )
    finally:
        setter.teardown()   # 使ったchを 0 V に戻す
        reader.teardown()

    a = np.array(rows)
    return {"cmd_mV": a[:, 0], "v_mon_V": a[:, 1], "i_mon_V": a[:, 2],
            "v_mV": a[:, 3], "i_uA": a[:, 4]}


def save_csv(out_dir: pathlib.Path, mixer: str, d: dict) -> pathlib.Path:
    path = out_dir / f"iv_{mixer}.csv"
    cols = ["cmd_mV", "v_mon_V", "i_mon_V", "v_mV", "i_uA"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for k in range(len(d["cmd_mV"])):
            f.write(",".join(f"{d[c][k]}" for c in cols) + "\n")
    return path


def save_plot(out_dir: pathlib.Path, results: dict, tuned: dict) -> None:
    """matplotlib が無ければ黙って諦める。CSVは既に書けている。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib が無いので作図は省略します。CSVは出ています。")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, mixer in zip(axes.flatten(), MIXERS):
        d = results[mixer]
        # サンプル点をそのまま残す。線は点をたどるための細い補助。
        ax.plot(d["v_mV"], d["i_uA"], lw=0.7, marker=".", ms=3.5, mew=0)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        if mixer in tuned:
            ax.axvline(tuned[mixer], color="tab:red", ls="--", lw=0.9,
                       label=f"tuned {tuned[mixer]} mV")
            ax.legend(fontsize=8)
        ax.set_title(mixer)
        ax.set_xlabel("Junction voltage [mV]")
        ax.set_ylabel("Current [uA]")
        ax.grid(alpha=0.3)

    fig.suptitle(f"SIS IV curves  {out_dir.name}")
    fig.tight_layout()
    path = out_dir / "iv_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"図: {path}")


def main() -> None:
    cfg = preflight()
    tuned = dict(cfg["sis_bias_setter"]["tuned"])

    stamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = HERE / f"iv_{stamp}{'_ros' if ROS else ''}"
    out_dir.mkdir()

    results = {}
    for mixer in MIXERS:
        print(f"\n=== {mixer} を測定 ===")
        d = measure(mixer)
        results[mixer] = d
        print(f"生データ: {save_csv(out_dir, mixer, d)}")

        # その場で見える要約。ギャップ位置と漏れ電流。
        v, i = d["v_mV"], d["i_uA"]
        sub = i[(v > 2.0) & (v < 5.0)]
        print(f"  サブギャップ電流(2-5 mV)の中央値: {np.median(sub):.1f} uA")
        print(f"  最大電流: {i.max():.1f} uA (V={v[int(np.argmax(i))]:.2f} mV)")
        # 電流が漏れ電流の5倍を初めて超えるところをギャップの目安とする
        thr = 5 * abs(np.median(sub))
        over = np.where((v > 5.0) & (i > thr))[0]
        if len(over):
            print(f"  ギャップ立ち上がり(漏れ電流の5倍を超える点): {v[over[0]]:.2f} mV")
        else:
            print(f"  ギャップ立ち上がりは {V_STOP} mV までに見えず")

    save_plot(out_dir, results, tuned)
    print(f"\n完了。{out_dir} を持ち帰ってください。")
    print("バイアスは 0 V に戻っています。観測に戻すときは tuned を入れ直してください。")


if __name__ == "__main__":
    main()
