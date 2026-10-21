# ~/tz_workspace/ に置いて使うスタンドアロン版。site-packages の tz_controller は
# 書き換えない。本番の SISReader とはクラスが別なので、DeviceBase のシングルトン
# (cls, device_name, identity) も別枠になり、本番インスタンスと衝突しない。
#
# 本家との差分は conversion_factor のみ。スカラーでも従来通り動き、
# テーブル形式 { v1 = -500, v1_v = -2.80 } を書いたときだけch別に適用される。
# await処理が必要
from __future__ import annotations

from typing import Dict, List

from tz_controller.device_base import DeviceBase


class SISReaderMulti(DeviceBase):
    """
    SIS bias reader for Interface CPZ3177, with per-channel conversion factors.

    Example:
        reader.setup()
        result = reader.run()

        reader.setup(v1=True, h2=True)
        result = reader.run()

    If no keyword arguments are given to setup(), all configured channels are read.
    If keyword arguments are given, only the specified channels are read.
    """

    Manufacturer = "Interface"
    Model = "CPZ3177A"
    Identifier = "rsw_id"

    VALID_RANGES = {"0_10V", "5V", "10V"}

    def __init__(self, device_name: str, config_file: str) -> None:
        super().__init__(device_name, config_file)

        import pyinterface

        self.ad = pyinterface.open(3177, self.device_config.rsw_id)
        self.ad.stop_sampling()
        self.ad.initialize()

        self.smpl_ch_req: List[Dict[str, object]] = []
        self.selected_names: List[str] = []

    def setup(self, **kwargs: bool) -> None:
        """
        Build sampling channel requests from config.

        Examples:
            setup()
            setup(v1=True)
            setup(v1=True, v2=True)

        Rules:
            - If no kwargs are given, all configured channels are used.
            - If kwargs are given, only keys with truthy values are used.
            - kwargs keys must exist in config.channel and config.ch_range.
        """
        channel_map = dict(self.device_config.channel)
        range_map = dict(self.device_config.ch_range)
        if kwargs:
            requested_names = [name for name, enabled in kwargs.items() if enabled]
        else:
            requested_names = list(channel_map.keys())

        if not requested_names:
            raise ValueError("No channels were selected in setup().")

        smpl_ch_req: List[Dict[str, object]] = []

        for name in requested_names:
            if name not in channel_map:
                valid = ", ".join(channel_map.keys())
                raise KeyError(
                    f"Unknown channel name: {name!r}. Valid names are: {valid}"
                )

            if name not in range_map:
                raise KeyError(f"Range is not defined for {name!r} in config.")

            range_name = range_map[name]
            if range_name not in self.VALID_RANGES:
                valid = ", ".join(sorted(self.VALID_RANGES))
                raise ValueError(
                    f"Invalid range for {name!r}: {range_name!r}. "
                    f"Valid ranges are: {valid}"
                )

            ch_no = self._parse_channel_number(channel_map[name])

            smpl_ch_req.append(
                {
                    "ch_no": ch_no,
                    "range": range_name,
                }
            )

        self.smpl_ch_req = smpl_ch_req
        self.selected_names = requested_names
        self.conversion_factors = self._resolve_conversion_factors(
            self.device_config.conversion_factor, requested_names
        )

    def run(self) -> Dict[str, float]:
        """
        Read averaged voltages from the configured sampling channels.

        Returns:
            A dictionary mapping logical channel names to averaged values.
        """
        if not self.smpl_ch_req:
            raise RuntimeError("setup() must be called before run().")
        
        self.ad.set_sampling_config(
            smpl_ch_req=self.smpl_ch_req,
            smpl_num=self.device_config.smpl_num,
            smpl_freq=self.device_config.smpl_freq,
            single_diff=self.device_config.single_diff,
            trig_mode="ETERNITY",
        )
        self.ad.start_sampling("ASYNC")
        ave_num = self.device_config.ave_num

        while True:
            status = self.ad.get_status()
            smpl_count = status["smpl_count"]
            if smpl_count > ave_num:
                break
            else:
                continue

        offset = smpl_count - ave_num
        if offset < 0:
            raise RuntimeError(
                f"Not enough samples available yet: smpl_count={smpl_count}, "
                f"required={ave_num}"
            )

        data = self.ad.read_sampling_buffer(ave_num, offset)

        ave_data: Dict[str, float] = {}
        for i, name in enumerate(self.selected_names):
            values = [data[k][i] for k in range(ave_num)]
            ave_data[name] = self.conversion_factors[name]*(sum(values) / ave_num)

        self.logger.info(f"SIS bias reading results are {ave_data}")
        return ave_data

    def teardown(self) -> None:
        """
        Stop sampling.
        """
        self.ad.stop_sampling()
        self.ad.clear_sampling_data()

    @staticmethod
    def _resolve_conversion_factors(
        conversion_factor: object,
        names: List[str],
    ) -> Dict[str, float]:
        """
        Resolve a per-channel conversion factor for each requested channel.

        The config value may be either:

            conversion_factor = -500                      # scalar, applied to all
            conversion_factor = { v1 = -500, v1_v = 2.82 }  # per-channel table

        A scalar is kept for backward compatibility, but note that current
        monitor channels and voltage monitor channels have different monitor
        gains. Whenever both kinds are registered in one section, the table
        form must be used, otherwise one of them is necessarily mis-scaled.
        """
        if isinstance(conversion_factor, (int, float)) and not isinstance(
            conversion_factor, bool
        ):
            return {name: float(conversion_factor) for name in names}

        if isinstance(conversion_factor, dict):
            factors: Dict[str, float] = {}
            for name in names:
                if name not in conversion_factor:
                    valid = ", ".join(conversion_factor.keys())
                    raise KeyError(
                        f"conversion_factor is not defined for {name!r} in config. "
                        f"Defined names are: {valid}"
                    )
                value = conversion_factor[name]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(
                        f"conversion_factor for {name!r} must be int or float, "
                        f"got {type(value).__name__}"
                    )
                factors[name] = float(value)
            return factors

        raise TypeError(
            "conversion_factor must be a number or a table of numbers, "
            f"got {type(conversion_factor).__name__}"
        )

    @staticmethod
    def _parse_channel_number(ch_label: str) -> int:
        """
        Convert a channel label like 'ch5' to integer 5.
        """
        if not isinstance(ch_label, str):
            raise TypeError(
                f"Channel label must be a string like 'ch5', got {type(ch_label).__name__}"
            )

        if not ch_label.startswith("ch"):
            raise ValueError(f"Invalid channel label: {ch_label!r}")

        try:
            return int(ch_label[2:])
        except ValueError as e:
            raise ValueError(f"Invalid channel label: {ch_label!r}") from e
