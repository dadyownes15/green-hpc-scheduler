import os
import csv
import math
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np


class CarbonIntensity:
    """
    Provides carbon intensity signals aligned to calendar-based splits.
    Episodes that extend past the split boundary wrap to the split start.
    """

    DATASET_START = datetime(2021, 1, 1, 0, 0, 0)
    SPLIT_WINDOWS: Dict[str, Tuple[datetime, datetime]] = {
        # Inclusive windows – each entry corresponds to an available CSV row.
        "training": (
            datetime(2021, 1, 1, 0, 0, 0),
            datetime(2023, 12, 31, 23, 59, 0),
        ),
        "validation": (
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2024, 3, 31, 23, 59, 0),
        ),
        "test": (
            datetime(2024, 4, 1, 0, 0, 0),
            datetime(2024, 12, 31, 23, 59, 0),
        ),
    }

    def __init__(self, green_win_length: int = 72, normalize: bool = True, custom_intensity: bool = False) -> None:
        self.green_win_length = green_win_length
        self.granularity = "minutely"  # Ensure consistent casing
        self.normalize = normalize
        self.custom_intensity = custom_intensity

        if self.granularity == "hourly":
            self.seconds_per_slot = 3600
            self.slots_per_day = 24
        elif self.granularity == "minutely":
            self.seconds_per_slot = 60
            self.slots_per_day = 24 * 60
        else:
            raise ValueError("Granularity must be 'hourly' or 'minutely'")

        self.slots_per_year = 365 * self.slots_per_day

        self.carbonIntensityList = self.loadCarbonIntensityData()
        self.total_slots = len(self.carbonIntensityList)
        if self.total_slots == 0:
            raise ValueError("Carbon intensity dataset is empty.")

        if self.normalize:
            self.mean = np.mean(self.carbonIntensityList)
            self.std = np.std(self.carbonIntensityList)
            if self.std > 0:
                self.carbonIntensityList = (self.carbonIntensityList - self.mean) / self.std

        # Window configuration; defaults to the full dataset until set_mode is called.
        self.window_start_slot = 0
        self.window_slot_count = self.total_slots
        self.window_start_seconds = 0
        self.window_length_seconds = self.window_slot_count * self.seconds_per_slot
        self.start_offset = self.window_start_seconds  # kept for back-compat where consumer relies on it

        self._configure_full_dataset()

    def loadCarbonIntensityData(self) -> np.ndarray:
        """Load carbon intensity data from CSV file."""
        current_dir = os.getcwd()
        carbon_file = os.path.join(current_dir, "data/DK-DK2_minutely_carbon_intensity_improved.csv")

        carbon_list = []

        try:
            with open(carbon_file, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    carbon_list.append(float(row[1]))
        except FileNotFoundError:
            print(f"Error: Carbon intensity file not found at {carbon_file}")
            return np.array([])  # Return empty array to avoid crash
        return np.array(carbon_list)

    # ------------------------------------------------------------------ #
    # Split configuration helpers
    # ------------------------------------------------------------------ #
    def _dataset_end_datetime(self) -> datetime:
        last_offset = (self.total_slots - 1) * self.seconds_per_slot
        return self.DATASET_START + timedelta(seconds=int(last_offset))

    def _datetime_to_slot(self, dt: datetime) -> int:
        delta = dt - self.DATASET_START
        seconds = delta.total_seconds()
        if seconds < 0:
            raise ValueError(f"Timestamp {dt} predates dataset start.")
        slot = int(seconds // self.seconds_per_slot)
        if slot >= self.total_slots:
            # Clamp to the last available slot so callers can request slightly
            # out-of-range endpoints without crashing (e.g., missing final minute).
            slot = self.total_slots - 1
        return slot

    def _configure_window(self, start_dt: datetime, end_dt: datetime) -> None:
        if end_dt < start_dt:
            raise ValueError("Carbon intensity window end must be >= start.")

        start_slot = self._datetime_to_slot(start_dt)
        end_slot = self._datetime_to_slot(end_dt)

        self.window_start_slot = start_slot
        self.window_slot_count = (end_slot - start_slot) + 1
        if self.window_slot_count <= 0:
            raise ValueError("Carbon intensity window has no slots.")

        self.window_start_seconds = self.window_start_slot * self.seconds_per_slot
        self.window_length_seconds = self.window_slot_count * self.seconds_per_slot
        self.start_offset = self.window_start_seconds

    def _configure_full_dataset(self) -> None:
        self._configure_window(self.DATASET_START, self._dataset_end_datetime())

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def set_mode(self, mode: str) -> None:
        """
        Restrict carbon intensity lookups to the calendar window associated with `mode`.
        Episodes automatically wrap to the start of the window on overflow.
        """
        mode = mode.lower()
        if mode not in self.SPLIT_WINDOWS:
            raise ValueError("mode must be one of {'training', 'validation', 'test'}.")
        start_dt, end_dt = self.SPLIT_WINDOWS[mode]
        self._configure_window(start_dt, end_dt)

    # ------------------------------------------------------------------ #
    # Lookup helpers
    # ------------------------------------------------------------------ #
    def _ensure_window_configured(self) -> None:
        if self.window_slot_count <= 0 or self.window_length_seconds <= 0:
            raise RuntimeError("Carbon intensity split window is not configured.")

    def _wrap_env_seconds(self, env_seconds: float) -> float:
        self._ensure_window_configured()
        relative = env_seconds % self.window_length_seconds
        return self.window_start_seconds + relative

    def _slot_index_for_env_time(self, env_seconds: float) -> int:
        wrapped_seconds = self._wrap_env_seconds(env_seconds)
        slot_index = int(wrapped_seconds // self.seconds_per_slot)
        return slot_index % self.total_slots

    def intensity_at(self, env_seconds: float) -> float:
        """
        Return the carbon intensity value for an environment timestamp, respecting split wrapping.
        """
        idx = self._slot_index_for_env_time(env_seconds)
        return self.carbonIntensityList[idx]

    # ------------------------------------------------------------------ #
    # Emission integration
    # ------------------------------------------------------------------ #
    def getCarbonEmissions(self, power: float, start: float, end: float) -> float:
        """
        Calculate total carbon emissions for a given power consumption over a time period.
        power: power consumption in watts
        start, end: time period in seconds
        Returns: total carbon emissions in gCO2eq
        """
        if end <= start or power <= 0:
            return 0.0

        self._ensure_window_configured()

        total_emissions = 0.0
        duration = end - start
        processed = 0.0
        current_offset = start % self.window_length_seconds

        while processed < duration:
            segment_len = min(self.window_length_seconds - current_offset, duration - processed)
            segment_start_abs = self.window_start_seconds + current_offset
            segment_end_abs = segment_start_abs + segment_len
            total_emissions += self._integrate_segment(power, segment_start_abs, segment_end_abs)

            processed += segment_len
            current_offset = 0.0  # subsequent segments start at window beginning

        return total_emissions

    def _integrate_segment(self, power: float, start_abs: float, end_abs: float) -> float:
        if end_abs <= start_abs:
            return 0.0

        start_index = int(math.floor(start_abs / self.seconds_per_slot))
        end_index = int(math.ceil(end_abs / self.seconds_per_slot)) - 1
        if end_index < start_index:
            end_index = start_index

        total = 0.0
        current_time = start_abs
        for index in range(start_index, end_index + 1):
            slot_end = min((index + 1) * self.seconds_per_slot, end_abs)
            duration_in_slot = slot_end - current_time
            if duration_in_slot <= 0:
                current_time = slot_end
                continue

            slot = index % self.total_slots
            carbon_intensity = self.carbonIntensityList[slot]
            energy_kwh = (power / 1000.0) * (duration_in_slot / 3600.0)
            total += energy_kwh * carbon_intensity
            current_time = slot_end

        return total

    # ------------------------------------------------------------------ #
    # Forecast encoding
    # ------------------------------------------------------------------ #
    def create_carbon_forecast_enconding(self, current_timestamp: float) -> np.ndarray:
        """
        Creates an encoding with current carbon context and future forecast.
        """
        self._ensure_window_configured()

        wrapped_seconds = self._wrap_env_seconds(current_timestamp)
        relative_seconds = (wrapped_seconds - self.window_start_seconds) % self.window_length_seconds

        current_slot_relative = int(relative_seconds // self.seconds_per_slot)
        current_slot_absolute = (self.window_start_slot + current_slot_relative) % self.total_slots

        remainder = wrapped_seconds % self.seconds_per_slot
        time_left_before_new_ci = (self.seconds_per_slot - remainder) / self.seconds_per_slot

        # Cyclical features are calculated from the absolute slot to preserve calendar context.
        hour_of_day = (current_slot_absolute % self.slots_per_day) / (self.slots_per_day / 24)
        day_of_week = (current_slot_absolute // self.slots_per_day) % 7
        slot_of_year = current_slot_absolute % self.slots_per_year

        two_pi = 2.0 * math.pi
        hour_sin = math.sin(two_pi * hour_of_day / 24.0)
        hour_cos = math.cos(two_pi * hour_of_day / 24.0)
        day_sin = math.sin(two_pi * day_of_week / 7.0)
        day_cos = math.cos(two_pi * day_of_week / 7.0)
        year_sin = math.sin(two_pi * slot_of_year / self.slots_per_year)
        year_cos = math.cos(two_pi * slot_of_year / self.slots_per_year)

        current_ci_norm = self.carbonIntensityList[current_slot_absolute]
        carbon_context = [
            current_ci_norm,
            time_left_before_new_ci,
            hour_sin,
            hour_cos,
            day_sin,
            day_cos,
            year_sin,
            year_cos,
        ]

        step_slots = 60 if self.granularity == "minutely" else 1
        forecast = []
        for step in range(1, self.green_win_length):
            future_relative = (current_slot_relative + step * step_slots) % self.window_slot_count
            future_absolute = (self.window_start_slot + future_relative) % self.total_slots
            forecast.append(self.carbonIntensityList[future_absolute])

        assert len(forecast) == self.green_win_length - 1

        carbon_encoding = np.concatenate((carbon_context, forecast))
        assert len(carbon_encoding) == 8 + self.green_win_length - 1
        return carbon_encoding
