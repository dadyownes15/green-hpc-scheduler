import os
import csv
import math
import numpy as np

class CarbonIntensity():
    def __init__(self, green_win_length=72, normalize=True, custom_intensity = False) -> None:
        self.green_win_length = green_win_length
        self.granularity = "minutely" # Ensure consistent casing
        self.normalize = normalize
        self.custom_intensity = custom_intensity        
        # CONFIGURATION BASED ON GRANULARITY
        # This is the key change to make the class adaptable
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
        print("Carbon intensity list: ", self.carbonIntensityList)
        self.total_slots = len(self.carbonIntensityList)
        assert (self.total_slots) > 0  
        self.start_offset = 0

        if self.normalize:
            mean = np.mean(self.carbonIntensityList)
            std = np.std(self.carbonIntensityList)
            # Avoid division by zero if all values are the same
            if std > 0:
                self.carbonIntensityList = (self.carbonIntensityList - mean) / std

    def getCarbonEmissions(self, power, start, end):
        """
        Calculate total carbon emissions for a given power consumption over a time period.
        power: power consumption in watts
        start, end: time period in seconds
        Returns: total carbon emissions in gCO2eq
        """
        totalEmissions = 0
        
        # DYNAMIC CALCULATION USING self.seconds_per_slot
        startIndex = int(start / self.seconds_per_slot)
        endIndex = int(end / self.seconds_per_slot)
        t = start

        for i in range(startIndex, endIndex + 1):
            if i == endIndex:
                duration_in_slot = end - t
            else:
                duration_in_slot = (i + 1) * self.seconds_per_slot - t

            # Use more descriptive variable names (slot_index instead of hour_index)
            slot_index = i % self.total_slots
            carbonIntensity = self.carbonIntensityList[slot_index]

            # The energy calculation is correct as it converts a duration in seconds to hours
            # Energy (kWh) = Power (kW) * Time (hours)
            energyKWh = (power / 1000.0) * (duration_in_slot / 3600.0)
            emissions = energyKWh * carbonIntensity  # gCO2eq
            totalEmissions += emissions

            t = (i + 1) * self.seconds_per_slot

        return totalEmissions

    
    def loadCarbonIntensityData(self):
        """Load carbon intensity data from CSV file."""
        current_dir = os.getcwd()

        carbon_file = os.path.join(current_dir, "data/DK-DK2_minutely_carbon_intensity_improved.csv")

        carbon_list = []

        try:
            with open(carbon_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    carbon_list.append(float(row[1]))
        except FileNotFoundError:
            print(f"Error: Carbon intensity file not found at {carbon_file}")
            return np.array([]) # Return empty array to avoid crash
        return np.array(carbon_list)

    def create_carbon_forecast_enconding(self, current_timestamp):
        """
        Creates an encoding with current carbon context and future forecast.
        """
        assert self.start_offset is not None
       
        # DYNAMIC CALCULATION using self.seconds_per_slot
        current_slot = int(current_timestamp // self.seconds_per_slot) % self.total_slots
        time_left_before_new_ci = (self.seconds_per_slot - (current_timestamp % self.seconds_per_slot)) / self.seconds_per_slot # Normalized

        # Cyclical features are now correctly calculated based on the actual time slot
        hour_of_day = (current_slot % self.slots_per_day) / (self.slots_per_day / 24)
        day_of_week = (current_slot // self.slots_per_day) % 7
        slot_of_year = current_slot % self.slots_per_year
        
        two_pi = 2.0 * math.pi
        hour_sin = math.sin(two_pi * hour_of_day / 24.0)
        hour_cos = math.cos(two_pi * hour_of_day / 24.0)
        day_sin = math.sin(two_pi * day_of_week / 7.0)
        day_cos = math.cos(two_pi * day_of_week / 7.0)
        year_sin = math.sin(two_pi * slot_of_year / self.slots_per_year)
        year_cos = math.cos(two_pi * slot_of_year / self.slots_per_year)

        current_ci_norm = self.carbonIntensityList[current_slot]
        carbon_context = [current_ci_norm, time_left_before_new_ci, hour_sin, hour_cos, day_sin, day_cos, year_sin, year_cos]

        forecast = []
        for t in range(1, self.green_win_length): # Start from 1 to get the *next* slots
            if self.granularity == "minutely":
                future_slot_index = (current_slot + t*60) % self.total_slots
            else:
                future_slot_index = (current_slot + t) % self.total_slots
            forecast.append(self.carbonIntensityList[future_slot_index])
            
        # The forecast should contain green_win_length-1 future values
        assert len(forecast) == self.green_win_length - 1

        carbon_encoding = np.concatenate((carbon_context, forecast))
        assert len(carbon_encoding) == 8 + self.green_win_length - 1
        return carbon_encoding
    
    def get_average_intensity_for_period(self, end_time_seconds: float) -> float:
        """
        Calculates the time-weighted average carbon intensity for a given period.
        This correctly handles partial slots at the beginning and end of the period.

        Args:
            start_time_seconds (float): The start time of the period in seconds.
            end_time_seconds (float): The end time of the period in seconds.

        Returns:
            float: The time-weighted average carbon intensity.
        """
        start_time_seconds = 0
        if start_time_seconds >= end_time_seconds:
            return 0.0

        total_duration = end_time_seconds - start_time_seconds
        total_weighted_intensity = 0.0
        
        start_index = int(start_time_seconds / self.seconds_per_slot)
        end_index = int(end_time_seconds / self.seconds_per_slot)

        current_time = start_time_seconds

        for i in range(start_index, end_index + 1):
            # Determine the end of the current time segment.
            # It's either the end of the slot or the end of the total period, whichever comes first.
            segment_end_time = min((i + 1) * self.seconds_per_slot, end_time_seconds)

            # Calculate the duration spent in this specific slot
            duration_in_slot = segment_end_time - current_time

            if duration_in_slot <= 0:
                continue

            # Get the carbon intensity for this slot, accounting for offset and wrap-around
            slot_index = (i + self.start_offset) % self.total_slots
            carbon_intensity = self.carbonIntensityList[slot_index]

            # Add the weighted intensity to the total
            total_weighted_intensity += carbon_intensity * duration_in_slot

            # Move time forward
            current_time = segment_end_time

        # The average is the total weighted intensity divided by the total duration
        return total_weighted_intensity / total_duration
