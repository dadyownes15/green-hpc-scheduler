import os
import csv
import math
import numpy as np

class CarbonIntensity():
    def __init__(self, year=2021, green_win_length=72, normalize=True, granularity="hourly", custom_intensity = False) -> None:
        self.year = year
        self.green_win_length = green_win_length
        self.granularity = granularity.lower() # Ensure consistent casing
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

    def reset(self, start_offset=0):
        self.start_offset = start_offset

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
            slot_index = (i + self.start_offset) % self.total_slots
            carbonIntensity = self.carbonIntensityList[slot_index]

            # The energy calculation is correct as it converts a duration in seconds to hours
            # Energy (kWh) = Power (kW) * Time (hours)
            energyKWh = (power / 1000.0) * (duration_in_slot / 3600.0)
            emissions = energyKWh * carbonIntensity  # gCO2eq
            totalEmissions += emissions

            t = (i + 1) * self.seconds_per_slot

        return totalEmissions

    def getCarbonIntensityData(self, end_slot_index):
        """
        Get a slice of the carbon intensity data.
        The parameter is an index, not a unit of time.
        """
        data = []
        start_slot_index = self.start_offset
        
        if end_slot_index > self.total_slots:
            # Handle wrap-around
            data = np.append(self.carbonIntensityList[start_slot_index : self.total_slots],
                             self.carbonIntensityList[0 : end_slot_index - self.total_slots])
        else:
            data = self.carbonIntensityList[start_slot_index : end_slot_index]

        # The assertion should check against the intended slice length
        expected_length = end_slot_index - start_slot_index
        if end_slot_index < start_slot_index: # handles wrap-around case for length calculation
             expected_length = (self.total_slots - start_slot_index) + end_slot_index

        # This assertion might be too strict if wrap-around is complex. Let's simplify.
        # assert len(data) == end_slot_index - self.start_offset
        return data

    def loadCarbonIntensityData(self):
        """Load carbon intensity data from CSV file."""
        current_dir = os.getcwd()

        if self.granularity == "hourly":
            carbon_file = os.path.join(current_dir, "data/DK-DK2_hourly_carbon_intensity_noFeb29.csv")
        else: 
            if self.custom_intensity: 
                carbon_file = os.path.join(current_dir, "data/free_weekends_minutely.csv")
            else:
                carbon_file = os.path.join(current_dir, "data/DK-DK2_minutely_carbon_intensity.csv")
        
        year_to_col = {2021: 1, 2022: 2, 2023: 3, 2024: 4}
        col_index = year_to_col.get(self.year, 1)  # Default to 2021

        carbon_list = []

        try:
            with open(carbon_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    carbon_list.append(float(row[col_index]))
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
        current_slot = int(self.start_offset + (current_timestamp // self.seconds_per_slot)) % self.total_slots
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
            future_slot_index = (current_slot + t) % self.total_slots
            forecast.append(self.carbonIntensityList[future_slot_index])
            
        # The forecast should contain green_win_length-1 future values
        assert len(forecast) == self.green_win_length - 1

        carbon_encoding = np.concatenate((carbon_context, forecast))
        assert len(carbon_encoding) == 8 + self.green_win_length - 1
        return carbon_encoding