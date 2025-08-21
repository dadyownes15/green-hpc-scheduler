import os
import csv
import math
import numpy as np

class CarbonIntensity():
    def __init__(self, year = 2021, green_win_length = 72) -> None: 
        self.year = year
        self.green_win_length = green_win_length
        self.carbonIntensityList = self.loadCarbonIntensityData()
    
    def reset(self, start_offset = 0):
        self.start_offset = start_offset
        
    def getCarbonEmissions(self, power, start, end):
        """
        Calculate total carbon emissions for a given power consumption over time period
        power: power consumption in watts
        start, end: time period in seconds
        Returns: total carbon emissions in gCO2eq
        """
        totalEmissions = 0
        startIndex = int(start / 3600)
        endIndex = int(end / 3600)
        t = start
        
        for i in range(startIndex, endIndex + 1):
            if i == endIndex:
                lastTime = end - t
            else:
                lastTime = (i + 1) * 3600 - t
            
            # Handle wrap-around for year-long data with start offset
            hour_index = (i + self.start_offset) % len(self.carbonIntensityList)
            carbonIntensity = self.carbonIntensityList[hour_index]
            
            # Convert power from watts to kW and time from seconds to hours
            energyKWh = (power / 1000.0) * (lastTime / 3600.0)
            emissions = energyKWh * carbonIntensity  # gCO2eq
            totalEmissions += emissions
            
            t = (i + 1) * 3600
        
        return totalEmissions
    
    def getCarbonItensityData(self, end_hour):
        data = []
        if end_hour > 8760:
            data = self.carbonIntensityList[self.start_offset : 8760].append(self.carbonIntensityList[0:end_hour-8760])
        else: 
            data = self.carbonIntensityList[self.start_offset : end_hour]
        
        assert(len(data) == end_hour - self.start_offset)
        return data
    
    
    def loadCarbonIntensityData(self):
        """Load carbon intensity data from CSV file"""
        current_dir = os.getcwd()
        carbon_file = os.path.join(current_dir, "./data/DK-DK2_hourly_carbon_intensity_noFeb29.csv")
        
        # Map year to column index
        year_to_col = {2021: 1, 2022: 2, 2023: 3, 2024: 4}
        col_index = year_to_col.get(self.year, 1)  # Default to 2021
        
        carbon_list = []
        with open(carbon_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                carbon_list.append(float(row[col_index]))
        
        return carbon_list

    def create_carbon_forecast_enconding(self, current_timestamp):
        # This function returns the following carbon enconding
        # CUrrent carbon and time enconding, and timeleft before switching 
        # CArbon forecast, included the next GREEN_WIN hours of carbon intensity


        # Cyclical encodings based on episode offset and current time
        total_hours = int(self.start_offset + (current_timestamp // 3600)) % 8760
        time_left_before_new_ci = (current_timestamp % 3600) / 3600 # Normalized
        hour_of_day = total_hours % 24
        day_of_week = (total_hours // 24) % 7
        hour_of_year = total_hours % (365 * 24)

        two_pi = 2.0 * math.pi
        hour_sin = math.sin(two_pi * hour_of_day / 24.0)
        hour_cos = math.cos(two_pi * hour_of_day / 24.0)
        day_sin = math.sin(two_pi * day_of_week / 7.0)
        day_cos = math.cos(two_pi * day_of_week / 7.0)
        year_sin = math.sin(two_pi * hour_of_year / (365.0 * 24.0))
        year_cos = math.cos(two_pi * hour_of_year / (365.0 * 24.0))

        assert total_hours < 8760
        current_ci_norm = self.carbonIntensityList[total_hours]
        carbon_context = [current_ci_norm, time_left_before_new_ci, hour_sin, hour_cos, day_sin, day_cos, year_sin, year_cos]

        forecast = []
        for t in range(self.green_win_length-1):
            hour_index = (total_hours + t) % 8760
            assert hour_index < 8760
            forecast.append(self.carbonIntensityList[hour_index])

        carbon_encoding = np.concatenate((carbon_context, forecast))
        assert len(carbon_encoding) == 8 + self.green_win_length - 1 
        return carbon_encoding