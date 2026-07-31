import csv
from datetime import datetime
from datetime import timedelta
from random import Random
import numpy as np
from functools import reduce

class Solar:
    def __init__(self, csv_path: str, scale_factor: float = 1, max_power:int = 0, enable_cache = False, prediction_accuracy:float = 1):
        self.values = []
        self.max_power_w = max_power
        self.scale_factor = scale_factor
        self.enable_cache = enable_cache
        self.cache = {}
        self.day_avg_cache = {}
        self.day_values_cache = {}
        self.prediction_accuracy = prediction_accuracy
        self.last_middle = 0.5
        self.last_bounds = (0.0,0.0)
        self.rng = Random(1234)
        self.start_time:int = 2**64
        self.start_datetime: datetime = None

        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)
            gti_index = header.index("gti")
            cloud_index = header.index("cloud_opacity")
            humidty_index = header.index("relative_humidity")
            pressure_index = header.index("surface_pressure")
            period_index = header.index("period_end")

            for line in reader:
                dt = datetime.fromisoformat(line[period_index])
                start_of_the_year = dt.replace(
                    month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                delta = dt - start_of_the_year
                info = {
                    'cloud_opacity':line[cloud_index],
                    'humidty':line[humidty_index],
                    'pressure':line[pressure_index],
                }
                
                value = (delta.total_seconds(), int(line[gti_index])*scale_factor, dt, info)
                self.values.append(value)

                if delta.total_seconds() < self.start_time:
                    self.start_time = delta.total_seconds()
                    self.start_datetime = dt

        self.time_map = {int(v[0]): v[1] for v in self.values}

    def get_solar_w(self, time_s: int) -> int:
        time_int = int(time_s)
        if time_int in self.time_map:
            v = self.time_map[time_int]
            v = min(self.max_power_w, v) if self.max_power_w > 0 else v
            return max(v, 0)

        for i in range(len(self.values)):
            if self.values[i][0] == time_s:
                v = self.values[i][1]
                v = min(self.max_power_w,v) if self.max_power_w > 0 else v
                v = max(v, 0)
                return v
            elif i+1 < len(self.values) and self.values[i][0] < time_s and self.values[i+1][0] > time_s:
                current_t = self.values[i][0]
                next_t = self.values[i+1][0]
                delta_second = next_t - current_t
                delta_value = self.values[i+1][1]-self.values[i][1]
                fraction = delta_value / delta_second
                v = self.values[i][1]+fraction*(time_s-current_t)
                v = min(self.max_power_w,v) if self.max_power_w > 0 else v
                return max(v, 0)
        return -1
    
    def get_info(self, time_s: int, field:str):
        # print("Solar time:", time_s)
        if self.enable_cache and str(time_s) in self.cache.keys():
            return self.cache[str(time_s)]
        
        for i in range(len(self.values)):
            if self.values[i][0] == time_s:
                v = self.values[i][3][field]
                # if self.enable_cache:
                #     self.cache[str(time_s)]=v
                return v
            elif i+1 < len(self.values) and self.values[i][0] < time_s and self.values[i+1][0] > time_s:
                # print(f"Found: {self.values[i]}")
                current_t = self.values[i][0]
                next_t = self.values[i+1][0]
                delta_second = next_t - current_t
                delta_value = self.values[i+1][3][field]-self.values[i][3][field]
                fraction = delta_value / delta_second
                v = self.values[i][3][field]+fraction*(time_s-current_t)
                v = min(self.max_power_w,v) if self.max_power_w > 0 else v
                # if self.enable_cache:
                #     self.cache[str(time_s)]=v
                return v
        # print("Solar not found")
        return -1
    
    def get_datetime(self, time_s: int) -> datetime:
        diff = time_s - self.start_time
        return self.start_datetime+timedelta(seconds=diff)

    def get_next_sunrise(self, time_s: int) -> int:
        # Step 1: Find next item in the data
        # Step 2: Find the sunrise
        for i in range(len(self.values)):
            if self.values[i][0] >= time_s:
                # next_item_found
                # Go head untill is night
                k = i
                while self.values[k][1] == 0:
                    k += 1
                return self.values[k][0]


    # Current step if the last with the night
    def is_sunrise(self, time_s:int, step_size_s:int, steps:int = 5):
        if self.get_solar_w(time_s) != 0: #Now i dark
            return False
        
        for i in range(steps-1):
            next_time = time_s + step_size_s*(i+1) #Next steps with light
            if self.get_solar_w(next_time) <= 0:
                return False
        return True
    
    def is_sunset(self, time_s:int, step_size_s:int, steps:int = 5):
        if self.get_solar_w(time_s) == 0: #If now there is light
            return False
        
        for i in range(steps-1): #Next steps witout light
            next_time = time_s + step_size_s*(i+1)
            if self.get_solar_w(next_time) > 0:
                return False
        return True        
    
    def get_sunset_time(self, time_s:int, step_size_s:int)->int:
        while not self.is_sunset(time_s, step_size_s):
            time_s += step_size_s
        return time_s
    
    def are_steps_with_at_least(self, time_s:int, step_size_s:int, steps:int, power_w:int) -> bool:
        for i in range(steps):
            next_time = time_s + step_size_s*i
            if self.get_solar_w(next_time) < power_w:
                return False
        return True
    
    def is_night(self, time_s:int, step_size_s:int, steps:int) -> bool:
        for i in range(steps):
            next_time = time_s + step_size_s*i
            if self.get_solar_w(next_time) > 0:
                return False
        return True
    
    def solar_slope(self, time_s:int, future_m:int) -> bool:
        a = self.get_solar_w(time_s)
        b = self.get_solar_w(time_s+future_m)
        if a < b:
            return 1
        elif a > b:
            return -1
        else: #a > b
            return 0
        
    def get_day_avg_w(self, day:int) -> float: #0-365
        if str(day) in self.day_avg_cache.keys():
            return self.day_avg_cache[str(day)]
        acc = 0
        #counter = 0
        for x in self.values:
            if x[2].timetuple().tm_yday == day:
                v = min(x[1], self.max_power_w)
                acc += v
                #counter += 1
        self.day_avg_cache[str(day)]=acc
        return acc
    
    def get_day_values(self, day:int) -> list[float]: #0-365
        if str(day) in self.day_values_cache.keys():
            return self.day_values_cache[str(day)]
        values = []
        for x in self.values:
            if x[2].timetuple().tm_yday == day:
                v = min(x[1], self.max_power_w)
                values.append(v/self.max_power_w)
                #counter += 1
        self.day_values_cache[str(day)]=values
        return values
    
    def get_real_future_prediction_j(self, time_s:int, step_size_s:int, interval_m:int) -> float:
        accumulated_power_j = 0
        for t in range(time_s, time_s+interval_m*60,step_size_s):
            e = self.get_solar_w(t)
            if e < 0:
                break
            accumulated_power_j += e*step_size_s
        return accumulated_power_j
    
    def get_estimate_future_prediction_j(self, time_s:int, step_size_s:int, window_size_m:int) -> tuple[float,float]:
        accumulated_power_lower_j = 0
        accumulated_power_upper_j = 0
        lower_bounds_list = []
        upper_bounds_list = []
        for t in range(time_s, time_s+window_size_m*60,step_size_s):
            lower_bound, upper_bound = self.get_solar_prediction_w(t)
            if lower_bound < 0 or upper_bound < 0:
                break
            lower_bounds_list.append(lower_bound)
            upper_bounds_list.append(upper_bound)

        # lower_bounds_list = np.convolve(lower_bounds_list, np.ones(window)/window, mode='same')
        # upper_bounds_list = np.convolve(upper_bounds_list, np.ones(window)/window, mode='same')

        for i in range(len(lower_bounds_list)):
            accumulated_power_lower_j += lower_bounds_list[i]*step_size_s
            accumulated_power_upper_j += upper_bounds_list[i]*step_size_s

        return accumulated_power_lower_j, accumulated_power_upper_j
    def get_estimate_future_single_prediction_j(self, time_s:int, step_size_s:int, window_size_m:int) -> float:
        lower_j, upper_j = self.get_estimate_future_prediction_j(time_s, step_size_s, window_size_m)
        return lower_j + (upper_j-lower_j)/2
        
    def get_solar_prediction_w(self, time_s: int, alpha1=0.8, alpha2=0.1) -> tuple[float, float]:
        real_production_w = self.get_solar_w(time_s)
        if real_production_w < 0:
            return -1, -1
        elif real_production_w == 0:
            return 0, 0
        uncertainty_w = (1 - self.prediction_accuracy) * self.max_power_w

        # Smoothing progressivo per middle
        self.last_middle = alpha1 * self.rng.random() + (1 - alpha1) * self.last_middle
        middle = self.last_middle

        upper_bound = real_production_w + uncertainty_w * middle
        lower_bound = real_production_w - uncertainty_w * (1 - middle)

        if self.prediction_accuracy == 1:
            return real_production_w, real_production_w

        # Smoothing progressivo sui bound (opzionale)
        lower_smooth, upper_smooth = (
            alpha2 * lower_bound + (1 - alpha2) * self.last_bounds[0],
            alpha2 * upper_bound + (1 - alpha2) * self.last_bounds[1]
        )
        lower_smooth = max(0, lower_smooth)
        upper_smooth = min(upper_smooth, self.max_power_w)
        if real_production_w < lower_smooth:
            lower_smooth = real_production_w
        if real_production_w > upper_smooth:
            upper_smooth = real_production_w
        
        self.last_bounds = lower_smooth, upper_smooth

        return lower_smooth, upper_smooth

        


