import gymnasium as gym
import numpy as np
from lib.solar.solar import Solar
import time
import random
from lib.utils import StateContent
import torch.nn as nn
import torch

class EnvBeeDay(gym.Env):
    def __init__(self,
                 solar: Solar,
                 step_s: int,
                 selected_day:int,
                 start_hour:int,
                 end_hour:int,
                 acquistion_speed_fps:int,
                 processing_speed_fps:int,
                 max_buffer_size:int = 2000,
                 battery_wh:float = 6.6*3.7,
                 device_idle_energy_w = 2.5,
                 device_full_energy_w = 7.5,
                 seed:int = 1234,
                 state_content:int = StateContent.SOLAR,
                 random_reset:bool = True,
                 terminated_days:int = 1,
                 forecast_steps:int = 128,
                 choose_forecast:bool = False,
                 latent_size:int = 24,
                 train_days:int = 30,
                 start_threshold:float = 0.1,
                 prevision_noise_amount:float = 0.2,
                 only_day_acquisition:bool = False,
                 battery_weight:float = 1.0,
                 buffer_weight:float = 1.0,
                 random_day_switch:bool = False,
                 discrete_action:bool = False,
                 reward_shape:int = 1
                   ):
        self.state_content = state_content
        self.random_reset = random_reset
        self.terminated_days = terminated_days
        self.forecast_steps = forecast_steps
        self.choose_forecast = choose_forecast
        self.selected_day = selected_day
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.acquisition_speed_fps = acquistion_speed_fps
        self.processing_speed_fps = processing_speed_fps
        self.max_buffer_size = max_buffer_size
        self.latent_size = latent_size
        self.train_days = train_days
        self.start_threshold = start_threshold
        self.prevision_noise_amount = prevision_noise_amount
        self.only_day_acquisition = only_day_acquisition
        self.battery_weight = battery_weight
        self.buffer_weight = buffer_weight
        self.random_day_switch = random_day_switch
        self.discrete_action = discrete_action
        self.reward_shape = reward_shape

        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.solar = solar

        self.battery_max_j = battery_wh*3600
        self.battery_curr_j = self.battery_max_j*0.5
        self.time_s = 5*60
        self.boot_time_s = self.time_s
        self.step_size_s = step_s
        self.max_buffer_size = self.step_size_s*acquistion_speed_fps*60*5
        if self.state_content & StateContent.NEXT_DAY or self.state_content & StateContent.DAY_AVG:
            self.max_avg_day = self.solar.max_power_w#max(self.solar.get_day_avg(x) for x in range(366))
        self.device_idle_energy_w = device_idle_energy_w
        self.device_full_energy_w = device_full_energy_w
        self.energy_for_idle_step_j = device_idle_energy_w*self.step_size_s
        self.energy_for_image_processing_j = device_full_energy_w/processing_speed_fps
        self.processed_images = 0
        self.haversted_energy_j = self.battery_curr_j
        self.buffer_length = 0
        if self.state_content & StateContent.EMBEDDED_PREV_NEXT_DAY or \
            self.state_content & StateContent.EMBEDDED_CURRENT_DAY or \
                self.state_content & StateContent.EMBEDDED_NEXT_DAY:
            self.linear_mlp = nn.Linear(self.forecast_steps, self.latent_size)
            torch.nn.init.xavier_uniform_(self.linear_mlp.weight)
            torch.nn.init.constant_(self.linear_mlp.bias, 0)
        else:
            self.linear_mlp = None
        self.observation_space = gym.spaces.Box(0.0, 1.0, [len(self.__get_obs()[0])])
        if choose_forecast:
            self.action_space = gym.spaces.Box(low=0,high=1,shape=(3,),dtype=float)
        else:
            if self.discrete_action:
                self.action_space = gym.spaces.Discrete(2) # 0: process all images, 1: process no images
            else:
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=float)

        self.reset()

    def reset(self, *, seed=None, options=None):

        # No random
        if (options is not None and options["norandom"] == True) or not self.random_reset:
            self.battery_curr_j = self.battery_max_j*0.3
            self.buffer_length = 0
            if options is not None and options["day"] is not None:
                day = options["day"]
            else:
                day = self.selected_day
        else:
            self.battery_curr_j = self.battery_max_j*(self.rng.randint(3,7)/10)
            #day = (300+self.rng.randint(0,120))%365
            day = self.rng.randint(0,self.train_days)
            self.buffer_length = self.rng.randint(0,3*self.max_buffer_size//4)

        if self.start_hour < 0 and self.start_threshold > 0:
            # Start with battery at least at {self.start_threshold}%
            self.battery_curr_j = 0
            self.time_s = day*24*60*60
            while self.battery_curr_j/self.battery_max_j <= self.start_threshold:
                self.time_s += self.step_size_s
                solar_power_w = self.solar.get_solar_w(self.time_s)
                solar_energy_j = max(0,solar_power_w*self.step_size_s)
                self.haversted_energy_j += solar_energy_j
                self.battery_curr_j += solar_energy_j
        elif self.start_hour < 0 and self.start_threshold <= 0:
            # random start hour
            self.start_hour = self.rng.randint(0,23)
            self.time_s = (day*24+self.start_hour)*60*60
        else:
            self.time_s = (day*24+self.start_hour)*60*60

        self.boot_time_s = self.time_s
        self.processed_images = 0
        self.haversted_energy_j = self.battery_curr_j
        self.elapsed_time_s = 0
        
        obs, fields = self.__get_obs()
        return obs, {"fields":fields}
    
    def step(self, action):
        done = False
        datetime = self.solar.get_datetime(self.time_s)

        # Set solar data and recharge battery
        solar_power_w = self.solar.get_solar_w(self.time_s) #Returns -1 if end of data
        if solar_power_w <= 0:
            pass
        solar_energy_j = max(0,solar_power_w*self.step_size_s)
        self.haversted_energy_j += solar_energy_j
        self.battery_curr_j += solar_energy_j

        '''
        # Set when episode terminate
        if self.terminated_days > 1:
            # terminate after {self.terminated_days} days
            terminated = self.get_uptime_s() >= self.terminated_days * 24 * 60 * 60
        elif self.terminated_days == 1 and self.end_hour > 0:
            # terminate after {self.end_hour}
            terminated = datetime.hour >= self.end_hour and datetime.minute >= 0 and datetime.second >= 0
        elif (self.terminated_days == 1 and solar_energy_j == 0 and self.end_hour < 0 and self.buffer_length == 0) or (self.get_uptime_s() > 16*60*60 and self.solar.get_solar_w(self.time_s+self.step_size_s) > 0):
            terminated = True # Terminate at sunset
        else:
            terminated = False
        '''
        # if (solar_energy_j == 0 or terminated) and self.only_day_acquisition:
        #     captured_images = 0
        # else:
        captured_images = self.acquisition_speed_fps * self.step_size_s

        processable_images = self.processing_speed_fps * self.step_size_s # Processable images must be > captured

        # Stable-Baselines can return either a scalar action (DQN/discrete)
        # or a 1D array (Box policies). Normalize both cases to a scalar.
        action = np.asarray(action).reshape(-1)[0]
        
        processed_amount = round(action*processable_images)                                   # Images processed in the next interval
        processed_amount = min(processed_amount, self.buffer_length + captured_images)        # Processed images has as upper bound the buffered images
        
        necessary_enegy_j = (self.energy_for_image_processing_j * processed_amount) + self.energy_for_idle_step_j
        if self.battery_curr_j > necessary_enegy_j: 
            # Enought battery to process
            self.processed_images+=processed_amount
            buffered_amount = captured_images - processed_amount
            self.buffer_length += buffered_amount
            #reward = processed_amount
        #else: # Turned off
        #    reward = -processed_amount - self.buffer_length  
        self.battery_curr_j -= necessary_enegy_j
        
        #r_batt = -1/((max(0,self.battery_curr_j)/self.battery_max_j)+0.0001)
        discarged_images = 0
        if self.buffer_length > self.max_buffer_size:
            discarged_images = self.buffer_length - self.max_buffer_size
            #reward -= discarged_images #Penalized if buffered images cannot be stored
            self.buffer_length = self.max_buffer_size

        #reward /= processable_images
        done = self.battery_curr_j<=0
        self.elapsed_time_s += self.step_size_s
        terminated = self.elapsed_time_s >= self.terminated_days * 24 * 60 * 60

        self.battery_curr_j = max(0,min(self.battery_max_j, self.battery_curr_j))

        #reward_buff = -1/(1-(self.buffer_length/self.max_buffer_size)+0.0001)
        #reward_batt = -1/((self.battery_curr_j/self.battery_max_j)+0.0001)
        #reward = 0.4*reward_buff# + 0.6*reward_batt
        if self.reward_shape == 1:
            reward = -self.battery_weight * abs(action - self.battery_curr_j/self.battery_max_j) - self.buffer_weight*self.buffer_length / self.max_buffer_size
        elif self.reward_shape == 2:
            reward = -self.battery_weight * abs(action - self.battery_curr_j/self.battery_max_j) - self.buffer_weight*(discarged_images/captured_images)
        elif self.reward_shape == 3:
            reward_elaborazione = processed_amount / processable_images  # Valore positivo tra 0 e 1
            penalita_scarico = self.buffer_weight * (discarged_images / captured_images)
            
            # Se la batteria si azzera, applichiamo una forte penalità di morte
            penalita_morte = 1.0 if done else 0.0 
            
            reward = reward_elaborazione - penalita_scarico - penalita_morte
        elif self.reward_shape == 4:
            buffer_level = self.buffer_length / self.max_buffer_size
            battery_level = self.battery_curr_j / self.battery_max_j
            processing_reward = processed_amount / processable_images
            buffer_penalty = self.buffer_weight * (buffer_level ** 2)
            battery_penalty = self.battery_weight * max(0, 0.2 - battery_level)
            death_penalty = 10.0 if done else 0.0
            reward = processing_reward - buffer_penalty - battery_penalty - death_penalty
        else:
            reward = 0

        '''
        action batt reward
        0 0 0 good
        0 1 -1 bad
        1 0 1 bad
        1 1 0 good
        0.5 0.5 0
        '''

        prev_day = self.time_s // (24 * 60 * 60)
        self.time_s += self.step_size_s
        new_day = self.time_s // (24 * 60 * 60)
        if new_day != prev_day and self.random_reset and self.random_day_switch:
            random_day = self.rng.randint(0, self.train_days)
            self.time_s = random_day * 24 * 60 * 60

        obs, fields = self.__get_obs()

        info = {"fields":fields,
                "cons":processed_amount/processable_images,
                "time":datetime,
                "images":captured_images,
                "amount":processed_amount
                }

        return obs, reward, done, terminated, info

    def render(self, mode="human"):
        pass

    def __get_obs(self, when_m:int = 0, windows_size_m:int = 60)->tuple[np.array,list]:
        fields = [
            "Battery"
        ]
        arr = [
            self.battery_curr_j/self.battery_max_j
        ]

        if self.state_content & StateContent.SOLAR:
            arr.append(self.solar.get_solar_w(self.time_s)/self.solar.max_power_w)
            fields.append("Solar")
        if self.state_content & StateContent.MONTH:
            arr.append(self.solar.get_datetime(self.time_s).month/12)
            fields.append("Month")
        if self.state_content & StateContent.HOUR:
            h = self.solar.get_datetime(self.time_s).hour #0-23
            arr.append(h/23)
            fields.append("Hour")
        if self.state_content & StateContent.MINUTE:
            m = self.solar.get_datetime(self.time_s).minute #0-59
            arr.append(m/59)
            fields.append("Minute")
        if self.state_content & StateContent.HOUR_MINUTE:
            h = self.solar.get_datetime(self.time_s).hour #0-23
            m = self.solar.get_datetime(self.time_s).minute #0-59
            h+=m/60
            sin_h = np.sin(h / 24.0)
            cos_h = np.cos(h / 24.0)
            arr.append(sin_h)
            arr.append(cos_h)
            fields.append("Sin(Hour Minute)")
            fields.append("Cos(Hour Minute)")
        if self.state_content & StateContent.DAY:
            arr.append(self.solar.get_datetime(self.time_s).timetuple().tm_yday/366)
            fields.append("Day")
        if self.state_content & StateContent.NEXT_DAY:
            day = self.solar.get_datetime(self.time_s).timetuple().tm_yday
            next_day_sun = self.solar.get_day_avg_w(day+1)/self.max_avg_day
            arr.append(next_day_sun)
            fields.append("Next day")
        if self.state_content & StateContent.DAY_AVG:
            day = self.solar.get_datetime(self.time_s).timetuple().tm_yday
            day_sun = self.solar.get_day_avg_w(day)/self.max_avg_day
            arr.append(day_sun)
            fields.append("Day avg")
        if self.state_content & StateContent.SUN_REAL_PREDICTION:
            energy_j = self.solar.get_real_future_prediction_j(self.time_s+when_m*60, self.step_size_s, windows_size_m)
            if windows_size_m > 0:
                energy_j /= self.solar.max_power_w*windows_size_m*60
            else:
                energy_j = 0.0
            arr.append(energy_j)
            fields.append("Sun real prediction")
        if self.state_content & StateContent.SUN_ESTIMATE_PREDICTION:
            lower_bound_energy_j, upper_bound_energy_j = self.solar.get_estimate_future_prediction_j(self.time_s+when_m*60, self.step_size_s, windows_size_m)
            # Normalize
            if windows_size_m > 0:
                lower_bound_energy_j /= self.solar.max_power_w*windows_size_m*60
                upper_bound_energy_j /= self.solar.max_power_w*windows_size_m*60
            else:
                lower_bound_energy_j = 0
                upper_bound_energy_j = 0
            arr.append(lower_bound_energy_j)
            arr.append(upper_bound_energy_j)
            fields.append("Sun estimate prediction ub")
            fields.append("Sun estimate prediction lb")
        if self.state_content & StateContent.SUN_ESTIMATE_SINGLE_PREDICTION:
            energy_j = self.solar.get_estimate_future_single_prediction_j(self.time_s+when_m*60, self.step_size_s, windows_size_m)
            if windows_size_m > 0:
                energy_j /= self.solar.max_power_w*windows_size_m*60
            else:
                energy_j = 0.0
            arr.append(energy_j)
            fields.append("Energy prediction")
        if self.state_content & StateContent.PRESSURE:
            val = self.solar.get_info(self.time_s,"pressure")
            arr.append(val/1000)
            fields.append("Pressure")
        if self.state_content & StateContent.HUMIDITY:
            val = self.solar.get_info(self.time_s,"humidity")
            arr.append(val/100)
            fields.append("Humidity")
        if self.state_content & StateContent.CLOUD:
            val = self.solar.get_info(self.time_s,"cloud_opacity")
            arr.append(val/100)
            fields.append("Cloud opacity")
        if self.state_content & StateContent.BUFFER:
            arr.append(self.buffer_length/self.max_buffer_size)
            fields.append("Memory")
        if self.state_content & StateContent.IMAGES:
            if self.solar.get_solar_w(self.time_s) > 0:
                arr.append((self.acquisition_speed_fps*self.step_size_s)/(self.acquisition_speed_fps*self.step_size_s))
            else:
                arr.append(0)
            fields.append("Images")
        if self.state_content & StateContent.SUNSET_TIME:
            sunset_time = self.solar.get_sunset_time(self.time_s, self.step_size_s)
            h = self.solar.get_datetime(sunset_time).hour #0-23
            m = self.solar.get_datetime(sunset_time).minute #0-59
            h+=m/60
            arr.append(h/24)
            fields.append("Sunset")
        if self.state_content & StateContent.EMBEDDED_CURRENT_DAY:
            day = self.solar.get_datetime(self.time_s).timetuple().tm_yday
            values = self.solar.get_day_values(day)
            print("Values",len(values))
            #TODO values are 228, cannot fit into 128
            while len(values) < self.forecast_steps:
                values.insert(0,0)
        if self.state_content & StateContent.EMBEDDED_NEXT_DAY or \
            self.state_content & StateContent.EMBEDDED_PREV_NEXT_DAY or \
            self.state_content & StateContent.QUANTIZED_PREV_DAY:
            today = self.solar.get_datetime(self.time_s).day
            # def f(t):
            #     #forecast_time_s = self.time_s+self.step_size_s*i
            #     today = self.solar.get_datetime(self.time_s).day
            #     if self.solar.get_datetime(t).day <= today: 
            #         return self.solar.get_solar_w(t)/self.solar.max_power_w
            #     else:
            #         return 0
            # times = list(range(self.time_s, self.time_s+self.step_size_s*128, self.step_size_s))
            # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exec:
            #     values = list(exec.map(f, times))
            values = []
            for i in range(self.forecast_steps):
                forecast_time_s = self.time_s+self.step_size_s*i
                if self.solar.get_datetime(forecast_time_s).day <= today: 
                    v = self.solar.get_solar_w(forecast_time_s)
                    v /= self.solar.max_power_w
                else:
                    v = 0
                values.append(v)
            #values = signal_noise(values, strength=self.prevision_noise_amount, max_value=max(values), rng=self.rng)
        
        if self.state_content & StateContent.EMBEDDED_CURRENT_DAY or \
              self.state_content & StateContent.EMBEDDED_NEXT_DAY or \
              self.state_content & StateContent.EMBEDDED_PREV_NEXT_DAY:

            for index,l in enumerate(values):
                arr.append(l)
                fields.append(f"Latent {index}")
        
        if self.state_content & StateContent.QUANTIZED_DAY or \
              self.state_content & StateContent.QUANTIZED_PREV_DAY:
            values = np.array(values, dtype=np.float32)
            if len(values) % self.latent_size != 0:
                raise Exception(f"Original space ({len(values)}) is not divisible by the latent space size {self.latent_size}")
            splits = int(self.forecast_steps/self.latent_size)
            values = values.reshape(-1, splits).mean(axis=1)
            for index,v in enumerate(values):
                arr.append(v)
                fields.append(f"Latent {index}")

        return np.array(arr,dtype=np.float32),fields


    def get_uptime_s(self) -> int:
        return self.time_s-self.boot_time_s

    def get_human_uptime(self) -> str:
        days = self.get_uptime_s() // 86400
        output = f"{days} days, "
        output += time.strftime("%H hours, %M minutes, %S seconds",
                                time.gmtime(self.get_uptime_s()))
        return output

