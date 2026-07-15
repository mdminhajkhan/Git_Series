import pandas as pd
import numpy as np
import datetime

def generate_weather_data(seed=42):
    np.random.seed(seed)
    
    # 5 years: 2021-01-01 to 2025-12-31
    start_date = datetime.date(2021, 1, 1)
    end_date = datetime.date(2025, 12, 31)
    delta = end_date - start_date
    num_days = delta.days + 1
    
    dates = [start_date + datetime.timedelta(days=i) for i in range(num_days)]
    
    # We want smooth seasonal base temperatures. 
    # Center days of months (1-indexed for interpolation)
    months_center_days = [15, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350]
    
    # Typical Indian city (e.g., Delhi/North India) seasonal temperature targets:
    # Cold winter, very hot pre-monsoon, warm and humid monsoon (narrower spread), cooling autumn.
    base_min = [10, 13, 18, 23, 26, 27, 26, 25, 24, 19, 14, 11]
    base_max = [21, 25, 32, 38, 41, 35, 32, 31, 32, 32, 28, 23]
    
    # Pad at both ends to handle wrap-around interpolation
    padded_days = [-15] + months_center_days + [380]
    padded_min = [11] + base_min + [10]
    padded_max = [23] + base_max + [21]
    
    data = []
    
    for i, date in enumerate(dates):
        # Day of year (1 to 365/366)
        day_of_year = date.timetuple().tm_yday
        month = date.month
        
        # Interpolated baseline temperatures
        base_min_temp = np.interp(day_of_year, padded_days, padded_min)
        base_max_temp = np.interp(day_of_year, padded_days, padded_max)
        
        # Add random noise to temperatures
        min_temp = base_min_temp + np.random.normal(0, 1.2)
        max_temp = base_max_temp + np.random.normal(0, 1.5)
        
        # Ensure max temp is always higher than min temp
        if max_temp <= min_temp:
            max_temp = min_temp + np.random.uniform(2.0, 5.0)
            
        spread = max_temp - min_temp
        
        # Monsoon factor: June (6), July (7), August (8)
        is_monsoon = month in [6, 7, 8]
        
        # Humidity: inversely correlated with temperature spread, spikes during monsoon.
        # In non-monsoon, humidity is lower (higher spread = lower humidity)
        if is_monsoon:
            # Heavily spiked humidity during monsoon
            humidity_base = 85.0 - 1.0 * spread
            humidity = humidity_base + np.random.normal(0, 4.0)
        else:
            # Seasonal humidity outside monsoon
            humidity_base = 75.0 - 2.2 * spread
            humidity = humidity_base + np.random.normal(0, 6.0)
            
        # Clip humidity to realistic bounds
        humidity = np.clip(humidity, 15.0, 100.0)
        
        # Pressure: Base pressure is higher in winter (~1018 hPa) and lower in summer/monsoon (~1002 hPa)
        # Pressure drops when humidity is high (indicating storm potential)
        pressure_base = 1012.0 + 7.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
        pressure = pressure_base - 0.08 * humidity + np.random.normal(0, 2.0)
        
        # Rainfall Logic:
        # Occurs primarily when humidity > 75% and pressure drops below a seasonal threshold (or overall < 1008 hPa)
        pressure_drop_factor = max(0, 1008.0 - pressure)
        
        # Probability of rain calculation
        if humidity > 75.0:
            # Base probability based on humidity
            p_rain = (humidity - 75.0) / 25.0 * 0.5
            # Increase probability if pressure is low
            p_rain += pressure_drop_factor * 0.15
            
            # Boost during monsoon season
            if is_monsoon:
                p_rain += 0.2
            else:
                p_rain -= 0.1
        else:
            p_rain = 0.02 # very low chance of light shower below 75% humidity
            
        p_rain = np.clip(p_rain, 0.0, 0.95)
        
        # Determine if it rains
        if np.random.rand() < p_rain:
            # Generate rainfall amount
            # During monsoon, rainfall can be very heavy
            if is_monsoon:
                rainfall = np.random.exponential(scale=22.0) + 0.1
            else:
                rainfall = np.random.exponential(scale=8.0) + 0.1
            
            # Round to 1 decimal place
            rainfall = round(rainfall, 1)
            
            # Significant rain causes temperature to drop and pressure to drop further
            if rainfall > 5.0:
                max_temp -= np.random.uniform(1.5, 4.0)
                # Re-verify spread and clip
                if max_temp <= min_temp:
                    max_temp = min_temp + np.random.uniform(1.0, 2.0)
                pressure -= np.random.uniform(1.5, 3.5)
        else:
            rainfall = 0.0
            
        # RainToday is 1 if Rainfall > 0.1mm, else 0
        rain_today = 1 if rainfall > 0.1 else 0
        
        # Round numerical features for clean representation
        min_temp = round(min_temp, 1)
        max_temp = round(max_temp, 1)
        humidity = int(round(humidity))
        pressure = int(round(pressure))
        
        data.append({
            "Date": date.strftime("%Y-%m-%d"),
            "MinTemp": min_temp,
            "MaxTemp": max_temp,
            "Humidity": humidity,
            "Pressure": pressure,
            "Rainfall": rainfall,
            "RainToday": rain_today
        })
        
    df = pd.DataFrame(data)
    df.to_csv("weather_data.csv", index=False)
    print(f"Generated {len(df)} rows of weather data and saved to weather_data.csv.")
    
if __name__ == "__main__":
    generate_weather_data()
