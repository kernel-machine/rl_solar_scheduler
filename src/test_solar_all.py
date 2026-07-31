from lib.solar.solar import Solar
panel_area_m2 = 2*0.55*0.51
efficiency = 0.1426
max_power_w = 80
solar = Solar("../solcast2025_solar_only.csv", scale_factor=panel_area_m2*efficiency, max_power=max_power_w, enable_cache=False)

hard_days = 0
for day in range(350):
    total_energy_wh = 0
    for hour in range(24):
        for minute in range(0, 60, 5):
            time_s = (day * 24 * 60 * 60) + (hour * 60 * 60) + (minute * 60)
            power_w = solar.get_solar_w(time_s)
            if power_w > 0:
                total_energy_wh += power_w * (5 / 60)
    if total_energy_wh < 48:
        hard_days += 1
print(f"Number of days with < 48 Wh: {hard_days}")
