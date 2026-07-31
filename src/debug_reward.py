def calculate_reward(battery_level):
    processing_reward = 6000 / 9000  # 0.666
    battery_penalty = 0.7 * max(0, 0.2 - battery_level)
    return processing_reward - battery_penalty

print("Battery 100%:", calculate_reward(1.0))
print("Battery  10%:", calculate_reward(0.1))
print("Battery   1%:", calculate_reward(0.01))
