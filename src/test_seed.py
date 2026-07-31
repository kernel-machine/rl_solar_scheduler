import random
rng = random.Random(42)
print(rng.randint(3,7)/10) # from reset() line 131
print(rng.randint(0,350)) # line 133
print(rng.randint(0, 3*90000//4)) # line 134
print(rng.randint(0,23)) # line 148
