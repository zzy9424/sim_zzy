import random

random_numbers = []

for i in range(20):
    if i % 2 == 0:
        random_numbers.append(round(random.random(), 2))
    else:
        random_numbers.append(round(-random.random(), 2))

print(random_numbers)