import csv
import random

# Step 1: Define the ranges for weights and heights
cat_weight_range = (3.0, 5.5)  # in kilograms
cat_height_range = (20, 30)    # in centimeters
dog_weight_range = (8.0, 30.0)  # in kilograms
dog_height_range = (45, 75)     # in centimeters

# Generate the dataset
num_entries = 2000  # total number of entries
data = []

for _ in range(num_entries // 2):
    # Generate cats data
    weight = round(random.uniform(*cat_weight_range), 1)
    height = random.randint(*cat_height_range)
    data.append([0, weight, height])
    
    # Generate dogs data
    weight = round(random.uniform(*dog_weight_range), 1)
    height = random.randint(*dog_height_range)
    data.append([1, weight, height])

# Step 2: Calculate the means of weights and heights
weights = [row[1] for row in data]
heights = [row[2] for row in data]

mean_weight = sum(weights) / len(weights)
mean_height = sum(heights) / len(heights)

# Step 3: Center the weights and heights
centered_data = []
for row in data:
    centered_weight = row[1] - mean_weight
    centered_height = row[2] - mean_height
    centered_data.append([row[0], centered_weight, centered_height])

# Step 4: Write the centered data to a new CSV file
with open('centered_cats_and_dogs.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['type', 'weight', 'height'])
    writer.writerows(centered_data)

print("CSV file 'centered_cats_and_dogs.csv' has been created with centered data.")