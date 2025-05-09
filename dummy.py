import random
import json

# Generate 1000 random samples from the range 1 to 10042
random_samples = random.sample(range(1, 10000), 1000)

# Save the list of random numbers to a JSON file
output_file = "sampled_indexes.json"
with open(output_file, "w") as f:
    json.dump(random_samples, f)

print(f"Generated 1000 random samples and saved to {output_file}.")
