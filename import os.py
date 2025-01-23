import os
# Define the base command
base_command = "python ./src/main.py --name PSR_J2124-3358 --radius 5 --exp_time 1_000_000 --catalog Xmm_DR13"

# Run the command 100 times
for i in range(1, 101):
    print(f"Running iteration {i}...")
    os.system(base_command)  # Execute the command
    print("Completed iteration {i}.")

print("All iterations completed!")