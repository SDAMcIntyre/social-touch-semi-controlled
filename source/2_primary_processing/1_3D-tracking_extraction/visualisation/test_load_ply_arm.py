import os

# PASTE THE EXACT, FULL, ABSOLUTE PATH YOU ARE USING HERE
absolute_path = "F:\\OneDrive - Linköpings universitet\\_Teams\\Social touch Kinect MNG\\data\\semi-controlled\\1_primary\\kinect\\2022-06-14_ST13-01\\NoIR_2022-06-14_15-42-43_controlled-touch-MNG_ST13_1_block1-arm_only.ply"

print(f"--- Testing File Access ---")
print(f"Testing path: {absolute_path}")

# Check 1: Existence and Type
print(f"Path exists: {os.path.exists(absolute_path)}")
print(f"Is a file: {os.path.isfile(absolute_path)}")

# Check 2: Permissions
print(f"Has read permission for this script: {os.access(absolute_path, os.R_OK)}")

# Check 3: The Ultimate Test - Can we open it?
print("\\nAttempting to open with basic Python I/O...")
try:
    with open(absolute_path, 'rb') as f: # 'rb' = read binary
        print("✅ SUCCESS: File was opened successfully by basic Python.")
        # Try to read the first few bytes to be absolutely sure
        first_bytes = f.read(16)
        print(f"First 16 bytes: {first_bytes}")
except Exception as e:
    print(f"❌ FAILED: Could not open/read the file with basic Python.")
    print(f"Error details: {e}")

print("--- Test Complete ---")