import re
import subprocess
import shutil
import os
import difflib

SOURCE_FILE = "image_caption_model.py"
TEMP_FILE = "testdup.py"
TEST_FILE = "test_regression.py"

# Mutations we’ll simulate (operator changes)
MUTATIONS = [
    ("==", "!="),
    (">", "<"),
    ("<", ">"),
    ("and", "or"),
    ("True", "False")
]

def run_pytest(file_under_test):
    """Run pytest on the mutated file and return True if all tests pass (mutant survived) or False if tests failed (mutant killed)."""
    # Copy mutated file to replace the source temporarily
    shutil.copyfile(file_under_test, SOURCE_FILE)
    result = subprocess.run(["pytest", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0, result.stdout.decode()

def mutate_and_test():
    with open(SOURCE_FILE, "r") as f:
        original_code = f.read()

    print("🔬 Starting Mutation Testing Simulation\n" + "="*60)
    mutant_index = 1
    killed = 0
    survived = 0

    for pattern, replacement in MUTATIONS:
        if pattern not in original_code:
            continue

        mutated_code = re.sub(pattern, replacement, original_code, count=1)
        with open(TEMP_FILE, "w") as f:
            f.write(mutated_code)

        print(f"\n🧬 Mutant #{mutant_index}: replaced '{pattern}' → '{replacement}'")
        success, output = run_pytest(TEMP_FILE)

        if success:
            print("❌ Mutant survived — test suite did NOT catch the change.")
            survived += 1
        else:
            print("✅ Mutant killed — tests detected the change.")
            killed += 1

        mutant_index += 1

    # Restore the original file
    with open(SOURCE_FILE, "w") as f:
        f.write(original_code)
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

    print("\n" + "="*60)
    print(f"Total mutants tested: {mutant_index-1}")
    print(f"Killed: {killed}")
    print(f"Survived: {survived}")
    score = (killed / (killed + survived) * 100) if (killed + survived) else 0
    print(f"Mutation Score: {score:.2f}%")
    print("="*60)

if __name__ == "__main__":
    mutate_and_test()
