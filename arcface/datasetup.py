import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloader import DataLoader


def main():
    loader = DataLoader()
    loader.setup_dataset()
    print("\n--- Step 2: Preprocessing ---")
    os.system(f"python {os.path.join(os.path.dirname(__file__), 'preprocessing.py')}")
    print("\n--- Step 3: Split ---")
    os.system(f"python {os.path.join(os.path.dirname(__file__), 'split_dataset.py')}")
    print("\nAll done.")


if __name__ == "__main__":
    main()
