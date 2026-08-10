import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run(script):
    script_path = SCRIPT_DIR / script

    subprocess.run(
        [sys.executable, str(script_path)],
        check=True
    )


def main():
    print("=== Validating dataset ===")
    run("validate_dataset.py")

    print("=== Preparing dataset ===")
    run("prepare_dataset.py")

    print("=== Training ===")
    # run("train_intent.py")
    # run("train_ner.py")


if __name__ == "__main__":
    main()