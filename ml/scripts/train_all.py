import subprocess
import sys


def run(script):

    result = subprocess.run(
        [sys.executable, script],
        check=True
    )

    return result.returncode


def main():

    print("=== Validating dataset ===")
    run("scripts/validate_dataset.py")

    print("\n=== Preparing dataset ===")
    run("scripts/prepare_dataset.py")

    print("\n=== Training intent model ===")
    run("scripts/train_intent.py")

    print("\n=== Training NER model ===")
    run("scripts/train_ner.py")

    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()