#Run full pipeline for congress speeches

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


PIPELINE_STEPS = [
    #"01_load_raw_speeches.py",
    #"02_match_speakers.py",
    #"03_preprocess_text.py",
    #"04_build_phrase_counts.py",
    #"05_descriptive_statistics.py",
    "06_penalized_poisson_partisanship.py",
    #"07_inference.py",
    "08_permutation_validation.py",
    #"09_out_of_sample_validation.py",
    "10_full_time_period_poisson.py",
]


def run_step(script_name: str):
    script_path = PROJECT_ROOT / "scripts" / "congress_speeches" / script_name

    print("=" * 80)
    print(f"Running: {script_path}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline stopped because this step failed: {script_name}"
        )

    print(f"Finished: {script_name}\n")


def main():
    print("Project root:", PROJECT_ROOT)

    for script_name in PIPELINE_STEPS:
        run_step(script_name)

    print("=" * 80)
    print("Pipeline finished successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()