#Load congress speeches
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

import pandas as pd

from political_speech.utils import (
    load_config,
    parse_date,
    read_json,
    extract_date_from_crec_folder,
    extract_speeches_from_json,
    should_keep_by_chamber,
)

def main():
    root = PROJECT_ROOT
    config = load_config(root)

    date_format = config["sample"]["date_format"]
    start_date = parse_date(config["sample"]["start_date"], date_format)
    end_date = parse_date(config["sample"]["end_date"], date_format)

    chamber_setting = config["sample"].get("chamber", "both")
    include_extensions = config["sample"].get(
        "include_extensions_of_remarks",
        False
    )

    print("Config loaded from:", root / "config" / "config.yaml")
    print("include_extensions_of_remarks:", include_extensions, type(include_extensions))
    print("chamber_setting:", chamber_setting)
    print("start_date:", start_date)
    print("end_date:", end_date)

    raw_root = root / config["paths"]["raw"] / "congress_speeches"
    interim_dir = root / config["paths"]["interim"]
    logs_dir = root / config["paths"]["logs"]

    interim_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw data folder does not exist: {raw_root}")

    rows = []
    errors = []

    year_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])

    for year_dir in year_dirs:
        crec_dirs = sorted([p for p in year_dir.iterdir() if p.is_dir()])

        for crec_dir in crec_dirs:
            date_string = extract_date_from_crec_folder(crec_dir.name)

            if date_string is None:
                continue

            current_date = parse_date(date_string, date_format)

            if current_date < start_date or current_date > end_date:
                continue

            json_dir = crec_dir / "json"

            if not json_dir.exists():
                errors.append({
                    "date": date_string,
                    "folder": str(crec_dir),
                    "error": "json folder missing",
                })
                continue

            json_files = sorted(json_dir.glob("*.json"))

            for json_file in json_files:
                try:
                    data = read_json(json_file)
                    speech_rows = extract_speeches_from_json(data)

                    for speech_row in speech_rows:
                        speech_row.update({
                            "date": date_string,
                            "crec_folder": crec_dir.name,
                            "json_file": json_file.name,
                            "json_path": str(json_file.relative_to(root)),
                        })

                        if not include_extensions and speech_row.get("extension") is True:
                            continue

                        if should_keep_by_chamber(speech_row, chamber_setting):
                            rows.append(speech_row)

                except Exception as e:
                    errors.append({
                        "date": date_string,
                        "folder": str(crec_dir),
                        "file": str(json_file),
                        "error": repr(e),
                    })

    speeches_df = pd.DataFrame(rows)
    errors_df = pd.DataFrame(errors)

    output_path = interim_dir / "speeches_raw.parquet"
    error_path = logs_dir / "raw_json_errors.csv"
    
    speeches_df.to_parquet(output_path, index=False)

    if len(errors_df) > 0:
        errors_df.to_csv(error_path, index=False)

    print("Done.")
    print(f"Found speeches: {len(speeches_df):,}")
    print(f"Saved speeches to: {output_path}")

    if len(errors_df) > 0:
        print(f"Errors/warnings: {len(errors_df):,}")
        print(f"Saved errors to: {error_path}")

if __name__ == "__main__":
    main()