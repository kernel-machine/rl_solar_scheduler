import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = [
    "period_end",
    "gti",
    "cloud_opacity",
    "relative_humidity",
    "surface_pressure",
]


def filter_solar_csv(input_path: str | Path, output_path: str | Path, columns: list[str] | None = None) -> None:
    selected_columns = columns or REQUIRED_COLUMNS

    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open(newline="") as source_file:
        reader = csv.DictReader(source_file)

        missing_columns = [column for column in selected_columns if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {input_path}: {', '.join(missing_columns)}"
            )

        with output_path.open("w", newline="") as destination_file:
            writer = csv.DictWriter(destination_file, fieldnames=selected_columns)
            writer.writeheader()

            for row in reader:
                writer.writerow({column: row[column] for column in selected_columns})


def build_output_path(input_path: str | Path) -> Path:
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}_solar_only{input_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a reduced CSV containing only the columns used by src/lib/solar/solar.py."
    )
    parser.add_argument("input_csv", help="Source CSV file")
    parser.add_argument(
        "output_csv",
        nargs="?",
        help="Destination CSV file. Defaults to <input>_solar_only.csv",
    )

    args = parser.parse_args()
    output_path = args.output_csv or build_output_path(args.input_csv)
    filter_solar_csv(args.input_csv, output_path)


if __name__ == "__main__":
    main()