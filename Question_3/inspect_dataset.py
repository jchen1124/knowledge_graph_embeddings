
import csv
from pathlib import Path

# Directory containing the dataset files
DATA_DIR = Path(__file__).resolve().parent / "data"

files = {
    "Training": "annotated_wd_data_train.txt",
    "Validation": "annotated_wd_data_valid.txt",
    "Testing": "annotated_wd_data_test.txt",
}

for split_name, filename in files.items():

    file_path = DATA_DIR / filename

    print(f"\n===== {split_name} Dataset =====")

    count = 0

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            if len(row) != 4:
                raise ValueError(
                    f"Unexpected format on line {count + 1}"
                )

            subject, relation, object_, question = row

            count += 1

            # Display the first 3 examples
            if count <= 3:
                print(f"\nExample {count}")
                print(f"Subject:  {subject}")
                print(f"Relation: {relation}")
                print(f"Object:   {object_}")
                print(f"Question: {question}")

    print(f"\nTotal questions: {count}")
