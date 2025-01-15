import json
import csv
from parameters import COLUMN_HUMAN_ANSWER, COLUMN_QUESTION
import re

class StaticExamplesTestCreator:
    def __init__(self, input_test_file, output_test_file):
        self.input_test_file = input_test_file
        self.output_test_file = output_test_file

    def create_test(self):
        # Read the file content
        with open(self.input_test_file, "r", encoding="utf-8") as file:
            raw_content = file.read()

        # Normalize quotes (convert single quotes to double quotes)
        normalized_content = re.sub(r"(?<!\\)'", '"', raw_content)

        try:
            # Parse the normalized content as JSON
            data = json.loads(normalized_content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return

        # Write to the CSV file
        with open(self.output_test_file, "w", newline='', encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            # Write the headers
            writer.writerow([COLUMN_QUESTION, COLUMN_HUMAN_ANSWER])
            
            # Write each question-answer pair 4 times
            for entry in data:
                for _ in range(4):  # Repeat each pair 4 times
                    writer.writerow([entry["Q"], entry["A"]])

        print(f"Base Test file created: {self.output_test_file}")
