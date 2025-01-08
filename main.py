import os
import csv
import json
import re
from dotenv import load_dotenv

# --- Import your custom classes/modules ---
from src.instructions_creation.file_importer import DocumentImporter
from src.instructions_creation.text_separator import TextSeparatorRunner
from src.instructions_creation.intructions_id_finder import AssistantDocFinder
from src.assistant_creator.assistant_creator import AssistantCreator
from src.assistant_finetuner.examples_to_jsonl import TxtToJsonlConverter
from src.assistant_finetuner.create_finetune_model import OpenAIFineTuner
from src.assistant_finetuner.upload_jsonl import OpenAIFileUploader
from src.assistant_testing.static_test_creator import StaticExamplesTestCreator
from src.assistant_testing.static_assistant_tester import StaticAssistantsRunner
from src.assistant_testing.static_grader_results import FileManagerGrader

# --- Import parameters from your parameters.py ---
from parameters import (
    # Assistant/model info
    ASSISTANT_NAME,
    BASE_MODEL_NAME,
    BASE_MODEL_SUFFIX,
    BASE_TEMPERATURE,
    BASE_TOP_P,

    # Fine-tuning
    NUM_WORST_EXAMPLES,
    FINE_TUNED_MODEL_SUFFIX,

    # Evaluator
    EVALUATOR_MODEL_NAME,
    EVALUATOR_TEMPERATURE,
    EVALUATOR_TOP_P,

    # CSV columns
    COLUMN_QUESTION,
    COLUMN_HUMAN_ANSWER,

    # Instructions/Examples
    PATH_INSTRUCTIONS_TXT,
    PATH_INSTRUCTIONS_NO_EXAMPLES,
    PATH_EXAMPLES_TXT,

    # Assistant IDs
    PATH_ASSISTANTS_IDS_TXT,
    PATH_ASSISTANT_ID_FINE_TUNED_TXT,
    PATH_INSTRUCTIONS_EVALUATOR_TXT,
    PATH_EVALUATOR_ID_TXT,

    # Test inputs
    PATH_TEST_EXAMPLES_CSV,

    # Separate answers & grades
    PATH_BASE_ANSWERS_CSV,
    PATH_BASE_GRADES_CSV,
    PATH_FINE_TUNED_ANSWERS_CSV,
    PATH_FINE_TUNED_GRADES_CSV,
    PATH_UNIFIED_RESULTS_CSV,

    # Worst Qs
    PATH_WORST_QUESTIONS_TXT,
    PATH_WORST_QUESTIONS_JSONL,
)


class Main:
    def __init__(self):
        load_dotenv()

        # -------------------------------------------------
        # Basic assistant & model config
        # -------------------------------------------------
        self.assistant_name = ASSISTANT_NAME
        self.base_model_name = BASE_MODEL_NAME
        self.base_model_suffix = BASE_MODEL_SUFFIX
        self.base_temperature = BASE_TEMPERATURE
        self.base_top_p = BASE_TOP_P

        # Fine-tuning
        self.num_worst_examples = NUM_WORST_EXAMPLES
        self.fine_tuned_model_suffix = FINE_TUNED_MODEL_SUFFIX

        # Evaluator
        self.evaluator_model_name = EVALUATOR_MODEL_NAME
        self.evaluator_temperature = EVALUATOR_TEMPERATURE
        self.evaluator_top_p = EVALUATOR_TOP_P

        # -------------------------------------------------
        # Local file paths
        # -------------------------------------------------
        self.path_instructions_txt = PATH_INSTRUCTIONS_TXT
        self.path_instructions_no_examples = PATH_INSTRUCTIONS_NO_EXAMPLES
        self.path_examples_txt = PATH_EXAMPLES_TXT
        self.path_assistants_ids_txt = PATH_ASSISTANTS_IDS_TXT
        self.path_assistant_id_fine_tuned_txt = PATH_ASSISTANT_ID_FINE_TUNED_TXT
        self.path_instructions_evaluator_txt = PATH_INSTRUCTIONS_EVALUATOR_TXT
        self.path_evaluator_id_txt = PATH_EVALUATOR_ID_TXT

        # Test sets
        self.path_test_examples_csv = PATH_TEST_EXAMPLES_CSV

        # Separate answers & grades
        self.path_base_answers_csv = PATH_BASE_ANSWERS_CSV
        self.path_base_grades_csv = PATH_BASE_GRADES_CSV
        self.path_fine_tuned_answers_csv = PATH_FINE_TUNED_ANSWERS_CSV
        self.path_fine_tuned_grades_csv = PATH_FINE_TUNED_GRADES_CSV
        self.path_unified_results_csv = PATH_UNIFIED_RESULTS_CSV

        # Worst questions
        self.path_worst_questions_txt = PATH_WORST_QUESTIONS_TXT
        self.path_worst_questions_jsonl = PATH_WORST_QUESTIONS_JSONL

        # -------------------------------------------------
        # Credentials (environment variables)
        # -------------------------------------------------
        self.service_account_path = os.getenv("SERVICE_ACCOUNT_FILE")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.separator_assistant_id = os.getenv("ID_ASSISTANT_TEXT_SEPARATOR")

        # -------------------------------------------------
        # Tools
        # -------------------------------------------------
        self.fine_tuner = OpenAIFineTuner(api_key=self.openai_api_key)
        self.static_test_creator = StaticExamplesTestCreator(
            input_test_file=self.path_examples_txt,
            output_test_file=self.path_test_examples_csv
        )

    # -------------------------------------------------------------------------
    # 1) CREATE INSTRUCTIONS & SEPARATE EXAMPLES
    # -------------------------------------------------------------------------
    def find_doc_id(self):
        finder = AssistantDocFinder()
        _, gdocs_address = finder.get_doc_id_by_assistant_name(self.assistant_name)
        self.document_id = gdocs_address
        print(f"Document ID for '{self.assistant_name}' found: {self.document_id}")

    def import_text_from_google_doc(self):
        importer = DocumentImporter(
            self.service_account_path,
            self.document_id,
            self.path_instructions_txt
        )
        importer.import_text()
        print("Text from Google Doc imported successfully.")

    def separate_text(self):
        separator_runner = TextSeparatorRunner(
            api_key=self.openai_api_key,
            assistant_id=self.separator_assistant_id
        )
        separator_runner.run()
        print("Text separation completed: instructions vs. examples.")

    def create_instructions(self):
        self.find_doc_id()
        self.import_text_from_google_doc()
        self.separate_text()

    # -------------------------------------------------------------------------
    # 2) CREATE THE TEST WITH QUESTIONS & HUMAN ANSWERS
    # -------------------------------------------------------------------------
    def create_static_tests(self):
        self.static_test_creator.create_test()
        print(f"Static test CSV created at: {self.path_test_examples_csv}")

    # -------------------------------------------------------------------------
    # 3) CREATE BASE ASSISTANT
    # -------------------------------------------------------------------------
    def create_base_assistant(self):
        assistant_creator = AssistantCreator(
            api_key=self.openai_api_key,
            instructions_path=self.path_instructions_txt
        )
        self.base_assistant = assistant_creator.create_assistant(
            name_suffix=self.base_model_suffix,
            model=self.base_model_name,
            tools=[],
            temperature=self.base_temperature,
            top_p=self.base_top_p
        )
        self._save_assistant_id(
            self.base_assistant.name,
            self.base_assistant.id,
            self.path_assistants_ids_txt
        )
        print(f"Base assistant created: {self.base_assistant.name}")

    def _save_assistant_id(self, assistant_name, new_assistant_id, path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{assistant_name, new_assistant_id}\n")

    # -------------------------------------------------------------------------
    # 4) GET BASE ANSWERS (store them in PATH_BASE_ANSWERS_CSV)
    # -------------------------------------------------------------------------
    def get_base_assistant_answers(self):
        runner = StaticAssistantsRunner(
            openai_api_key=self.openai_api_key,
            txt_file_path=self.path_assistants_ids_txt,
            csv_file_path=self.path_test_examples_csv,
            output_csv_path=self.path_base_answers_csv
        )
        runner.run_all()
        print(f"Base assistant answers stored in: {self.path_base_answers_csv}")

    # -------------------------------------------------------------------------
    # 5) CREATE EVALUATOR ASSISTANT
    # -------------------------------------------------------------------------
    def create_evaluator_assistant(self):
        assistant_creator = AssistantCreator(
            api_key=self.openai_api_key,
            instructions_path=self.path_instructions_evaluator_txt
        )
        self.evaluator_assistant = assistant_creator.create_assistant(
            name_suffix="static_evaluator",
            model=self.evaluator_model_name,
            tools=[],
            temperature=self.evaluator_temperature,
            top_p=self.evaluator_top_p
        )
        self._save_assistant_id(
            self.evaluator_assistant.name,
            self.evaluator_assistant.id,
            self.path_evaluator_id_txt
        )
        print(f"Evaluator assistant created: {self.evaluator_assistant.name}")

    def _extract_assistant_id_from_file(self, path):
        pattern = re.compile(r"\('([^']+)',\s*'([^']+)'\)")
        assistant_id = None
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    match = pattern.match(line)
                    if match:
                        assistant_id = match.group(2)
                        break
        return assistant_id

    # -------------------------------------------------------------------------
    # 6) GRADE BASE ANSWERS => store grades in PATH_BASE_GRADES_CSV
    # -------------------------------------------------------------------------
    def grade_base_assistant_responses(self):
        evaluator_id = self._extract_assistant_id_from_file(self.path_evaluator_id_txt)
        grader = FileManagerGrader(
            openai_api_key=self.openai_api_key,
            assistant_id=evaluator_id,
            csv_input_path=self.path_base_answers_csv
        )
        base_answer_col = f"{self.assistant_name}_{self.base_model_suffix}"
        grader.run(
            question_column=COLUMN_QUESTION,
            human_answer_column=COLUMN_HUMAN_ANSWER,
            machine_answer_column=base_answer_col,
            output_csv_path=self.path_base_grades_csv
        )
        print(f"Base assistant responses graded. Results saved in: {self.path_base_grades_csv}")

    # -------------------------------------------------------------------------
    # 7) GATHER WORST QUESTIONS (lowest grades)
    # -------------------------------------------------------------------------
    def gather_worst_indices(self, worst_n=None):
        """
        Return the row indices of the 'worst' (lowest-grade) examples from path_base_grades_csv.
        """
        if worst_n is None:
            worst_n = self.num_worst_examples

        if not os.path.exists(self.path_base_grades_csv):
            print(f"No base grades found at {self.path_base_grades_csv}.")
            return []

        # We'll read the entire grades CSV
        with open(self.path_base_grades_csv, 'r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))  # materialize in list

        # Sort by 'grade' ascending (lowest is worst)
        # Keep track of each row's index
        indexed_rows = list(enumerate(reader))  # (idx, row_dict)
        sorted_indexed_rows = sorted(
            indexed_rows, 
            key=lambda x: float(x[1].get('grade', 0))
        )

        # Take the first 'worst_n' items
        worst_indexed = sorted_indexed_rows[:worst_n]

        # Return just the indices
        worst_indices = [item[0] for item in worst_indexed]
        print(f"Worst {len(worst_indices)} row indices: {worst_indices}")
        return worst_indices

    def create_worst_questions_file(self, worst_indices):
        """
        Reads path_base_answers_csv, extracts question/human_answer for each 
        row index in worst_indices, and saves them as a JSON array:
        [
          {"Q": "...", "A": "..."},
          {"Q": "...", "A": "..."}
        ]
        in path_worst_questions_txt.
        """
        if not worst_indices:
            print("No worst indices found. Skipping creation of worst questions file.")
            return

        # Read the base answers CSV
        with open(self.path_base_answers_csv, 'r', encoding='utf-8') as f_ans:
            ans_rows = list(csv.DictReader(f_ans))

        data_list = []
        for idx in worst_indices:
            # If idx is within range:
            if 0 <= idx < len(ans_rows):
                row = ans_rows[idx]
                question = row.get(COLUMN_QUESTION, "").strip()
                answer = row.get(COLUMN_HUMAN_ANSWER, "").strip()
                data_list.append({"Q": question, "A": answer})

        # Write out to path_worst_questions_txt
        with open(self.path_worst_questions_txt, "w", encoding="utf-8") as f_out:
            json.dump(data_list, f_out, ensure_ascii=False, indent=2)

        print(f"Worst questions saved in JSON format to {self.path_worst_questions_txt}")

    # -------------------------------------------------------------------------
    # 8, 9, 10) Convert .txt to .jsonl, upload, fine-tune
    # -------------------------------------------------------------------------
    def convert_worst_txt_to_jsonl(self):
        converter = TxtToJsonlConverter(
            input_examples_txt_path=self.path_worst_questions_txt,
            input_prompt_txt=self.path_instructions_no_examples,
            output_jsonl_path=self.path_worst_questions_jsonl
        )
        converter.convert()
        print(f"Converted {self.path_worst_questions_txt} to {self.path_worst_questions_jsonl}")

    def upload_worst_jsonl(self):
        uploader = OpenAIFileUploader(api_key=self.openai_api_key)
        response = uploader.upload_file(
            file_path=self.path_worst_questions_jsonl,
            purpose="fine-tune"
        )
        print("Worst questions JSONL uploaded successfully.")
        return response.id

    def create_fine_tuning_job(self, fine_tune_file_id):
        fine_tune_job_response = self.fine_tuner.create_fine_tuning_job(
            training_file_id=fine_tune_file_id,
            model=self.base_model_name,
            suffix=f"{self.assistant_name}_{self.fine_tuned_model_suffix}",
        )
        fine_tune_job_id = fine_tune_job_response.id
        self.fine_tune_model = self.fine_tuner.monitor_fine_tuning_job(fine_tune_job_id)
        print(f"Fine-tuned model is ready: {self.fine_tune_model}")

    def create_fine_tuned_assistant(self):
        assistant_creator = AssistantCreator(
            api_key=self.openai_api_key,
            instructions_path=self.path_instructions_txt
        )
        self.new_fine_tuned_assistant = assistant_creator.create_assistant(
            name_suffix=self.fine_tuned_model_suffix,
            model=self.fine_tune_model,
            tools=[],
            temperature=self.base_temperature,
            top_p=self.base_top_p
        )
        self._save_assistant_id(
            self.new_fine_tuned_assistant.name,
            self.new_fine_tuned_assistant.id,
            self.path_assistant_id_fine_tuned_txt
        )
        print(f"Fine-tuned assistant created: {self.new_fine_tuned_assistant.name}")

    def fine_tune_new_assistant_workflow(self):
        """
        Helper method that uses the steps:
         - convert_worst_txt_to_jsonl
         - upload_worst_jsonl
         - create_fine_tuning_job
         - create_fine_tuned_assistant
        """
        self.convert_worst_txt_to_jsonl()
        file_id = self.upload_worst_jsonl()
        self.create_fine_tuning_job(file_id)
        self.create_fine_tuned_assistant()

    # -------------------------------------------------------------------------
    # 11) GET FINE-TUNED ANSWERS => store them in PATH_FINE_TUNED_ANSWERS_CSV
    # -------------------------------------------------------------------------
    def get_fine_tuned_assistant_answers(self):
        runner = StaticAssistantsRunner(
            openai_api_key=self.openai_api_key,
            txt_file_path=self.path_assistant_id_fine_tuned_txt,
            csv_file_path=self.path_test_examples_csv,
            output_csv_path=self.path_fine_tuned_answers_csv
        )
        runner.run_all()
        print(f"Fine-tuned assistant answers stored in: {self.path_fine_tuned_answers_csv}")

    # -------------------------------------------------------------------------
    # 12) GRADE FINE-TUNED ANSWERS => store in PATH_FINE_TUNED_GRADES_CSV
    # -------------------------------------------------------------------------
    def grade_fine_tuned_assistant_responses(self):
        evaluator_id = self._extract_assistant_id_from_file(self.path_evaluator_id_txt)
        grader = FileManagerGrader(
            openai_api_key=self.openai_api_key,
            assistant_id=evaluator_id,
            csv_input_path=self.path_fine_tuned_answers_csv
        )
        ft_answer_col = f"{self.assistant_name}_{self.fine_tuned_model_suffix}"
        grader.run(
            question_column=COLUMN_QUESTION,
            human_answer_column=COLUMN_HUMAN_ANSWER,
            machine_answer_column=ft_answer_col,
            output_csv_path=self.path_fine_tuned_grades_csv
        )
        print(f"Fine-tuned assistant responses graded. Results in: {self.path_fine_tuned_grades_csv}")

    # -------------------------------------------------------------------------
    # 13) UNIFY RESULTS
    # -------------------------------------------------------------------------
    def unify_results_in_single_csv(self):
        """
        Merge these 4 files by row index:
        - base answers (PATH_BASE_ANSWERS_CSV)
        - base grades (PATH_BASE_GRADES_CSV)
        - fine-tuned answers (PATH_FINE_TUNED_ANSWERS_CSV)
        - fine-tuned grades (PATH_FINE_TUNED_GRADES_CSV)
        into PATH_UNIFIED_RESULTS_CSV.
        """
        out_file = self.path_unified_results_csv

        # 1. Ensure all files exist
        required_files = [
            self.path_base_answers_csv,
            self.path_base_grades_csv,
            self.path_fine_tuned_answers_csv,
            self.path_fine_tuned_grades_csv,
        ]
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                return

        # 2. Read each file into a list of rows
        with open(self.path_base_answers_csv, "r", encoding="utf-8") as f:
            base_answers = list(csv.DictReader(f))

        with open(self.path_base_grades_csv, "r", encoding="utf-8") as f:
            base_grades = list(csv.DictReader(f))

        with open(self.path_fine_tuned_answers_csv, "r", encoding="utf-8") as f:
            fine_tuned_answers = list(csv.DictReader(f))

        with open(self.path_fine_tuned_grades_csv, "r", encoding="utf-8") as f:
            fine_tuned_grades = list(csv.DictReader(f))

        # 3. Combine rows by index
        combined_rows = []
        for idx in range(len(base_answers)):
            base_row = base_answers[idx] if idx < len(base_answers) else {}
            base_grade_row = base_grades[idx] if idx < len(base_grades) else {}
            fine_tuned_row = fine_tuned_answers[idx] if idx < len(fine_tuned_answers) else {}
            fine_tuned_grade_row = fine_tuned_grades[idx] if idx < len(fine_tuned_grades) else {}

            combined_rows.append({
                "question": base_row.get(COLUMN_QUESTION, ""),
                "human_response": base_row.get(COLUMN_HUMAN_ANSWER, ""),
                f"{self.assistant_name}_base_answer": base_row.get(f"{self.assistant_name}_base", ""),
                f"{self.assistant_name}_base_grade": base_grade_row.get("grade", ""),
                f"{self.assistant_name}_fine_tuned_answer": fine_tuned_row.get(f"{self.assistant_name}_fine_tuned_with_worst", ""),
                f"{self.assistant_name}_fine_tuned_grade": fine_tuned_grade_row.get("grade", ""),
            })

        # 4. Write the unified CSV
        fieldnames = [
            "question",
            "human_response",
            f"{self.assistant_name}_base_answer",
            f"{self.assistant_name}_base_grade",
            f"{self.assistant_name}_fine_tuned_answer",
            f"{self.assistant_name}_fine_tuned_grade",
        ]
        with open(out_file, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)

        print(f"Unified CSV created at: {out_file}")

    # -------------------------------------------------------------------------
    # RUN: MAIN WORKFLOW
    # -------------------------------------------------------------------------
    def run(self):
        """
        Steps:
          1) create_instructions()
          2) create_static_tests()
          3) create_base_assistant()
          4) get_base_assistant_answers()
          5) create_evaluator_assistant()
          6) grade_base_assistant_responses()
          7) gather worst questions indices + create worst questions file
          8) convert worst txt to JSONL
          9) upload JSONL
          10) create fine-tuned model
          11) get fine-tuned answers
          12) grade fine-tuned answers
          13) unify CSV
        """
                # 1) Create instructions & separate examples
        # self.create_instructions()

        # 2) Create the test CSV
        #self.create_static_tests()

        # 3) Create base assistant
        #self.create_base_assistant()

        # 4) Get base answers
        #self.get_base_assistant_answers()

        # 5) Create evaluator
        #self.create_evaluator_assistant()

        # 6) Grade base answers
        #self.grade_base_assistant_responses()

        # 7) Gather worst questions
        #worst_questions = self.gather_worst_indices()
        #self.create_worst_questions_file(worst_questions)

        # 8,9,10) Fine-tune model + create fine-tuned assistant
        #self.fine_tune_new_assistant_workflow()

        # 11) Get fine-tuned answers
        #self.get_fine_tuned_assistant_answers()

        # 12) Grade fine-tuned answers
        #self.grade_fine_tuned_assistant_responses()

        # 13) Unify results
        self.unify_results_in_single_csv()

        print("Done!")


if __name__ == "__main__":
    main_app = Main()
    main_app.run()