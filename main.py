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

# --- Import parameters from your new parametros.py ---
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

    # Paths
    PATH_INSTRUCTIONS_TXT,
    PATH_INSTRUCTIONS_NO_EXAMPLES,
    PATH_EXAMPLES_TXT,
    PATH_ASSISTANTS_IDS_TXT,
    PATH_ASSISTANT_ID_FINE_TUNED_TXT,
    PATH_INSTRUCTIONS_EVALUATOR_TXT,
    PATH_EVALUATOR_ID_TXT,
    PATH_TEST_EXAMPLES_CSV,
    PATH_TEST_RESULTS_BASE_CSV,
    PATH_TEST_RESULTS_FINE_TUNED_CSV,
    PATH_TEST_RESULTS_UNIFIED_CSV,
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

        # Test + results
        self.path_test_examples_csv = PATH_TEST_EXAMPLES_CSV
        self.path_test_results_base_csv = PATH_TEST_RESULTS_BASE_CSV
        self.path_test_results_fine_tuned_csv = PATH_TEST_RESULTS_FINE_TUNED_CSV
        self.path_test_results_unified_csv = PATH_TEST_RESULTS_UNIFIED_CSV

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
        """
        Find the Google Doc ID by the assistant's name.
        """
        finder = AssistantDocFinder()
        _, gdocs_address = finder.get_doc_id_by_assistant_name(self.assistant_name)
        self.document_id = gdocs_address
        print(f"Document ID for '{self.assistant_name}' found: {self.document_id}")

    def import_text_from_google_doc(self):
        """
        Import the text from the doc ID into a local TXT file.
        """
        importer = DocumentImporter(
            self.service_account_path,
            self.document_id,
            self.path_instructions_txt
        )
        importer.import_text()
        print("Text from Google Doc imported successfully.")

    def separate_text(self):
        """
        Use a specialized assistant to separate text into instructions and examples.
        Outputs:
            - PATH_INSTRUCTIONS_NO_EXAMPLES
            - PATH_EXAMPLES_TXT
        """
        separator_runner = TextSeparatorRunner(
            api_key=self.openai_api_key,
            assistant_id=self.separator_assistant_id
        )
        separator_runner.run()
        print("Text separation completed: instructions vs. examples.")

    def create_instructions(self):
        """
        Combined step to create instructions and separate the examples.
        """
        self.find_doc_id()
        self.import_text_from_google_doc()
        self.separate_text()

    # -------------------------------------------------------------------------
    # 2) CREATE THE TEST WITH QUESTIONS & HUMAN ANSWERS
    # -------------------------------------------------------------------------
    def create_static_tests(self):
        """
        Take the newly separated examples and turn them into a CSV test file.
        """
        self.static_test_creator.create_test()
        print(f"Static test CSV created at: {self.path_test_examples_csv}")

    # -------------------------------------------------------------------------
    # 3) CREATE BASE ASSISTANT
    # -------------------------------------------------------------------------
    def create_base_assistant(self):
        """
        Read final instructions from local file and create a base assistant.
        """
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
        self.save_assistant_id(
            self.base_assistant.name,
            self.base_assistant.id,
            self.path_assistants_ids_txt
        )
        print(f"Base assistant created: {self.base_assistant.name}")

    def save_assistant_id(self, assistant_name, new_assistant_id, path):
        """
        Append the newly created assistant's name and ID to a local file.
        """
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{assistant_name, new_assistant_id}\n")

    # -------------------------------------------------------------------------
    # 4) RECEIVE & STORE MACHINE ANSWERS (Base Assistant) 
    # -------------------------------------------------------------------------
    def evaluate_base_assistant(self):
        """
        Run the static test using the base assistant. 
        The results (answers) get stored in PATH_TEST_RESULTS_BASE_CSV.
        """
        runner = StaticAssistantsRunner(
            openai_api_key=self.openai_api_key,
            txt_file_path=self.path_assistants_ids_txt,
            csv_file_path=self.path_test_examples_csv,
            output_csv_path=self.path_test_results_base_csv
        )
        runner.run_all()
        print(f"Base assistant answers stored in: {self.path_test_results_base_csv}")

    # -------------------------------------------------------------------------
    # 5) CREATE EVALUATOR ASSISTANT
    # -------------------------------------------------------------------------
    def create_evaluator_assistant(self):
        """
        Create a specialized assistant that grades or evaluates responses.
        """
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
        self.save_assistant_id(
            self.evaluator_assistant.name,
            self.evaluator_assistant.id,
            self.path_evaluator_id_txt
        )
        print(f"Evaluator assistant created: {self.evaluator_assistant.name}")

    # -------------------------------------------------------------------------
    # 6) EVALUATE (GRADE) EACH RESPONSE OF BASE ASSISTANT
    # -------------------------------------------------------------------------
    def grade_base_assistant_responses(self):
        """
        Use the evaluator assistant to grade the base assistant's responses 
        from the CSV. Adds a new column with the grade (e.g. 'ASSISTANT_NAME_base_grade').
        """
        evaluator_id = self._extract_assistant_id_from_file(self.path_evaluator_id_txt)
        grader = FileManagerGrader(
            openai_api_key=self.openai_api_key,
            assistant_id=evaluator_id,
            csv_input_path=self.path_test_results_base_csv
        )
        # The name of the column that has the base assistant's answer:
        machine_answer_column = f"{self.assistant_name}_{self.base_model_suffix}"

        # Grader will add a 'grade' column in the same file by default
        grader.run(
            question_column=COLUMN_QUESTION,
            human_answer_column=COLUMN_HUMAN_ANSWER,
            machine_answer_column=machine_answer_column,
            output_csv_path=self.path_test_results_base_csv
        )
        print("Base assistant responses graded. Results updated in the same CSV.")

    def _extract_assistant_id_from_file(self, path):
        """
        Utility to parse the assistant ID from the local text file 
        (lines like: ('AssistantName', 'ID-XXXXXX')).
        """
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
    # 7) GATHER WORST QUESTIONS (lowest grades)
    # -------------------------------------------------------------------------
    def gather_worst_questions(self, worst_n=None):
        """
        Sort the test results by the grade assigned to the base assistant 
        (lowest grades first) and pick the N worst.
        """
        if worst_n is None:
            worst_n = self.num_worst_examples

        if not os.path.exists(self.path_test_results_base_csv):
            print(f"No base test results found at {self.path_test_results_base_csv}.")
            return []

        worst_questions = []
        with open(self.path_test_results_base_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # If the grader's numeric column is 'grade', we parse it.
            sorted_rows = sorted(reader, key=lambda x: float(x.get('grade', 0)))
            worst_questions = sorted_rows[:worst_n]

        print(f"Gathered {len(worst_questions)} worst questions.")
        return worst_questions

    # -------------------------------------------------------------------------
    # 8) & 9) CREATE JSONL + UPLOAD TO OPENAI & 10) CREATE FINE-TUNED MODEL
    # -------------------------------------------------------------------------
    def fine_tune_new_assistant(self, worst_questions):
        """
        Fine-tune a new model using the worst questions:
          1) Save worst Q&A to a .txt file
          2) Convert to JSONL
          3) Upload to OpenAI
          4) Create fine-tuning job & wait
          5) Create fine-tuned assistant
        """
        if not worst_questions:
            print("No worst questions found. Fine-tuning skipped.")
            return

        # 1) Save worst questions to .txt
        self._save_worst_questions_as_json(worst_questions, self.path_worst_questions_txt)

        # 2) Convert that .txt to JSONL
        self._convert_txt_to_jsonl(
            input_txt=self.path_worst_questions_txt,
            output_jsonl=self.path_worst_questions_jsonl
        )

        # 3) Upload to OpenAI
        uploader = OpenAIFileUploader(api_key=self.openai_api_key)
        upload_response = uploader.upload_file(
            file_path=self.path_worst_questions_jsonl,
            purpose="fine-tune"
        )
        fine_tune_file_id = upload_response.id
        print("Worst questions JSONL uploaded successfully.")

        # 4) Create fine-tuning job
        fine_tune_job_response = self.fine_tuner.create_fine_tuning_job(
            training_file_id=fine_tune_file_id,
            model=self.base_model_name,
            suffix=f"{self.assistant_name}_{self.fine_tuned_model_suffix}",
        )
        fine_tune_job_id = fine_tune_job_response.id
        self.fine_tune_model = self.fine_tuner.monitor_fine_tuning_job(fine_tune_job_id)
        print(f"Fine-tuned model ready: {self.fine_tune_model}")

        # 5) Create the fine-tuned assistant
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
        self.save_assistant_id(
            self.new_fine_tuned_assistant.name,
            self.new_fine_tuned_assistant.id,
            self.path_assistant_id_fine_tuned_txt
        )
        print(f"Fine-tuned assistant created: {self.new_fine_tuned_assistant.name}")

    def _save_worst_questions_as_json(self, worst_questions, txt_file_path):
        """
        Save the worst questions to a file as a JSON array: [ {"Q":"...", "A":"..."} ]
        """
        data_list = []
        for row in worst_questions:
            question = row.get(COLUMN_QUESTION, "").strip()
            answer = row.get(COLUMN_HUMAN_ANSWER, "").strip()
            data_list.append({"Q": question, "A": answer})

        with open(txt_file_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        print(f"Saved worst questions in JSON format to {txt_file_path}.")

    def _convert_txt_to_jsonl(self, input_txt, output_jsonl):
        """
        Convert the textual Q&A file into a JSONL, using the same structure
        that your TxtToJsonlConverter expects.
        """
        converter = TxtToJsonlConverter(
            input_examples_txt_path=input_txt,
            input_prompt_txt=self.path_instructions_no_examples,  # or any prompt if needed
            output_jsonl_path=output_jsonl
        )
        converter.convert()
        print(f"Converted {input_txt} to {output_jsonl}")

    # -------------------------------------------------------------------------
    # 11) RECEIVE THE ANSWERS OF THE FINE-TUNED MODEL & STORE THEM
    # -------------------------------------------------------------------------
    def evaluate_fine_tuned_assistant(self):
        """
        Run the static test using the new fine-tuned assistant.
        Store answers in PATH_TEST_RESULTS_FINE_TUNED_CSV.
        """
        runner = StaticAssistantsRunner(
            openai_api_key=self.openai_api_key,
            txt_file_path=self.path_assistant_id_fine_tuned_txt,
            csv_file_path=self.path_test_examples_csv,
            output_csv_path=self.path_test_results_fine_tuned_csv
        )
        runner.run_all()
        print(f"Fine-tuned assistant answers stored in: {self.path_test_results_fine_tuned_csv}")

    # -------------------------------------------------------------------------
    # 12) EVALUATE (GRADE) FINE-TUNED ANSWERS
    # -------------------------------------------------------------------------
    def grade_fine_tuned_assistant_responses(self):
        """
        Use the same evaluator assistant to grade the fine-tuned assistant's responses.
        """
        evaluator_id = self._extract_assistant_id_from_file(self.path_evaluator_id_txt)
        grader = FileManagerGrader(
            openai_api_key=self.openai_api_key,
            assistant_id=evaluator_id,
            csv_input_path=self.path_test_results_fine_tuned_csv
        )
        machine_answer_column = f"{self.assistant_name}_{self.fine_tuned_model_suffix}"
        grader.run(
            question_column=COLUMN_QUESTION,
            human_answer_column=COLUMN_HUMAN_ANSWER,
            machine_answer_column=machine_answer_column,
            output_csv_path=self.path_test_results_fine_tuned_csv
        )
        print("Fine-tuned assistant responses graded. Results updated in the CSV.")

    # -------------------------------------------------------------------------
    # 13) MAKE A UNIFIED CSV
    # -------------------------------------------------------------------------
    def unify_results_in_single_csv(self):
        """
        Produce a CSV with columns:
          question, human_answer,
          ASSISTANT_NAME_base_answer, ASSISTANT_NAME_base_grade,
          ASSISTANT_NAME_fine_tuned_answer, ASSISTANT_NAME_fine_tuned_grade
        by merging base vs. fine-tuned results.
        """
        base_file = self.path_test_results_base_csv
        ft_file = self.path_test_results_fine_tuned_csv
        out_file = self.path_test_results_unified_csv

        if not os.path.exists(base_file):
            print(f"No base results at {base_file}")
            return
        if not os.path.exists(ft_file):
            print(f"No fine-tuned results at {ft_file}")
            return

        # Read base results
        with open(base_file, 'r', encoding='utf-8') as f_base:
            base_reader = csv.DictReader(f_base)
            base_rows = list(base_reader)

        # Read fine-tuned
        with open(ft_file, 'r', encoding='utf-8') as f_ft:
            ft_reader = csv.DictReader(f_ft)
            ft_rows = list(ft_reader)
            # Map by question
            ft_map = {row[COLUMN_QUESTION]: row for row in ft_rows}

        # We want final columns:
        # question, human_answer, 
        #   ASSISTANT_NAME_base_answer, ASSISTANT_NAME_base_grade,
        #   ASSISTANT_NAME_fine_tuned_answer, ASSISTANT_NAME_fine_tuned_grade
        base_answer_col = f"{self.assistant_name}_{self.base_model_suffix}"
        base_grade_col = "grade"
        ft_answer_col = f"{self.assistant_name}_{self.fine_tuned_model_suffix}"
        ft_grade_col = "grade"

        final_cols = [
            COLUMN_QUESTION,
            COLUMN_HUMAN_ANSWER,
            f"{self.assistant_name}_base_answer",
            f"{self.assistant_name}_base_grade",
            f"{self.assistant_name}_fine_tuned_answer",
            f"{self.assistant_name}_fine_tuned_grade",
        ]

        with open(out_file, 'w', newline='', encoding='utf-8') as out:
            writer = csv.DictWriter(out, fieldnames=final_cols)
            writer.writeheader()

            for base_row in base_rows:
                q = base_row.get(COLUMN_QUESTION, "")
                human_ans = base_row.get(COLUMN_HUMAN_ANSWER, "")
                base_ans = base_row.get(base_answer_col, "")
                base_grade = base_row.get(base_grade_col, "")

                row_out = {
                    COLUMN_QUESTION: q,
                    COLUMN_HUMAN_ANSWER: human_ans,
                    f"{self.assistant_name}_base_answer": base_ans,
                    f"{self.assistant_name}_base_grade": base_grade,
                    f"{self.assistant_name}_fine_tuned_answer": "",
                    f"{self.assistant_name}_fine_tuned_grade": "",
                }
                # Merge the fine-tuned row if exists
                ft_match = ft_map.get(q)
                if ft_match:
                    row_out[f"{self.assistant_name}_fine_tuned_answer"] = ft_match.get(ft_answer_col, "")
                    row_out[f"{self.assistant_name}_fine_tuned_grade"] = ft_match.get(ft_grade_col, "")

                writer.writerow(row_out)

        print(f"Unified CSV created at: {out_file}")

    # -------------------------------------------------------------------------
    # RUN: MAIN WORKFLOW 
    # -------------------------------------------------------------------------
    def run(self):
        """
        Steps:
          1) Create instructions & separate examples
          2) Create static test
          3) Create base assistant
          4) Get base answers
          5) Create evaluator
          6) Grade base answers
          7) Gather worst questions
          8) Convert to JSONL & Upload
          9) Fine-tune model
          10) Create fine-tuned assistant
          11) Get fine-tuned answers
          12) Grade fine-tuned answers
          13) Unify into CSV
        """
        # 1) create instructions & separate examples
        self.create_instructions()

        """
        # 2) create test CSV
        self.create_static_tests()

        # 3) create base assistant
        self.create_base_assistant()

        # 4) get base answers
        self.evaluate_base_assistant()

        # 5) create evaluator
        self.create_evaluator_assistant()

        # 6) grade base answers
        self.grade_base_assistant_responses()

        # 7) gather worst questions
        worst_questions = self.gather_worst_questions()

        # 8,9,10) fine-tune model & create fine-tuned assistant
        self.fine_tune_new_assistant(worst_questions)

        # 11) get fine-tuned answers
        self.evaluate_fine_tuned_assistant()

        # 12) grade fine-tuned answers
        self.grade_fine_tuned_assistant_responses()

        # 13) unify CSV
        self.unify_results_in_single_csv()
        """


if __name__ == "__main__":
    main_app = Main()
    main_app.run()
