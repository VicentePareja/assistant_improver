import os
import csv
import re
from dotenv import load_dotenv

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

from parametros import (
    INSTRUCTIONS_PATH, TEXT_WITHOUT_EXAMPLES_PATH, EXAMPLES_PATH,
    JSONL_EXAMPLES_PATH, NAME, BASE_MODEL, ID_ASSISTANTS_PATH, BASE_TEST_EXAMPLES_PATH,
    BASE_TEST_RESULTS_PATH, INTRUCTIONS_STATIC_EVALUATOR_PATH, ID_STATIC_EVALUATOR_PATH, 
    CSV_STATIC_RESULTS_PATH, TEMPERATURE, TOP_P, N_EPOCHS, EVAL_MODEL, EVAL_TEMPERATURE,
    EVAL_TOP_P, N_WORST_EXAMPLES, FINE_TUNED_MODEL_SUFIX, BASE_MODEL_SUFIX
)


class Main:
    def __init__(self):
        load_dotenv()

        # Paths and configuration
        self.name = NAME
        self.assistant_id_path = ID_ASSISTANTS_PATH
        self.base_instructions_path = INSTRUCTIONS_PATH
        self.intructions_without_examples_path = TEXT_WITHOUT_EXAMPLES_PATH
        self.examples_path = EXAMPLES_PATH
        self.jsonl_examples_path = JSONL_EXAMPLES_PATH
        self.promt_path = TEXT_WITHOUT_EXAMPLES_PATH
        self.base_test_results_path = BASE_TEST_RESULTS_PATH
        self.static_evaluator_promt_path = INTRUCTIONS_STATIC_EVALUATOR_PATH
        self.static_evaluator_id_path = ID_STATIC_EVALUATOR_PATH
        self.static_results_path = CSV_STATIC_RESULTS_PATH
        self.temperature = TEMPERATURE
        self.top_p = TOP_P
        self.n_epochs = N_EPOCHS
        self.eval_model = EVAL_MODEL
        self.eval_temperature = EVAL_TEMPERATURE
        self.eval_top_p = EVAL_TOP_P

        # Credentials
        self.service_account_path = os.getenv("SERVICE_ACCOUNT_FILE")
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.assistant_id = os.getenv('ID_ASSISTANT_TEXT_SEPARATOR')

        # Tools
        self.fine_tuner = OpenAIFineTuner(api_key=self.openai_api_key)
        self.static_test_creator = StaticExamplesTestCreator(
            input_test_file=self.examples_path, 
            output_test_file=BASE_TEST_EXAMPLES_PATH
        )
        self.static_assistant_runner = StaticAssistantsRunner(
            openai_api_key=self.openai_api_key,
            txt_file_path=self.assistant_id_path,
            csv_file_path=BASE_TEST_EXAMPLES_PATH,
            output_csv_path=self.base_test_results_path
        )

    # -------------------------------------------------------------------------
    # --------------------------  HELPER METHODS  ------------------------------
    # -------------------------------------------------------------------------
    def find_doc_id(self, assistant_name):
        """
        Fetch the doc ID and store it internally.
        """
        finder = AssistantDocFinder()
        assistant_id, gdocs_address = finder.get_doc_id_by_assistant_name(assistant_name)
        self.document_id = gdocs_address
        print(f"Document with assistant's instructions found for '{assistant_name}'.")

    def import_text_from_google_doc(self):
        """
        Use the doc ID to import text from a Google Doc into a local file.
        """
        importer = DocumentImporter(
            self.service_account_path,
            self.document_id,
            self.base_instructions_path
        )
        importer.import_text()

    def separate_text(self):
        """
        Separates text into sections using an existing assistant.
        """
        separator_runner = TextSeparatorRunner(
            api_key=self.openai_api_key,
            assistant_id=self.assistant_id
        )
        separator_runner.run()

    def save_assistant_id(self, assistant_name, assistant_id, path):
        """
        Simple helper to append the new assistant id to a local file.
        """
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{assistant_name, assistant_id}\n")

    def create_static_tests(self):
        """
        Create your test CSV from a base set of examples (if needed).
        """
        self.static_test_creator.create_test()

    # -------------------------------------------------------------------------
    # ------------------------  INSTRUCTIONS (Step 1)  -------------------------
    # -------------------------------------------------------------------------
    def create_instructions(self):
        """
        1. Find doc ID in Drive (by name).
        2. Import text from Google Doc.
        3. Separate text into instructions.
        """
        self.find_doc_id(self.name)
        self.import_text_from_google_doc()
        self.separate_text()

    # -------------------------------------------------------------------------
    # ----------------------  BASE ASSISTANT (Step 3)  ------------------------
    # -------------------------------------------------------------------------
    def create_base_assistant(self):
        """
        Create the base assistant from the instructions.
        """
        assistant_creator = AssistantCreator(
            api_key=self.openai_api_key,
            instructions_path=self.base_instructions_path
        )
        self.base_assistant = assistant_creator.create_assistant(
            name_suffix=BASE_MODEL_SUFIX,
            model=BASE_MODEL,
            tools=[],
            temperature=self.temperature,
            top_p=self.top_p
        )
        self.save_assistant_id(self.base_assistant.name,
                               self.base_assistant.id,
                               self.assistant_id_path)
        print("Base assistant created.")

    def evaluate_base_assistant(self):
        """
        Evaluate only the base assistant on the static tests.
        This uses the StaticAssistantsRunner but restricts to a single model column.
        """
        # The runner by default might evaluate all assistants in the txt file.
        # If you only want to evaluate the base assistant,
        # you can either remove others from the ID file or adjust the runner.
        # For simplicity, let's re-use the runner's 'run_all()' but you
        # could also implement a 'run_single_assistant()' version if needed.
        self.static_assistant_runner.run_all()
        print("Base assistant evaluation completed. Results stored in CSV.")

    def store_evaluation_data(self):
        """
        Store or unify evaluation data. 
        (Optional) You can unify or rename as you want to differentiate base data.
        """
        # For example, you could call a grading step or unify the CSV results:
        # self.grade_static_tests()   # if you want a numeric grade
        # self.generate_unified_csv_results()  # if you want a single CSV
        print("Base assistant data stored (placeholder).")

    # -------------------------------------------------------------------------
    # -------------------  GATHER WORST QUESTIONS (Step 4)  -------------------
    # -------------------------------------------------------------------------
    def gather_worst_questions(self, worst_n=N_WORST_EXAMPLES):
        """
        Parse the CSV with results from the base assistant to find
        the N worst questions. This is a placeholder example.
        """
        worst_questions = []
        if not os.path.exists(self.base_test_results_path):
            print("Base test results file not found. Cannot gather worst questions.")
            return worst_questions

        # Example: we look for rows with the lowest "score" or something.
        # Adjust to your actual CSV format and scoring logic.
        with open(self.base_test_results_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Suppose you have a column "score_base" or similar
            sorted_rows = sorted(reader, key=lambda x: float(x.get("score_base", 0)))
            worst_questions = sorted_rows[:worst_n]

        print(f"Worst {worst_n} questions gathered.")
        return worst_questions

    # -------------------------------------------------------------------------
    # ---------  FINE-TUNE NEW ASSISTANT WITH WORST QUESTIONS (Step 5)  -------
    # -------------------------------------------------------------------------
    def fine_tune_new_assistant(self, worst_questions):
        """
        Fine-tune a new model with the worst questions found in base tests.
        1) Convert worst questions to JSONL
        2) Upload JSONL to OpenAI
        3) Launch fine-tuning job
        4) Create new assistant from the fine-tuned model
        """
        if not worst_questions:
            print("No worst questions found. Skipping fine-tune.")
            return

        # 1) Convert worst questions to JSONL
        temp_jsonl_path = "data/fine_tune/worst_questions.jsonl"
        self.create_jsonl_from_questions(worst_questions, temp_jsonl_path)

        # 2) Upload JSONL
        uploader = OpenAIFileUploader(api_key=self.openai_api_key)
        upload_response = uploader.upload_file(file_path=temp_jsonl_path, purpose="fine-tune with worst examples")
        fine_tune_file_id = upload_response.id
        print("Worst questions JSONL uploaded successfully.")

        # 3) Launch fine-tuning job
        job_response = self.fine_tuner.create_fine_tuning_job(
            training_file_id=fine_tune_file_id,
            model=BASE_MODEL,
            suffix=f"{NAME}_{FINE_TUNED_MODEL_SUFIX}"
        )
        fine_tune_job_id = job_response.id
        self.fine_tune_model = self.fine_tuner.monitor_fine_tuning_job(fine_tune_job_id)

        # 4) Create new assistant with the newly fine-tuned model
        assistant_creator = AssistantCreator(
            api_key=self.openai_api_key,
            instructions_path=self.base_instructions_path
        )
        self.new_fine_tuned_assistant = assistant_creator.create_assistant(
            name_suffix=FINE_TUNED_MODEL_SUFIX,
            model=self.fine_tune_model,
            tools=[],
            temperature=self.temperature,
            top_p=self.top_p
        )
        self.save_assistant_id(
            self.new_fine_tuned_assistant.name,
            self.new_fine_tuned_assistant.id,
            self.assistant_id_path
        )
        print("New fine-tuned assistant created.")

    def create_jsonl_from_questions(self, worst_questions, output_path):
        """
        A small helper to transform the worst questions into a JSONL
        with the structure expected by OpenAI for fine-tuning.
        """
        # Example structure: {"prompt": "...", "completion": "..."}
        # Adjust logic to your actual format.
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in worst_questions:
                prompt = row.get("question", "")
                completion = row.get("human_response", "")
                data = {
                    "prompt": prompt.strip(),
                    "completion": " " + completion.strip()  # space at start per OpenAI docs
                }
                f.write(f"{data}\n")

    # -------------------------------------------------------------------------
    # ------------  EVALUATE NEW ASSISTANT (Step 6) & STORE (Step 7)  ---------
    # -------------------------------------------------------------------------
    def evaluate_new_assistant(self):
        """
        Evaluate the newly fine-tuned assistant on the same test set (or a new one).
        """
        # By default, run_all() checks for any assistant IDs in the text file.
        # Ensure the new assistant ID is in that file so it can be evaluated.
        self.static_assistant_runner.run_all()
        print("New assistant evaluation completed.")

    def store_new_assistant_data(self):
        """
        Gather or unify final data regarding the new assistant's performance.
        """
        # e.g., generating an updated unified CSV or separate file
        # self.grade_static_tests() or self.generate_unified_csv_results() if desired
        print("New assistant data stored (placeholder).")

    # -------------------------------------------------------------------------
    # -----------------------------  MAIN FLOW  --------------------------------
    # -------------------------------------------------------------------------
    def run(self):
        """
        Final orchestrated flow:
            1. Create the instructions
            2. Create the tests
            3. Evaluate the base assistant
                3.1. Store the data
            4. Gather worst questions
            5. Fine-tune a new assistant with the worst questions
            6. Evaluate the new assistant
            7. Store the data
        """
        # 1) Create instructions
        #self.create_instructions()

        # 2) Create tests
        #self.create_static_tests()

        # 3) Evaluate base assistant
        #self.create_base_assistant()        # creates the base assistant
        #self.evaluate_base_assistant()      # run static tests for the base assistant

        # 3.1) Store data
        #self.store_evaluation_data()        # placeholder method for storing/evaluating data

        # 4) Gather worst questions
        worst_questions = self.gather_worst_questions(worst_n=N_WORST_EXAMPLES)

        # 5) Fine-tune new assistant with the worst questions
        self.fine_tune_new_assistant(worst_questions)

        # 6) Evaluate the new assistant
        self.evaluate_new_assistant()

        # 7) Store final data
        self.store_new_assistant_data()


if __name__ == "__main__":
    main_app = Main()
    main_app.run()
