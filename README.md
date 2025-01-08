Below is a **comprehensive `README.md`** file you can place at the **root of your project**. It explains:

1. **What the project does**  
2. **How it’s organized**  
3. **How to install and run everything**  
4. **A detailed breakdown** of the main steps in the workflow  

Feel free to adapt paths, commands, or instructions as needed for your specific environment.

---

```markdown
# Assistant Improver

This repository provides a **complete pipeline** to create, refine, test, and evaluate an **OpenAI-based assistant** (referred to here as *"House of Spencer"*). The pipeline uses custom Python scripts and a well-defined series of steps to:

1. **Extract instructions** from a Google Doc, separate examples from the rest.  
2. **Generate a test set** of questions and human answers.  
3. **Create a base assistant** (using instructions).  
4. **Obtain** that base assistant’s **answers** on the test.  
5. **Create an evaluator** assistant.  
6. **Grade** each of the base assistant’s answers (store numeric scores).  
7. **Identify** the worst-performing questions (lowest grades).  
8. **Convert** those worst questions into JSON/JSONL.  
9. **Upload** them to OpenAI and **fine-tune** a new model.  
10. **Create** the fine-tuned assistant.  
11. **Obtain** the fine-tuned assistant’s answers.  
12. **Grade** those answers with the same evaluator.  
13. **Unify** everything in one final CSV.

---

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Requirements](#requirements)  
3. [Installation](#installation)  
4. [Environment Setup](#environment-setup)  
5. [Usage and Workflow](#usage-and-workflow)  
   1. [Step 1: Instructions](#step-1-instructions)  
   2. [Step 2: Create Test CSV](#step-2-create-test-csv)  
   3. [Step 3: Create Base Assistant](#step-3-create-base-assistant)  
   4. [Step 4: Get Base Answers](#step-4-get-base-answers)  
   5. [Step 5: Create Evaluator Assistant](#step-5-create-evaluator-assistant)  
   6. [Step 6: Grade Base Answers](#step-6-grade-base-answers)  
   7. [Step 7: Gather Worst Questions](#step-7-gather-worst-questions)  
   8. [Steps 8-10: Fine-Tuning Workflow](#steps-8-10-fine-tuning-workflow)  
   9. [Step 11: Fine-Tuned Answers](#step-11-fine-tuned-answers)  
   10. [Step 12: Grade Fine-Tuned Answers](#step-12-grade-fine-tuned-answers)  
   11. [Step 13: Unify Everything](#step-13-unify-everything)  
6. [Troubleshooting](#troubleshooting)  
7. [License](#license)

---

## 1. Project Structure

A high-level look at the repository’s folders and files:

```
assistant_improver/
├── data/
│   ├── assistants_ids/
│   │   ├── House of Spencer_assistants_ids.txt
│   │   └── House of Spencer_static_evaluator_id.txt
│   ├── fine_tune/
│   ├── original_instructions/
│   │   └── House of Spencer_original_instructions.txt
│   ├── separate_examples_from_text/
│   │   ├── House of Spencer_examples.txt
│   │   └── House of Spencer_text_without_examples.txt
│   ├── test/
│   │   ├── House of Spencer_base_test_examples.csv
│   │   ├── House of Spencer_base_assistant_answers.csv
│   │   ├── House of Spencer_base_assistant_grades.csv
│   │   ├── House of Spencer_fine_tuned_assistant_answers.csv
│   │   └── House of Spencer_fine_tuned_assistant_grades.csv
│   ├── worst_questions/
│   │   ├── House of Spencer_worst_questions.txt
│   │   └── House of Spencer_worst_questions.jsonl
│   └── evaluator/
│       └── House of Spencer_unified_results.csv
├── evaluator_prompt/
│   ├── static/
│   │   └── House of Spencer_static_evaluator_prompt.txt
│   └── dynamic/
├── src/
│   ├── instructions_creation/
│   │   ├── file_importer.py
│   │   ├── text_separator.py
│   │   └── intructions_id_finder.py
│   ├── assistant_creator/
│   │   └── assistant_creator.py
│   ├── assistant_finetuner/
│   │   ├── examples_to_jsonl.py
│   │   ├── create_finetune_model.py
│   │   └── upload_jsonl.py
│   └── assistant_testing/
│       ├── static_test_creator.py
│       ├── static_assistant_tester.py
│       └── static_grader_results.py
├── main.py                # The main orchestrator script
├── parameters.py          # Holds all constants, paths, and model settings
├── requirements.txt       # Python dependencies
├── README.md              # (You are here)
└── .env                   # For environment variables (OpenAI keys, etc.)
```

---

## 2. Requirements

All Python dependencies are listed in **`requirements.txt`**. Typically, they include:

- `openai`  
- `python-dotenv`  
- Possibly other libraries depending on your environment (e.g., `requests`, `pandas`).

---

## 3. Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/your-user/assistant_improver.git
   cd assistant_improver
   ```
2. **Create and activate** a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Linux/Mac
   # or venv\Scripts\activate  # On Windows
   ```
3. **Install** all dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables** in `.env` (see [Environment Setup](#environment-setup)).

---

## 4. Environment Setup

Create a file named **`.env`** in the project’s root directory with the following variables (adapt as needed):

```ini
# .env
OPENAI_API_KEY=your_api_key_here
SERVICE_ACCOUNT_FILE=/absolute/path/to/service_account_credentials.json
ID_ASSISTANT_TEXT_SEPARATOR=ID-of-text-separator-assistant
```

- **`OPENAI_API_KEY`**: Your OpenAI API key  
- **`SERVICE_ACCOUNT_FILE`**: Path to Google service account credentials (for accessing Google Docs, if used)  
- **`ID_ASSISTANT_TEXT_SEPARATOR`**: The ID of an existing assistant specialized in “text separating” (if your pipeline uses one)

---

## 5. Usage and Workflow

All major steps are **orchestrated** in **`main.py`**. You can **uncomment** or **comment** lines in the `run()` method to choose which steps to run.

### Step 1: Instructions

1. **`create_instructions()`**  
   - Finds the doc ID on Google Drive for your assistant’s instructions.  
   - Imports the doc text locally.  
   - Separates “examples” from the rest using a specialized “text separator” assistant.

### Step 2: Create Test CSV

1. **`create_static_tests()`**  
   - Reads the newly extracted examples (`House of Spencer_examples.txt`) and creates a standardized CSV of question-and-answer pairs.  
   - Example output: `House of Spencer_base_test_examples.csv`.

### Step 3: Create Base Assistant

1. **`create_base_assistant()`**  
   - Reads your instructions from `House of Spencer_original_instructions.txt`.  
   - Creates an assistant on OpenAI (the “base” assistant).  
   - Writes the new assistant’s ID to `data/assistants_ids/House of Spencer_assistants_ids.txt`.

### Step 4: Get Base Answers

1. **`get_base_assistant_answers()`**  
   - Runs the **base** assistant over `House of Spencer_base_test_examples.csv`.  
   - Stores just the answers (with questions & human answers) in `House of Spencer_base_assistant_answers.csv`.

### Step 5: Create Evaluator Assistant

1. **`create_evaluator_assistant()`**  
   - Creates a special assistant that can “grade” or “score” answers.  
   - Uses `House of Spencer_static_evaluator_prompt.txt` for instructions.  
   - Writes the ID to `data/assistants_ids/House of Spencer_static_evaluator_id.txt`.

### Step 6: Grade Base Answers

1. **`grade_base_assistant_responses()`**  
   - Uses the evaluator assistant to assign numeric grades to each answer from **Step 4**.  
   - Outputs a new file, `House of Spencer_base_assistant_grades.csv`, which includes a “grade” column.

### Step 7: Gather Worst Questions

1. **`gather_worst_indices()`**  
   - Reads the “grades” CSV, sorts by grade ascending (lowest first).  
   - Retrieves the row indices of the worst (lowest-scoring) entries.  
2. **`create_worst_questions_file()`**  
   - Looks up those row indices in `House of Spencer_base_assistant_answers.csv` and extracts the `question` + `human_answer`.  
   - Writes them to a new file `House of Spencer_worst_questions.txt` as an **array** of objects:
     ```json
     [
       {"Q": "question1", "A": "answer1"},
       {"Q": "question2", "A": "answer2"}
     ]
     ```

### Steps 8–10: Fine-Tuning Workflow

1. **`convert_worst_txt_to_jsonl()`**  
   - Converts the `.txt` with Q&A to `.jsonl` format (OpenAI’s required input).  
2. **`upload_worst_jsonl()`**  
   - Uploads the `.jsonl` to OpenAI.  
3. **`create_fine_tuning_job()`**  
   - Initiates the fine-tuning job with the newly uploaded file.  
   - Monitors until completion.  
4. **`create_fine_tuned_assistant()`**  
   - Wraps the fine-tuned model in an assistant (similar to the base assistant creation).  

### Step 11: Fine-Tuned Answers

1. **`get_fine_tuned_assistant_answers()`**  
   - Runs the newly created fine-tuned assistant on the same test CSV.  
   - Stores the output in `House of Spencer_fine_tuned_assistant_answers.csv`.

### Step 12: Grade Fine-Tuned Answers

1. **`grade_fine_tuned_assistant_responses()`**  
   - Uses the **same** evaluator assistant from Step 5 to score the fine-tuned answers.  
   - Produces `House of Spencer_fine_tuned_assistant_grades.csv`.

### Step 13: Unify Everything

1. **`unify_results_in_single_csv()`**  
   - Reads **four** files:
     - Base answers  
     - Base grades  
     - Fine-tuned answers  
     - Fine-tuned grades  
   - Merges them by **index** (or question) into one CSV: `House of Spencer_unified_results.csv`.  
   - Columns:
     ```
     question,
     human_response,
     House of Spencer_base_answer,
     House of Spencer_base_grade,
     House of Spencer_fine_tuned_answer,
     House of Spencer_fine_tuned_grade
     ```

---

## 6. Troubleshooting

- **Missing `.env`**: If `OPENAI_API_KEY` or `SERVICE_ACCOUNT_FILE` is not found, your code may raise connection errors.  
- **File Not Found**: Check your paths in `parameters.py` if you see “Missing file: …” messages.  
- **Sorting / Index Mismatch**: The code uses row **indices** to match answers with grades. Ensure you do **not** manually sort CSVs in between steps, or else the row alignment will break.  
- **Google Docs Import**: If your doc ID or service account credentials are invalid, you may get errors from the `file_importer.py` script.

---

## 7. License

This project is distributed under your preferred license. Update this section as appropriate (e.g., MIT, Apache 2.0, etc.).

---

**Enjoy refining your AI assistant!** If you have any questions or run into issues, feel free to open a ticket or reach out to the repository maintainers.
```