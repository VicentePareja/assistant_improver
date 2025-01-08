# HOS Assistant Creation and Evaluation

This project automates the creation, fine-tuning, and evaluation of AI assistants based on OpenAI's language models. It provides a pipeline to generate assistants with different configurations, test their performance, and grade their outputs for quality.

## Project Structure

```
ASSISTANTS-CREATOR/
├── __pycache__/
├── data/
│   ├── assistant_responses/
│   ├── assistants_ids/
│   │   └── HOSid_assistants.txt
│   ├── separate_examples_from_text/
│   │   ├── HOSexamples.jsonl
│   │   ├── HOSexamples.txt
│   │   ├── HOStext_without_examples.txt
│   ├── evaluator/
│   │   ├── HOSstatic_evaluator_results.csv
│   │   ├── unified_results.csv
│   ├── test/
│   │   ├── HOSstatic_test_examples.txt
│   │   ├── HOSstatic_test_results.csv
├── src/
│   ├── instructions_creation/
│   │   ├── file_importer.py
│   │   ├── text_separator.py
│   ├── assistant_creator/
│   │   └── assistant_creator.py
│   ├── assistant_finetuner/
│   │   ├── create_finetune_model.py
│   │   ├── examples_to_jsonl.py
│   │   ├── upload_jsonl.py
│   ├── assistant_testing/
│   │   ├── static_test_creator.py
│   │   ├── static_assistant_tester.py
│   │   ├── static_grader_results.py
├── main.py
├── parametros.py
├── .env
```

## tasks of the code:

1. create the instructions and separete the examples from the rest.

2. create the test with the questions and the human answers

3. create the base assistant

4. recieve and store the machine answers of the tests

5. create an evaluator assistant

6. evaluate (grade) each response of the base assistant.

7. Gather the worst questions (worsts grades) and put them in the correct format (JSONL)

8. upload the jsonl file to open ai

9. Create a fine tuned model

10. create a fine tuned assistant

11. reacieve the answers of the fine tuned model and store them

12. with the same evaluator grade each answer

13. Make a unified .csv file with the following structure: Question, human answer, NAME_base_answer, NAME_base_grade, NAME_fine_tuned_answer, NAME_fine_tuned_grade


## Setup

### Prerequisites
- Python 3.10+
- OpenAI API key
- Google service account credentials
- `.env` file for sensitive information

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd ASSISTANTS-CREATOR
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following content:
   ```env
   OPENAI_API_KEY=<your_openai_api_key>
   SERVICE_ACCOUNT_FILE=<path_to_google_service_account_json>
   DOCUMENT_ID=<google_doc_id>
   ID_ASSISTANT_TEXT_SEPARATOR=<assistant_id>
   ```

### Google Services Setup

To enable Google Docs access for the project:

1. **Enable Google Docs API**:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/).
   - Navigate to **APIs & Services** > **Library**.
   - Search for **Google Docs API**.
   - Click **Enable**.

2. **Create a Service Account**:
   - Navigate to **IAM & Admin** > **Service Accounts**.
   - Click **Create Service Account**.
   - Enter a name (e.g., "AssistantCreator Service Account") and click **Create and Continue**.
   - Assign the **Editor** role to the service account.
   - Click **Done**.

3. **Generate Service Account Credentials**:
   - In the **Service Accounts** section, find your service account.
   - Click on the service account and go to the **Keys** tab.
   - Click **Add Key** > **Create New Key**.
   - Choose **JSON** and download the key file.
   - Save the file in your project directory and update the `SERVICE_ACCOUNT_FILE` path in your `.env` file.

4. **Share Google Document with Service Account**:
   - Open the Google Document you want to use.
   - Click **Share** in the top-right corner.
   - Add the service account email (e.g., `your-service-account@your-project.iam.gserviceaccount.com`) as a Viewer or Editor.
   - Save the changes.

## Usage

1. **Run the Main Script**:
   ```bash
   python main.py
   ```

2. **Pipeline Steps**:
   - Import instructions from Google Docs.
   - Create assistants:
     - Without examples
     - Base assistant
     - Fine-tuned assistant
   - Generate and evaluate static tests.

3. **Results**:
   - Check assistant IDs in `data/assistants_ids/`.
   - Review test results in `data/test/`.
   - Analyze unified grades in `data/evaluator/unified_results.csv`.

## Parameters
Defined in `parametros.py`:

- `NAME`: Assistant name.
- `BASE_MODEL`: Base OpenAI model to use.
- Paths for instructions, examples, JSONL files, test results, and evaluation results.

## File Outputs

### Assistants
- `data/assistants_ids/`: Stores IDs of created assistants.

### Tests
- `data/test/`: Contains static test examples and results.

### Evaluation
- `data/evaluator/`: Contains graded results and the unified results CSV.

## Extending the Project

1. **Adding New Tests**:
   - Update `StaticExamplesTestCreator` with additional test cases.

2. **Supporting More Models**:
   - Modify `parametros.py` to include new model configurations.

## Contributing
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/new-feature
   ```
3. Commit changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/new-feature
   ```
5. Create a pull request.

## License
This project is licensed under the MIT License.