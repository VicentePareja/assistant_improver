"""
parameters.py

Centralizes all key parameters and file paths used in the pipeline.
Update references in your main code to match these variable names.
"""

# ------------------------------------------------------------------
# 1) Assistant/Model Basic Info
# ------------------------------------------------------------------
# parameters.py

ASSISTANT_NAME = "House of Spencer" #Change this to your assistant's name
BASE_MODEL_NAME = "gpt-4o-mini-2024-07-18"   # The base model to start from
BASE_MODEL_SUFFIX = "base"                  # Suffix to identify the base assistant

# Tuning & inference parameters for the base assistant
BASE_TEMPERATURE = 0.5
BASE_TOP_P = 1

# ------------------------------------------------------------------
# 2) Fine-Tuning Settings
# ------------------------------------------------------------------
NUM_WORST_EXAMPLES = 19
FINE_TUNED_MODEL_SUFFIX = "fine_tuned_with_worst"

# ------------------------------------------------------------------
# 3) Evaluator Assistant Settings
# ------------------------------------------------------------------
EVALUATOR_MODEL_NAME = "gpt-4o-mini-2024-07-18"  # The model used for evaluation
EVALUATOR_TEMPERATURE = 0
EVALUATOR_TOP_P = 0.5

EVALUATOR_INTRODUCTION_PROMT = """Tu labor es evaluar un asistente de IA y compararlo con respuestas humanas.
Es decir, se te entregara una pregunta y dos respuestas, una realizada por el humano y otra por el asistente de IA y debes compararlas.

Para que tengas contexto acerca del asistente, te dejo sus intrucciones:"""

EVALUATOR_DESCRIPTION_PROMPT = """Para evaluar esto lo haras con un número del 1 al 5.

1: La respuesta está incorrecta
2. La respuesta está incorrecta pero tiene algo de verdad
3. La respuesta es parcialemente correcta pero falta información
4. La respuesta es tan correcta como la humana
5. La respuesta es mejor que la humana

Responde solo con el  numero. Nada más. Esto es de suma importancia.

Nota adicional: Has especial énfasis en la correcititud de datos como los enlaces o los numeros.

A continuación, se te entregará la pregunta, la respuesta humana y la pregunta del asistente."""

# ------------------------------------------------------------------
# 4) CSV Column Names
# ------------------------------------------------------------------
COLUMN_QUESTION = "question"
COLUMN_HUMAN_ANSWER = "human_response"

# ------------------------------------------------------------------
# 5) Local File Paths & Directories
# ------------------------------------------------------------------

# Original instructions text, as exported from a Google Doc
PATH_INSTRUCTIONS_TXT = f"data/original_instructions/{ASSISTANT_NAME}_original_instructions.txt"

# Text without examples (generated by the text separator)
PATH_INSTRUCTIONS_NO_EXAMPLES = f"data/separate_examples_from_text/{ASSISTANT_NAME}_text_without_examples.txt"

# File containing only the extracted examples (Q&A) from the instructions
PATH_EXAMPLES_TXT = f"data/separate_examples_from_text/{ASSISTANT_NAME}_examples.txt"

# Where you store the assistant's ID(s) after creation
PATH_ASSISTANTS_IDS_TXT = f"data/assistants_ids/{ASSISTANT_NAME}_assistants_ids.txt"

# Separate ID file for the fine-tuned assistant
PATH_ASSISTANT_ID_FINE_TUNED_TXT = f"data/assistants_ids/{ASSISTANT_NAME}_fine_tuned_assistant_id.txt"

# Prompts for the static evaluator assistant
PATH_INSTRUCTIONS_EVALUATOR_TXT = f"evaluator_prompt/static/{ASSISTANT_NAME}_static_evaluator_prompt.txt"

# Where you store the evaluator assistant's ID
PATH_EVALUATOR_ID_TXT = f"data/assistants_ids/{ASSISTANT_NAME}_static_evaluator_id.txt"

# ------------------------------------------------------------------
# 6) Testing/Results Paths
# ------------------------------------------------------------------

# The CSV containing test questions & human answers (generated from examples)
PATH_TEST_EXAMPLES_CSV = f"data/test/{ASSISTANT_NAME}_base_test_examples.csv"

# STEP 4) The CSV with the base assistant's answers (only answers, no grades)
PATH_BASE_ANSWERS_CSV = f"data/test/{ASSISTANT_NAME}_base_assistant_answers.csv"

# STEP 6) The CSV with the base assistant's grades
PATH_BASE_GRADES_CSV = f"data/test/{ASSISTANT_NAME}_base_assistant_grades.csv"

# STEP 11) The CSV with the fine-tuned assistant's answers (only answers, no grades)
PATH_FINE_TUNED_ANSWERS_CSV = f"data/test/{ASSISTANT_NAME}_fine_tuned_assistant_answers.csv"

# STEP 12) The CSV with the fine-tuned assistant's grades
PATH_FINE_TUNED_GRADES_CSV = f"data/test/{ASSISTANT_NAME}_fine_tuned_assistant_grades.csv"

# The CSV that unifies everything in step 13
PATH_UNIFIED_RESULTS_CSV = f"data/results/{ASSISTANT_NAME}_unified_results.csv"

# ------------------------------------------------------------------
# 7) Fine-Tuning Data for the Worst Questions
# ------------------------------------------------------------------

# Temporary file to store the worst Q&A in a JSON array (before JSONL conversion)
PATH_WORST_QUESTIONS_TXT = f"data/worst_questions/{ASSISTANT_NAME}_worst_questions.txt"

# The JSONL file used for fine-tuning (uploaded to OpenAI)
PATH_WORST_QUESTIONS_JSONL = f"data/worst_questions/{ASSISTANT_NAME}_worst_questions.jsonl"

