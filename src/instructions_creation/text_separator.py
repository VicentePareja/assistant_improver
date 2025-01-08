import sys
sys.stdout.reconfigure(encoding='utf-8')  # Ensure UTF-8 output

import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override

# Import updated parameter names from parameters.py
from parameters import (
    PATH_INSTRUCTIONS_NO_EXAMPLES,
    PATH_INSTRUCTIONS_TXT,
    PATH_EXAMPLES_TXT,
)

################################################################################
# EventHandler: Handles streaming events from OpenAI (already OOP).
################################################################################
class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        # Overriding to do nothing or minimal prints
        pass
    
    @override
    def on_text_delta(self, delta, snapshot):
        # Overriding to do nothing or minimal prints
        pass
    
    def on_tool_call_created(self, tool_call):
        # Overriding to do nothing or minimal prints
        pass
    
    def on_tool_call_delta(self, delta, snapshot):
        # Overriding to do nothing or minimal prints
        pass


################################################################################
# TextSeparator: Our main OOP class that handles reading instructions,
# calling OpenAI, extracting JSON, and saving results.
################################################################################
class TextSeparator:
    def __init__(self, api_key: str, assistant_id: str):
        """
        :param api_key: Your OpenAI API key
        :param assistant_id: The ID of your target assistant on OpenAI
        """
        self.api_key = api_key
        self.assistant_id = assistant_id
        self.client = OpenAI(api_key=self.api_key)

    def run(self):
        """
        Reads the instructions file, sends it to the assistant, 
        extracts JSON, writes text outputs, etc.
        """
        # 1. Read your instructions from a local file
        prompt = self._read_instructions(PATH_INSTRUCTIONS_TXT)

        # 2. Send prompt to the assistant and capture the combined response
        combined_response = self._ask_assistant(prompt)
        if not combined_response:
            print("No valid response from assistant or error encountered.")
            return

        # 3. Extract the JSON portion from the combined response
        actual_json_str = self._extract_json(combined_response)
        if not actual_json_str:
            print("No JSON object found in the response.")
            return

        # 4. Parse that JSON
        text_without_examples, only_examples = self._parse_json(actual_json_str)
        if text_without_examples is None and only_examples is None:
            # Means we failed to parse
            return

        # 5. Write the results to file
        self._write_results(text_without_examples, only_examples)
        print(
            f"Saved JSON fields into '{PATH_INSTRUCTIONS_NO_EXAMPLES}' "
            f"and '{PATH_EXAMPLES_TXT}'"
        )

    ########################################################################
    # Internal helper methods (all OOP)
    ########################################################################
    def _read_instructions(self, file_path: str) -> str:
        """
        Reads the entire instructions file as a string.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content

    def _ask_assistant(self, prompt: str) -> str:
        """
        Sends `prompt` to the assistant and returns a combined string
        of all assistant messages.
        """
        try:
            # Create the conversation thread
            thread = self.client.beta.threads.create()

            # Send user prompt
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )

            # Stream the assistant's response
            with self.client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=self.assistant_id,
                event_handler=EventHandler(),
            ) as stream:
                stream.until_done()

            # Retrieve the full conversation (including final assistant message)
            response_message = self.client.beta.threads.messages.list(thread_id=thread.id)
            if not response_message or not response_message.data:
                print("No response from assistant.")
                return ""

            # Filter out only the messages from the assistant
            assistant_responses = [
                msg.content
                for msg in response_message.data
                if msg.role == 'assistant'
            ]

            if not assistant_responses:
                print("No assistant messages found.")
                return ""

            # Combine all assistant messages into one string
            combined_response = "\n".join(str(r) for r in assistant_responses)
            return combined_response

        except Exception as e:
            print(f"Error running prompt: {e}")
            return ""

    def _extract_json(self, combined_response: str) -> str:
        """
        Extracts the JSON portion of the combined_response using a regex.
        Sanitizes the content before returning.
        """
        regex_pattern = r"value='(\{.*?\})'"
        match = re.search(regex_pattern, combined_response, re.DOTALL)
        if not match:
            print("No JSON object found in the response (regex match failed).")
            return ""

        raw_json_str = match.group(1)

        # Replace certain escape sequences
        raw_json_str = raw_json_str.replace("\\\n", "\\n")
        raw_json_str = raw_json_str.replace("\\\t", "\\t")

        # Replace "smart quotes"
        raw_json_str = raw_json_str.replace("‘", "'").replace("’", "'")

        # Convert \n and \t to literal JSON sequences
        sanitized_json_str = raw_json_str.replace("\\n", "\\\\n").replace("\\t", "\\\\t")

        # Fix stray backslashes that aren't valid JSON escapes
        sanitized_json_str = re.sub(
            r'\\(?=[^"\\/bfnrtu])',  # look for \ not followed by a valid escape
            r'\\\\',                # replace with double backslash
            sanitized_json_str
        )

        return sanitized_json_str

    def _parse_json(self, json_str: str):
        """
        Attempts to parse the JSON string. If it fails, returns (None, None).
        """
        try:
            parsed_json = json.loads(json_str)
            text_without_examples = parsed_json.get("text_without_examples", "")
            only_examples = parsed_json.get("only_examples", "")
            return text_without_examples, only_examples

        except json.JSONDecodeError:
            return None, None

    def _write_results(self, text_without_examples: str, only_examples):
        """
        Writes the extracted strings to the specified output files.
        """
        with open(PATH_INSTRUCTIONS_NO_EXAMPLES, "w", encoding="utf-8") as f1:
            f1.write(text_without_examples)

        with open(PATH_EXAMPLES_TXT, "w", encoding="utf-8") as f2:
            if isinstance(only_examples, list):
                f2.write(json.dumps(only_examples, ensure_ascii=False, indent=2))
            else:
                f2.write(str(only_examples))


class TextSeparatorRunner:
    def __init__(self, api_key: str, assistant_id: str):
        self.api_key = api_key
        self.assistant_id = assistant_id

    def run(self):
        separator = TextSeparator(api_key=self.api_key, assistant_id=self.assistant_id)
        separator.run()
