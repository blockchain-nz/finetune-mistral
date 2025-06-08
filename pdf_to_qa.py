import fitz  # PyMuPDF
import json
import argparse
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

# Load environment variables from .env file
load_dotenv()

# --- Configuration for Gemini API ---
# Ensure GOOGLE_API_KEY is set in your .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

# Model configuration - using a recent Gemini model suitable for text generation
# You might need to adjust this based on availability and your specific needs.
# Check https://ai.google.dev/models/gemini for available models.
MODEL_NAME = "gemini-1.5-flash-latest" # Or "gemini-pro" or other compatible model

# Safety settings for content generation (adjust as needed)
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

GENERATION_CONFIG = {
    "temperature": 0.7, # Controls randomness. Lower for more factual, higher for more creative.
    "top_p": 0.95,
    "top_k": 40,
    # "max_output_tokens": 2048, # Adjust as needed, but be mindful of model limits
}

# --- End Configuration ---

def extract_pdf_text(pdf_path: str) -> str:
    """Extracts text from all pages of a PDF file."""
    doc = fitz.open(pdf_path)
    text = """"
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def generate_text_with_gemini(prompt: str) -> str | None:
    """
    Generates text using the Gemini API with a more robust retry mechanism.
    - Initial 3 retries with shorter delays.
    - Subsequent 6 retries (total 9) with longer delays.
    - Aborts if 9 consecutive failures occur for a single prompt.
    """
    model = genai.GenerativeModel(
        MODEL_NAME,
        safety_settings=SAFETY_SETTINGS,
        generation_config=GENERATION_CONFIG
    )

    max_total_attempts = 9
    consecutive_failures = 0

    # Delays in seconds: first 3 attempts have shorter delays, next 6 have longer ones.
    # Beyond the explicitly defined, it will use the last value for remaining attempts up to max_total_attempts.
    short_delay = 10  # seconds for first 3 attempts
    long_delay = 45   # seconds for attempts 4-9

    current_delay_tier = [short_delay] * 3 + [long_delay] * (max_total_attempts - 3)

    for attempt in range(max_total_attempts):
        try:
            print(f"Gemini API call attempt {attempt + 1}/{max_total_attempts}...")
            response = model.generate_content(prompt)

            # Successful response, reset consecutive failures and return text
            if response.parts:
                consecutive_failures = 0 # Reset on success
                generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                return generated_text

            # Handle cases where response.parts is empty (e.g. blocked prompt)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"WARN: Prompt blocked by API. Reason: {response.prompt_feedback.block_reason}. This counts as a failure for retry logic.")
                # This is treated as a failure by the API, so we increment failure count
            else:
                print("WARN: Received an empty response (no parts) from Gemini API. This counts as a failure for retry logic.")

            # If we reach here, it means response.parts was empty, so it's a failure for this attempt.
            # Fall through to the exception handling for retry logic, or increment failure count
            # This specific path (empty parts without an exception) will be handled as a failure below.

        except Exception as e:
            print(f"Error calling Gemini API (attempt {attempt + 1}/{max_total_attempts}): {e}")
            # This exception means an operational error occurred, count as failure.

        # If we are here, the attempt failed (either due to exception or empty/blocked response handled above)
        consecutive_failures += 1
        print(f"Consecutive API call failures: {consecutive_failures}")

        if consecutive_failures >= max_total_attempts:
            print(f"CRITICAL: Gemini API call failed {max_total_attempts} consecutive times. Aborting for this prompt.")
            return None

        # Determine delay for next retry
        # The current_delay_tier list has enough entries for all attempts up to max_total_attempts
        delay_seconds = current_delay_tier[attempt]

        print(f"Retrying in {delay_seconds} seconds...")
        time.sleep(delay_seconds)

    # Should not be reached if logic is correct, but as a fallback:
    print(f"CRITICAL: Exhausted all {max_total_attempts} attempts for the prompt. Aborting.")
    return None


def chunk_text(text: str, max_chars: int = 15000) -> list[str]: # Gemini 1.5 Flash has a large context window, but let's be safe
    """Splits text into chunks of a maximum character length."""
    # This is a simple character-based chunking. More sophisticated chunking
    # (e.g., by sentence or paragraph, or using tokenizers) might be better.
    # Max input tokens for Gemini 1.5 Flash is very large (e.g., 1M), so raw char count is a rough proxy.
    # Typical token to char ratio is ~1:4. 15k chars ~ 3.7k tokens.
    # Let's adjust this depending on average content.
    # The prompt itself also consumes tokens.

    chunks = []
    current_chunk = ""
    for sentence in text.split(". "): # Basic sentence splitting
        if len(current_chunk) + len(sentence) + 2 < max_chars: # +2 for ". "
            current_chunk += sentence + ". "
        else:
            if current_chunk: # Add the current chunk if it's not empty
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". " # Start new chunk
    if current_chunk: # Add the last chunk
        chunks.append(current_chunk.strip())
    return chunks


def generate_qa_pairs(text_content: str, num_questions_per_chunk: int = 5) -> list:
    """
    Generates Q&A pairs from the extracted text using the Google Gemini API.

    IMPORTANT: The quality, accuracy, and relevance of the Q&A pairs
    generated here are CRITICAL for successful model fine-tuning.
    Review and curate the generated pairs carefully.
    """
    all_qa_pairs = []

    # Check if API key was loaded (already done globally, but good to be aware)
    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY not configured. Cannot call Gemini API.")
        print("Please create a .env file with your GOOGLE_API_KEY or set it as an environment variable.")
        return [
            {"instruction": "Error: API Key not configured.", "input": "", "output": "Please check setup."}
        ]

    print(f"Preparing to generate Q&A pairs using Gemini model: {MODEL_NAME}")

    text_chunks = chunk_text(text_content)
    print(f"Text divided into {len(text_chunks)} chunk(s).")

    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i + 1}/{len(text_chunks)}...")

        # You can customize this prompt extensively.
        # The num_questions_per_chunk parameter is a general guideline for the total output.
        # The prompt below aims for exhaustive factual Q&A and then a few summaries,
        # guided by common CV sections.
        prompt = f"""
Your task is to act as a meticulous data extractor specializing in resumes and CVs. From the provided text chunk, you must generate two types of question-answer pairs:
1.  **Detailed Factual Q&A Pairs**: Based on common CV sections, extract specific, granular details.
2.  **Summarization Q&A Pairs**: If the chunk lends itself to it, create 1 (max 2) Q&A pairs where the question asks for a summary of a significant part of the chunk (e.g., a whole job experience if it fits, or a summary of skills).

These pairs will be used for fine-tuning a large language model.

**CV Section-Guided Instructions for Detailed Factual Q&A Generation:**

*   **If Personal Details are present (Name, Contact Info, Links):**
    *   Q&A for full name.
    *   Q&A for email address.
    *   Q&A for phone number.
    *   Q&A for LinkedIn profile URL (if present).
    *   Q&A for GitHub profile URL (if present).
    *   Q&A for personal website/portfolio URL (if present).
*   **If a Professional Summary/Objective/Personal Statement is present:**
    *   Q&A for the core message or years of experience mentioned.
    *   Q&A for specific key skills or attributes highlighted in the summary.
*   **For EACH Work Experience entry found in the chunk:**
    *   Q&A for Company Name.
    *   Q&A for Job Title/Role.
    *   Q&A for Start Date (month/year if available).
    *   Q&A for End Date (month/year if available, or "Present").
    *   Multiple Q&A pairs for distinct Key Responsibilities (be specific).
    *   Multiple Q&A pairs for distinct Achievements or Projects mentioned under that role (be specific).
    *   Q&A for specific Technologies, Tools, or Methodologies used in that role.
*   **For EACH Education entry found in the chunk:**
    *   Q&A for Institution Name.
    *   Q&A for Degree Name (e.g., Master of Science, Bachelor of Arts).
    *   Q&A for Major/Field of Study.
    *   Q&A for Graduation Date (or expected graduation date).
    *   Q&A for any specific honors, thesis, or relevant coursework mentioned.
*   **If a Skills section is present (or skills mentioned throughout):**
    *   Multiple Q&A pairs for different categories of skills (e.g., Programming Languages, Tools, Databases, Methodologies).
    *   Multiple Q&A pairs for specific skills listed under those categories.
*   **For EACH Project entry found (if separate from Work Experience):**
    *   Q&A for Project Name/Title.
    *   Q&A for your role in the project.
    *   Multiple Q&A pairs for specific details of the project description or goals.
    *   Q&A for specific Technologies or Tools used.
    *   Q&A for project outcomes or impact, if mentioned.
*   **For EACH Certification entry found:**
    *   Q&A for Certification Name.
    *   Q&A for Issuing Organization.
    *   Q&A for Date Obtained (if available).
*   **If Hobbies/Interests are mentioned:**
    *   Q&A for specific hobbies or interests listed.

**Instructions for Summarization Q&A Generation (1-2 pairs per chunk, if applicable):**
*   Based on the overall content of the chunk, identify a significant section (like a complete job description if it fits, or a summary of all skills presented) that could be summarized.
*   Formulate a question asking for a summary of that section (e.g., "Summarize the work experience at [Company X] as described in this text.", "Provide a summary of the AI/ML skills listed here.").
*   The answer should be a concise summary of that section, derived from the text chunk.

**General Instructions for ALL Q&A Pairs:**
*   **Literal and Granular for Facts**: For factual Q&A, be very literal. If a sentence contains multiple facts relevant to the above categories, try to create separate Q&A pairs.
*   **Comprehensive**: Try to cover all applicable categories and details found in the text chunk.
*   **JSON Format**: Each Q&A pair must be a JSON object with `"instruction"`, `"input": ""`, and `"output"` keys.
*   **Output Structure**: The final output MUST be a single, valid JSON list of these objects.
*   **Total Quantity**: Generate as many Q&A pairs as needed to be comprehensive according to these instructions, particularly for the factual details. Aim for a total of at least {num_questions_per_chunk} Q&A pairs if the content allows, but prioritize thoroughness based on these structured guidelines over hitting an exact number.

Provided Text Chunk:
---
{chunk}
---

Valid JSON Output (list of Q&A objects, including detailed factual pairs based on CV structure and 1-2 summarization pairs):
        """

        print(f"Sending prompt for chunk {i+1} to Gemini API (first 100 chars of prompt): {prompt[:100]}...")
        gemini_response_str = generate_text_with_gemini(prompt)

        if gemini_response_str:
            print(f"Received response from Gemini for chunk {i+1}.")
            try:
                # The response might include markdown ```json ... ```, try to extract it
                if "```json" in gemini_response_str:
                    json_str = gemini_response_str.split("```json")[1].split("```")[0].strip()
                elif "```" in gemini_response_str and gemini_response_str.startswith("["): # if it's just ```
# This part of the original code had a syntax error `[...]` which is not valid python.
# Assuming it was meant to be a comment or a simple check for '[' at the start after ```
                    json_str = gemini_response_str.split("```")[1].strip() if gemini_response_str.count("```") >= 2 else gemini_response_str.strip()
                    if not json_str.startswith("["): # If after splitting ```, it doesn't start with [, maybe it was just one ```
                         json_str = gemini_response_str.split("```")[0].strip() if gemini_response_str.startswith("[") else gemini_response_str.strip()


                elif gemini_response_str.strip().startswith("[") and gemini_response_str.strip().endswith("]"):
                    json_str = gemini_response_str.strip()
                else: # Assume the whole response is the JSON string if no markdown
                    json_str = gemini_response_str.strip()

                # print(f"Attempting to parse JSON: {json_str[:200]}...") # For debugging
                chunk_qa_pairs = json.loads(json_str)

                if isinstance(chunk_qa_pairs, list) and all(isinstance(item, dict) and "instruction" in item and "output" in item for item in chunk_qa_pairs):
                    # Add the 'input' field if missing, as per our desired format
                    for pair in chunk_qa_pairs:
                        if "input" not in pair:
                            pair["input"] = ""
                    all_qa_pairs.extend(chunk_qa_pairs)
                    print(f"Successfully parsed and added {len(chunk_qa_pairs)} Q&A pairs for chunk {i+1}.")
                else:
                    print(f"WARN: Parsed JSON for chunk {i+1} is not in the expected format (list of dicts with 'instruction' and 'output'). Skipping.")
                    print(f"Problematic JSON string: {json_str[:500]}")


            except json.JSONDecodeError as e:
                print(f"WARN: Failed to decode JSON response from Gemini for chunk {i+1}: {e}")
                print(f"Problematic Gemini response string for chunk {i+1}: {gemini_response_str[:500]}...") # Print first 500 chars
            except Exception as e:
                print(f"An unexpected error occurred while processing Gemini response for chunk {i+1}: {e}")
                print(f"Problematic Gemini response string for chunk {i+1}: {gemini_response_str[:500]}...")
        else:
            print(f"WARN: No valid response received from Gemini for chunk {i+1}.")

    if not all_qa_pairs:
        print("No Q&A pairs were generated. Returning placeholder.")
        return [
            {"instruction": "Placeholder: No Q&A generated.", "input": "", "output": "Please check PDF content or Gemini API interaction."}
        ]

    print(f"Total Q&A pairs generated: {len(all_qa_pairs)}")
    return all_qa_pairs


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF and generate Q&A pairs using Google Gemini API.")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to the input PDF file.")
    parser.add_argument("--output_json", type=str, default="qa_dataset.json", help="Path to save the generated Q&A JSON file.")
    parser.add_argument("--num_questions_per_chunk", type=int, default=10, help="Approximate number of detailed, factual Q&A pairs to generate per text chunk.")
    parser.add_argument("--max_chunk_chars", type=int, default=9000, help="Maximum characters per text chunk sent to Gemini. Smaller chunks may lead to more focused and detailed Q&A. Prompts also consume tokens.")


    args = parser.parse_args()

    if not GOOGLE_API_KEY:
        print("CRITICAL ERROR: GOOGLE_API_KEY is not set. Please create a .env file with your GOOGLE_API_KEY.")
        print("""Example .env file content:
GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY" """) # Python multiline string
        return

    print(f"Extracting text from: {args.pdf_path}")
    try:
        extracted_text = extract_pdf_text(args.pdf_path)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return

    if not extracted_text.strip():
        print("No text extracted from PDF. Exiting.")
        return

    print(f"Successfully extracted text. Total characters: {len(extracted_text)}")

    # Generate Q&A pairs using Gemini
    qa_data = generate_qa_pairs(extracted_text, args.num_questions_per_chunk)

    # Save Q&A pairs to JSON
    try:
        with open(args.output_json, "w", encoding='utf-8') as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved Q&A pairs to: {args.output_json}")
    except Exception as e:
        print(f"Error saving Q&A data to JSON: {e}")

if __name__ == "__main__":
    main()
