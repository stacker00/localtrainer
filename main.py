
import os
import re
import json
import time
import queue
import threading
import psutil
import requests

from bs4 import BeautifulSoup
import torch
import gradio as gr

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TextIteratorStreamer
)
from datasets import load_dataset
from typing import List, Dict, Any


# =============================================================================
# GLOBALS & SETTINGS
# =============================================================================

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LOCAL_MODEL_DIR = "./models"
TEACHER_MODEL_NAME = "SakanaAI/TAID-LLM-1.5B"  # Default teacher model
DEFAULT_DATASET = "./dataset.json"
FEEDBACK_DATASET = "./feedback_dataset.json"
RAG_DATA_FILE = "./rag_data.json"  # file storing additional RAG instructions
SNAPSHOT_DIR = os.path.join(LOCAL_MODEL_DIR, "snapshots")

# Ensure required directories/files exist
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

if not os.path.exists(RAG_DATA_FILE):
    with open(RAG_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)

# Dictionary that shows available model references
AVAILABLE_MODELS = {
    "Base Model": MODEL_NAME,
    "Trained Model": LOCAL_MODEL_DIR,
    "Teacher Model": TEACHER_MODEL_NAME
}

# Stop event to handle user-initiated stop of generation
stop_generation_event = threading.Event()

# Custom CSS for the Gradio interface
CUSTOM_CSS = """
/* Thinner websearch box */
#websearch_progress_container {
    max-height: 70px !important;
    overflow: auto !important;
}

/* Side by side for CoT boxes */
.cot-split {
    display: flex;
    flex-direction: row;
    gap: 10px;
}
.cot-box {
    width: 48%;
}

/* Feedback area styling */
.feedback-section {
    border: 1px solid #ddd;
    padding: 10px;
    margin-top: 10px;
    background: #f8f8f8;
}

/* Dashboard styling */
#dashboard-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 14px;
    color: #333;
}

/* Highlighted training stage styling */
.highlight {
    background-color: #fff59d;
    font-weight: bold;
}

/* More modern look for buttons */
button {
    background-color: #4a90e2 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 8px 12px !important;
    cursor: pointer !important;
}
button:hover {
    background-color: #3c7dc0 !important;
}

/* Textbox improvements */
textarea, input[type="text"] {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    font-size: 14px !important;
}
"""


# =============================================================================
# METAMASK AUTH HTML/JS
# =============================================================================

metamask_html = """
<div id="metamask_area">
  <button id="connectMMButton" style="padding:8px 12px;cursor:pointer;">Connect MetaMask</button>
  <span id="mm_status" style="margin-left:10px;">Unconnected</span>

  <script>
    const connectMMButton = document.getElementById("connectMMButton");
    const mmStatusSpan = document.getElementById("mm_status");
    const walletInput = document.getElementById("wallet_input");

    connectMMButton.onclick = async function() {
      if (typeof window.ethereum !== 'undefined') {
        try {
          const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
          const account = accounts[0] || "";
          mmStatusSpan.innerText = account.substring(0, 4) + "..." + account.slice(-4);
          walletInput.value = account;
        } catch(e) {
          console.log("MetaMask error", e);
          mmStatusSpan.innerText = "Error connecting";
        }
      } else {
        mmStatusSpan.innerText = "MetaMask not installed";
      }
    };
  </script>
</div>
"""


# =============================================================================
# DATASET & MODEL UTILITIES
# =============================================================================

def create_default_dataset() -> None:
    """
    Create a small sample JSONL dataset at DEFAULT_DATASET if it does not exist.
    This default dataset can be used for demonstration or quick training tests.
    """
    sample_data = [
        {
            "instruction": "What is AI?",
            "response": "Artificial Intelligence (AI) is the simulation of human intelligence by machines."
        },
        {
            "instruction": "Explain gravity",
            "response": "Gravity is a natural phenomenon causing masses to be attracted to each other."
        },
        {
            "instruction": "How to make tea?",
            "response": "Boil water, steep tea leaves for 3-5 minutes, then serve."
        }
    ]
    if not os.path.exists(DEFAULT_DATASET):
        with open(DEFAULT_DATASET, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")


def download_model() -> None:
    """
    Ensure the default (base) model is downloaded or cached locally.
    If the model is not found, it will be downloaded automatically.
    """
    try:
        AutoTokenizer.from_pretrained(MODEL_NAME)
        AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        print(f"Model {MODEL_NAME} is already cached locally.")
    except Exception as e:
        print(f"Downloading model {MODEL_NAME} due to: {e}")
        AutoTokenizer.from_pretrained(MODEL_NAME, force_download=True)
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, force_download=True)


def load_model(model_key: str):
    """
    Load a model and tokenizer based on the selected key from AVAILABLE_MODELS,
    or from a custom path typed by the user.

    :param model_key: Key in AVAILABLE_MODELS or a custom path string.
    :return: (model, tokenizer) tuple.
    :raises RuntimeError: if loading fails.
    """
    if model_key in AVAILABLE_MODELS:
        model_path = AVAILABLE_MODELS[model_key]
    else:
        model_path = model_key  # custom path

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")


# =============================================================================
# WEB SEARCH FUNCTIONS
# =============================================================================

def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Perform a DuckDuckGo API search for a given query string, returning up to max_results items.

    :param query: The search query text.
    :param max_results: Number of results to limit.
    :return: List of result dictionaries with 'title' and 'snippet'.
    """
    url = "https://api.duckduckgo.com"
    params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1, "skip_disambig": 1}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = []
        if "RelatedTopics" in data:
            for item in data["RelatedTopics"]:
                if "Text" in item:
                    snippet = item["Text"]
                    title = item.get("Result", item.get("FirstURL", "No Title"))
                    results.append({"title": title, "snippet": snippet})
                    if len(results) >= max_results:
                        break
        return results
    except Exception as e:
        print(f"[DuckDuckGo Error]: {e}")
        return []


def search_bing(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Perform a Bing web search by scraping the HTML result page.
    This approach is less reliable but does not require a Bing API key.

    :param query: The search query text.
    :param max_results: Number of results to limit.
    :return: List of result dictionaries with 'title', 'snippet', and 'link'.
    """
    results = []
    try:
        url = "https://www.bing.com/search"
        params = {"q": query}
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        b_algo_results = soup.find_all('li', class_='b_algo')
        for result in b_algo_results[:max_results]:
            h2 = result.find('h2')
            title = h2.get_text() if h2 else "No Title"
            a = h2.find('a') if h2 else None
            link = a['href'] if a and a.has_attr('href') else ""
            snippet_tag = result.find('p')
            snippet = snippet_tag.get_text() if snippet_tag else ""
            results.append({"title": title, "snippet": snippet, "link": link})
    except Exception as e:
        print(f"[Bing Error]: {e}")
    return results


def search_all_engines(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Combine DuckDuckGo and Bing search results into a single list,
    returning up to max_results items in total.

    :param query: Search text.
    :param max_results: Number of combined results to return.
    :return: Combined search results.
    """
    ddg_results = search_duckduckgo(query, max_results)
    bing_results = search_bing(query, max_results)
    combined = ddg_results[:]
    for res in bing_results:
        if res not in combined:
            combined.append(res)
    return combined[:max_results]


def format_search_results(results: List[Dict[str, str]]) -> str:
    """
    Format the search results into a readable string for display or prompt context.

    :param results: List of result dictionaries with 'title', 'snippet', etc.
    :return: A formatted string for easy display.
    """
    if not results:
        return "No relevant search results found."
    lines = ["Search Results:"]
    for idx, res in enumerate(results, start=1):
        lines.append(
            f"{idx}. Title: {res.get('title', 'N/A')}\n"
            f"   Snippet: {res.get('snippet', 'N/A')}\n"
        )
    return "\n".join(lines)


def perform_websearch(user_msg: str):
    """
    A generator that yields step-by-step updates about the web search,
    and finally returns a formatted summary of found results.

    :param user_msg: The search query text from the user.
    :yield: Intermediate status strings, then returns final results.
    """
    yield f"Performing web search for: '{user_msg}'..."
    time.sleep(1)
    results = search_all_engines(user_msg)
    if results:
        yield f"Found {len(results)} results."
    else:
        yield "No results found."
    return format_search_results(results)


# =============================================================================
# PROMPT & CONTEXT
# =============================================================================

def retrieve_local_context() -> str:
    """
    Load optional local context from a file named 'context.txt' if it exists.

    :return: The text content of context.txt or empty string if it doesn't exist.
    """
    context_file = "context.txt"
    if os.path.exists(context_file):
        with open(context_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def enough_resources_for_parallel() -> bool:
    """
    Check if approximately 2GB of free memory (GPU VRAM or CPU RAM) is available,
    to allow parallel generation of two Chains-of-Thought.

    :return: True if enough memory is available, False otherwise.
    """
    threshold = 2 * 1024 * 1024 * 1024  # 2GB
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_vram = props.total_memory - torch.cuda.memory_allocated(0)
        return free_vram > threshold
    else:
        mem = psutil.virtual_memory()
        return mem.available > threshold


def parse_cot_and_answer(full_text: str) -> (str, str):
    """
    Split the output text into the chain-of-thought (CoT) part and final answer.
    If 'Final Answer:' is found, everything before it is CoT, everything after is the final answer.
    Otherwise, the entire text is treated as both CoT and final answer.

    :param full_text: The model-generated text containing reasoning and final answer.
    :return: (cot, ans) string tuple.
    """
    if "Final Answer:" in full_text:
        parts = full_text.split("Final Answer:", 1)
        cot = parts[0].strip()
        ans = parts[1].strip()
    else:
        cot = full_text.strip()
        ans = full_text.strip()
    return cot, ans


def engineer_prompt(user_query: str, web_results: str, mode: str, rag_instructions: str = "") -> str:
    """
    Assemble a prompt to be given to the language model, depending on the specified mode.

    :param user_query: The user's question or instruction.
    :param web_results: Formatted results from web searches.
    :param mode: Either "Simple Conversation" or "Teaching".
    :param rag_instructions: Optional instructions from RAG data for more context.
    :return: A final prompt string to pass to the model.
    """
    local_context = retrieve_local_context()
    if rag_instructions.strip():
        rag_str = f"Additional RAG/Prompt Instructions:\n{rag_instructions}\n\n"
    else:
        rag_str = ""

    if mode == "Teaching":
        prompt = (
            f"{rag_str}"
            "You are a highly intelligent assistant that explains your reasoning step by step, then provides a final answer.\n\n"
            f"Web Results:\n{web_results}\n\nLocal Context:\n{local_context}\n\n"
            f"Instruction: {user_query}\n\nChain-of-Thought:\n"
        )
    else:
        # Simple conversation
        prompt = (
            f"{rag_str}"
            "You are a helpful assistant. Provide a concise answer.\n\n"
            f"Instruction: {user_query}\n\nAnswer:"
        )
    return prompt


# =============================================================================
# STREAMING & PARALLEL GENERATION
# =============================================================================

def partial_token_stream(model, tokenizer, prompt, temperature=0.9):
    """
    A generator that streams tokens from the model's generate() method using a TextIteratorStreamer.
    This allows partial updates to be returned as the model continues to generate.

    :param model: A loaded AutoModelForCausalLM instance.
    :param tokenizer: The corresponding AutoTokenizer instance.
    :param prompt: Prompt text to feed the model.
    :param temperature: Sampling temperature for generation.
    :yield: Incrementally built text as tokens are generated.
    """
    text_so_far = ""
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    def gen_thread():
        model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=temperature, streamer=streamer)

    t = threading.Thread(target=gen_thread)
    t.start()
    for token in streamer:
        if stop_generation_event.is_set():
            break
        text_so_far += token
        yield text_so_far
    t.join()
    return text_so_far


def stream_chat_response(
    user_msg: str,
    history: List[Dict[str, str]],
    model_key: str,
    mode: str,
    include_web: bool,
    rag_instructions: str,
    use_teacher_for_cot2: bool
):
    """
    Main function to handle user chat input and model responses, potentially with two parallel CoTs.

    :param user_msg: User's message.
    :param history: List of dict items representing conversation history.
    :param model_key: Model key or path for the main model.
    :param mode: "Simple Conversation" or "Teaching".
    :param include_web: Whether to perform a web search first.
    :param rag_instructions: Any RAG instruction text to include in the prompt.
    :param use_teacher_for_cot2: If True, load the teacher model for the second CoT.
    :yield: Each step returns an 8-tuple of partial states to update the UI in real-time:
        (user_msg, new_history, partial_cot1, partial_cot2, ans1, ans2, web_progress, final_answer).
    """
    stop_generation_event.clear()
    websearch_progress = ""
    web_results_str = ""

    # Optionally perform web search if in "Teaching" mode
    if include_web and (mode == "Teaching"):
        gen = perform_websearch(user_msg)
        try:
            while True:
                update = next(gen)
                websearch_progress = update
                yield (user_msg, history, "", "", "", "", websearch_progress, "")
        except StopIteration as exc:
            web_results_str = exc.value or ""

    # Load the selected model and build the final prompt
    model, tokenizer = load_model(model_key)
    prompt = engineer_prompt(user_msg, web_results_str, mode, rag_instructions=rag_instructions)

    # If the mode is "Simple", do a single pass generation
    if mode == "Simple Conversation":
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=300,
            num_beams=4,
            early_stopping=True
        )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cot, ans = parse_cot_and_answer(full_text)
        final_msg = {"role": "assistant", "content": ans}
        new_history = history + [{"role": "user", "content": user_msg}, final_msg]
        yield ("", new_history, cot, "", ans, "", websearch_progress, ans)
        return

    # "Teaching" mode: attempt parallel CoTs if memory is sufficient
    parallel_ok = enough_resources_for_parallel()

    # Decide if we should load a teacher model
    if use_teacher_for_cot2:
        teacher_model, teacher_tokenizer = load_model("Teacher Model")
    else:
        teacher_model = model
        teacher_tokenizer = tokenizer

    if parallel_ok:
        # Parallel generation using two threads
        q = queue.Queue()

        def generate_cot(label, used_model, used_tokenizer, used_prompt):
            streamer = TextIteratorStreamer(used_tokenizer, skip_special_tokens=True)
            inputs = used_tokenizer(used_prompt, return_tensors="pt", truncation=True, max_length=2048)

            def gen_thread():
                used_model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.9, streamer=streamer)

            t = threading.Thread(target=gen_thread)
            t.start()
            text_so_far = ""
            for token in streamer:
                if stop_generation_event.is_set():
                    break
                text_so_far += token
                q.put((label, text_so_far, False))
            t.join()
            q.put((label, text_so_far, True))

        # Launch both threads
        t1 = threading.Thread(target=generate_cot, args=("cot1", model, tokenizer, prompt))
        t2 = threading.Thread(target=generate_cot, args=("cot2", teacher_model, teacher_tokenizer, prompt))
        t1.start()
        t2.start()

        partial_cot1 = ""
        partial_cot2 = ""
        done1 = False
        done2 = False

        # Collect partial updates from the threads
        while not (done1 and done2):
            try:
                label, text, is_done = q.get(timeout=1)
                if label == "cot1":
                    partial_cot1 = text
                    if is_done:
                        done1 = True
                else:
                    partial_cot2 = text
                    if is_done:
                        done2 = True
                yield ("", history, partial_cot1, partial_cot2, "", "", websearch_progress, "")
            except queue.Empty:
                continue

        # After generation finishes, parse both CoTs
        cot1, ans1 = parse_cot_and_answer(partial_cot1)
        cot2, ans2 = parse_cot_and_answer(partial_cot2)
        final_msg = {
            "role": "assistant",
            "content": ans1,
            "cot_1": cot1,
            "cot_2": cot2
        }
        new_history = history + [{"role": "user", "content": user_msg}, final_msg]
        yield ("", new_history, cot1, cot2, ans1, ans2, websearch_progress, ans1)

    else:
        # Fallback to sequential if not enough resources for parallel
        partial_cot1 = ""
        for partial_text in partial_token_stream(model, tokenizer, prompt):
            if stop_generation_event.is_set():
                break
            partial_cot1 = partial_text
            yield ("", history, partial_cot1, "", "", "", websearch_progress, "")

        partial_cot2 = ""
        for partial_text in partial_token_stream(teacher_model, teacher_tokenizer, prompt):
            if stop_generation_event.is_set():
                break
            partial_cot2 = partial_text
            yield ("", history, partial_cot1, partial_cot2, "", "", websearch_progress, "")

        # Final parse
        cot1, ans1 = parse_cot_and_answer(partial_cot1)
        cot2, ans2 = parse_cot_and_answer(partial_cot2)
        final_msg = {
            "role": "assistant",
            "content": ans1,
            "cot_1": cot1,
            "cot_2": cot2
        }
        new_history = history + [{"role": "user", "content": user_msg}, final_msg]
        yield ("", new_history, cot1, cot2, ans1, ans2, websearch_progress, ans1)


def stop_generation_callback() -> str:
    """
    Signal the stop_generation_event to halt any ongoing generation.

    :return: A message confirming the stop.
    """
    stop_generation_event.set()
    return "Generation stopped."


# =============================================================================
# COT SPLITTING & RATING UTILITIES
# =============================================================================

def split_cot_into_steps(cot_text: str, initial_rating: str = "0") -> str:
    """
    Split a multi-line chain-of-thought into separate steps, appending a rating bracket to each line.

    :param cot_text: The chain-of-thought text to split.
    :param initial_rating: The default rating to place in each step.
    :return: Reconstructed multiline text with each line containing a rating bracket, e.g. "[Rating: X]".
    """
    lines = [line.strip() for line in cot_text.split("\n") if line.strip()]
    rated_lines = []
    for step_text in lines:
        rated_lines.append(f"{step_text} [Rating: {initial_rating}]")
    return "\n".join(rated_lines)


def parse_line_ratings(steps_text: str) -> str:
    """
    Parse each line in the steps_text to extract [Rating: #], returning the rating values as a comma-separated string.

    :param steps_text: Multiline text with lines containing "[Rating: #]".
    :return: A comma-separated string of all rating values in the same order.
    """
    lines = steps_text.strip().split("\n")
    line_ratings = []
    for line in lines:
        match = re.search(r"\[Rating:\s*([\-0-9]+)\]", line)
        if match:
            line_ratings.append(match.group(1))
        else:
            line_ratings.append("0")
    return ", ".join(line_ratings)


def update_ratings_in_steps(steps_text: str, ratings_text: str) -> str:
    """
    Overwrite the ratings in each line of steps_text using the corresponding ratings
    from a comma-separated ratings_text.

    :param steps_text: Existing multiline text of steps with "[Rating: #]".
    :param ratings_text: Comma-separated rating values.
    :return: Updated steps text with new rating values in place.
    """
    lines = steps_text.strip().split("\n")
    new_ratings = [r.strip() for r in ratings_text.split(",") if r.strip()]
    updated_lines = []
    for i, line in enumerate(lines):
        match = re.search(r"\[Rating:\s*([\-\d]+)\]", line)
        if match:
            prefix = line[:match.start()]
            suffix = line[match.end():]
        else:
            prefix = line
            suffix = ""
        rating_str = new_ratings[i] if i < len(new_ratings) else "0"
        new_line = f"{prefix.strip()} [Rating: {rating_str}]{suffix}"
        updated_lines.append(new_line.strip())
    return "\n".join(updated_lines)


def save_cot_splitting_feedback(
    cot1_steps: str,
    rating_steps_1: str,
    supervised_cot1: str,
    supervised_ans1: str,
    cot2_steps: str,
    rating_steps_2: str,
    supervised_cot2: str,
    supervised_ans2: str
) -> str:
    """
    Append a single record to feedback_dataset.json with step ratings
    and supervised feedback for both CoTs and final answers.

    :param cot1_steps: The multiline text of CoT #1 steps (with [Rating: #]).
    :param rating_steps_1: A comma-separated list of numeric ratings for each step in CoT #1.
    :param supervised_cot1: Human-corrected chain-of-thought for CoT #1.
    :param supervised_ans1: Human-corrected final answer for CoT #1.
    :param cot2_steps: Same for CoT #2 steps.
    :param rating_steps_2: Comma-separated list of numeric ratings for CoT #2 steps.
    :param supervised_cot2: Human-corrected chain-of-thought for CoT #2.
    :param supervised_ans2: Human-corrected final answer for CoT #2.
    :return: A status message.
    """
    entry = {
        "chat_history": f"CoT1 Steps:\n{cot1_steps}\n\nCoT2 Steps:\n{cot2_steps}",
        "chosen_cot": "",
        "cot1_rating": "",
        "cot2_rating": "",
        "step_rankings": f"CoT1 Step Ratings: {rating_steps_1}\nCoT2 Step Ratings: {rating_steps_2}",
        "step_supervised": f"CoT1: {supervised_cot1}\nCoT2: {supervised_cot2}",
        "full_replacement1": supervised_ans1,
        "full_replacement2": supervised_ans2,
        "rag_instruction": ""
    }
    try:
        with open(FEEDBACK_DATASET, "a", encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
        return "CoT feedback saved successfully."
    except Exception as e:
        return f"Error saving feedback: {e}"


# =============================================================================
# ADVANCED TRAINING (RLHF, Supervised, Teacher)
# =============================================================================

def train_model_advanced(
    dataset_path: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    output_dir: str,
    model_key: str,
    training_mode: str,
    teacher_model_path: str
):
    """
    A generator-based function to perform advanced training (RLHF, Supervised, Teacher).
    Yields (progress_msg, logs_msg) at each major step.

    :param dataset_path: Path to a JSON or JSONL dataset.
    :param learning_rate: The learning rate to use during training.
    :param batch_size: The batch size per device.
    :param epochs: Number of training epochs.
    :param output_dir: Where to save fine-tuned model and tokenizer.
    :param model_key: The base or custom model path/key for training.
    :param training_mode: "RLHF", "Supervised", or "Teacher".
    :param teacher_model_path: Path for the teacher model if needed.
    :yield: Tuple (status_message, logs_message) for UI updates.
    """
    # Stage 1: Check/Create default dataset
    yield ("Stage 1: Checking/Creating default dataset...", "")
    create_default_dataset()

    # Stage 2: Load dataset
    yield ("Stage 2: Loading dataset...", "")
    if not os.path.isfile(dataset_path):
        yield (f"Error: Dataset file not found at {dataset_path}", "")
        return
    try:
        dataset = load_dataset('json', data_files=dataset_path)['train']
        yield ("Dataset loaded.", "Dataset loaded.")
    except Exception as e:
        yield (f"Error loading dataset: {e}", "")
        return

    # Stage 3: Tokenize
    yield ("Stage 3: Tokenizing dataset...", "")
    try:
        def format_prompt(example):
            # Combine instruction and response into a single text field
            return {"input_text": f"Instruction: {example['instruction']}\nResponse: {example['response']}"}

        dataset = dataset.map(format_prompt)

        model, tokenizer = load_model(model_key)

        def tokenize_fn(examples):
            return tokenizer(examples['input_text'], truncation=True, padding='max_length', max_length=512)

        tokenized_dataset = dataset.map(tokenize_fn, batched=True)
        yield ("Tokenization complete.", "Tokenization complete.")
    except Exception as e:
        yield (f"Tokenization error: {e}", "")
        return

    # Stage 4: (Optional) Teacher model usage
    if training_mode == "Teacher":
        # Attempt to load the teacher model
        try:
            teacher_model, teacher_tokenizer = load_model("Teacher Model")
            yield ("Teacher model loaded: " + TEACHER_MODEL_NAME, "Teacher model loaded.")
        except Exception as e:
            yield (f"Warning: Could not load teacher model: {e}", "Skipping teacher model usage.")

    # Stage 5: Training
    yield (f"Stage 4 (or 5): Starting {training_mode} training...", "Training started.")
    try:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            evaluation_strategy="steps",
            eval_steps=200,
            fp16=True,
            report_to="none"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        trainer.train()
        yield ("Training completed.", f"{training_mode} training done.")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        yield (f"Model saved in {output_dir}.", f"Model saved in {output_dir}.")
    except Exception as e:
        yield (f"Training error: {e}", "")


# =============================================================================
# DASHBOARD UTILS
# =============================================================================

def get_dashboard_data() -> str:
    """
    Collect and return some basic usage metrics: number of feedback entries, CPU memory,
    and GPU VRAM availability.

    :return: A formatted string summarizing the dashboard info.
    """
    total_feedback = 0
    if os.path.exists(FEEDBACK_DATASET):
        with open(FEEDBACK_DATASET, 'r', encoding='utf-8') as f:
            total_feedback = sum(1 for line in f if line.strip())
    cpu_mem = psutil.virtual_memory().available / (1024 ** 3)
    gpu_info = "No GPU available"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        free_vram = total_vram - allocated
        gpu_info = f"GPU VRAM: {free_vram:.2f}GB free / {total_vram:.2f}GB total"

    info = (
        f"Feedback entries: {total_feedback}\n"
        f"Available CPU Mem: {cpu_mem:.2f} GB\n"
        f"{gpu_info}"
    )
    return info


# =============================================================================
# FEEDBACK HISTORY
# =============================================================================

def load_feedback() -> List[Any]:
    """
    Load feedback entries from FEEDBACK_DATASET (JSONL).
    Each line is parsed as a JSON record.

    :return: A list of feedback dictionary entries.
    """
    feedback_entries = []
    if os.path.exists(FEEDBACK_DATASET):
        with open(FEEDBACK_DATASET, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    feedback_entries.append(json.loads(line))
    return feedback_entries


def delete_feedback_row(feedback_data, index):
    """
    Delete a specific row from the feedback data list, then rewrite feedback_dataset.json.

    :param feedback_data: The current in-memory list-of-dicts (or list-of-lists).
    :param index: The integer index of the row to delete.
    :return: Updated data and a status message.
    """
    try:
        idx = int(index)
        if 0 <= idx < len(feedback_data):
            del feedback_data[idx]
        with open(FEEDBACK_DATASET, 'w', encoding='utf-8') as f:
            for entry in feedback_data:
                f.write(json.dumps(entry) + "\n")
        return feedback_data, "Feedback row deleted."
    except Exception as e:
        return feedback_data, f"Error: {e}"


def save_feedback_changes(feedback_data):
    """
    Save the entire feedback_data structure back to FEEDBACK_DATASET.
    The data is expected to match the 9 columns used in the UI.

    :param feedback_data: A list of either dicts or lists representing the feedback.
    :return: (updated_data, status_message)
    """
    updated = []
    for row in feedback_data:
        if isinstance(row, dict):
            updated.append(row)
        else:
            updated.append({
                "chat_history": row[0] if len(row) > 0 else "",
                "chosen_cot": row[1] if len(row) > 1 else "",
                "cot1_rating": row[2] if len(row) > 2 else "",
                "cot2_rating": row[3] if len(row) > 3 else "",
                "step_rankings": row[4] if len(row) > 4 else "",
                "step_supervised": row[5] if len(row) > 5 else "",
                "full_replacement1": row[6] if len(row) > 6 else "",
                "full_replacement2": row[7] if len(row) > 7 else "",
                "rag_instruction": row[8] if len(row) > 8 else "",
            })
    try:
        with open(FEEDBACK_DATASET, 'w', encoding='utf-8') as f:
            for entry in updated:
                f.write(json.dumps(entry) + "\n")
        return updated, "Feedback changes saved."
    except Exception as e:
        return feedback_data, f"Error: {e}"


# =============================================================================
# RAG DATA MANAGEMENT
# =============================================================================

def load_rag_data() -> List[str]:
    """
    Load the RAG instructions list from RAG_DATA_FILE.

    :return: A list of RAG instruction strings.
    """
    if not os.path.exists(RAG_DATA_FILE):
        return []
    try:
        with open(RAG_DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []


def save_rag_data(rag_list: List[str]) -> None:
    """
    Write the entire list of RAG instructions back to RAG_DATA_FILE in JSON format.

    :param rag_list: A list of instruction strings.
    """
    with open(RAG_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(rag_list, f, indent=2)


def add_rag_instruction(new_instruction: str) -> str:
    """
    Append a new instruction to the RAG list and save to RAG_DATA_FILE.

    :param new_instruction: The instruction text to add.
    :return: A status message.
    """
    rag_list = load_rag_data()
    rag_list.append(new_instruction.strip())
    save_rag_data(rag_list)
    return "New RAG instruction saved."


# =============================================================================
# BUILD GRADIO INTERFACE
# =============================================================================

with gr.Blocks(css=CUSTOM_CSS, title="BTC+ AI Trainer") as app:
    gr.Markdown("## BTC+ No code AI Trainer\nAdvanced CoT, RLHF, teacher LLM, etc.")

    # Global states
    conversation_state = gr.State([])

    # ------------------- TAB: Chat with Model -------------------
    with gr.Tab("Chat with Model"):
        # MetaMask
        metamask_component = gr.HTML(metamask_html)
        wallet_id = gr.Textbox(
            label="Wallet ID",
            value="Unconnected",
            visible=True,
            interactive=False,
            elem_id="wallet_input"
        )

        with gr.Row():
            user_msg = gr.Textbox(label="Your Message", lines=2)
            mode_radio = gr.Radio(
                choices=["Simple Conversation", "Teaching"],
                value="Simple Conversation",
                label="Mode"
            )
            web_search_check = gr.Checkbox(label="Include Web Search?", value=False)
            model_selector = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value="Base Model",
                label="Select Model",
                allow_custom_value=True
            )
            rag_instructions_for_chat = gr.Textbox(label="RAG Instruction for Chat", lines=2, value="")

        with gr.Row():
            use_teacher_for_cot2_check = gr.Checkbox(label="Use Teacher Model for CoT #2?", value=False)
            submit_btn = gr.Button("Submit")
            stop_btn = gr.Button("Stop Generation")

        websearch_progress = gr.Textbox(
            label="Websearch Progress",
            interactive=False,
            lines=3,
            elem_id="websearch_progress_container"
        )

        chatbot = gr.Chatbot(label="Conversation", type="messages")

        with gr.Row(elem_classes="cot-split"):
            cot1_box = gr.Textbox(label="CoT #1", lines=8, elem_classes="cot-box", interactive=False)
            cot2_box = gr.Textbox(label="CoT #2", lines=8, elem_classes="cot-box", interactive=False)

        with gr.Row():
            ans_box_1 = gr.Textbox(label="Answer #1", lines=2, interactive=False)
            ans_box_2 = gr.Textbox(label="Answer #2", lines=2, interactive=False)

        with gr.Row():
            cot1_rating_radio = gr.Radio(choices=["-1", "0", "1"], value="0", label="CoT #1 Rating")
            cot2_rating_radio = gr.Radio(choices=["-1", "0", "1"], value="0", label="CoT #2 Rating")

        with gr.Row():
            choose_cot_radio = gr.Radio(choices=["cot1", "cot2"], label="Select Best Answer", value="cot1")
            apply_selection_btn = gr.Button("Apply Selection")
            final_answer_box = gr.Textbox(label="Final Answer (Selected)", lines=2, interactive=False)

        def on_submit(msg, hist, model_key, mode, web, rag_inst, teacher_cot2):
            """
            Wrapper around stream_chat_response generator.
            Yields partial state updates step-by-step for the UI.
            """
            try:
                generator = stream_chat_response(
                    user_msg=msg,
                    history=hist,
                    model_key=model_key,
                    mode=mode,
                    include_web=web,
                    rag_instructions=rag_inst,
                    use_teacher_for_cot2=teacher_cot2
                )
                for step in generator:
                    yield step
            except Exception as e:
                yield (msg, hist, "", "", "", "", f"Error: {e}", "")

        submit_btn.click(
            fn=on_submit,
            inputs=[
                user_msg,
                conversation_state,
                model_selector,
                mode_radio,
                web_search_check,
                rag_instructions_for_chat,
                use_teacher_for_cot2_check
            ],
            outputs=[
                user_msg,            # 0
                chatbot,             # 1
                cot1_box,            # 2
                cot2_box,            # 3
                ans_box_1,           # 4
                ans_box_2,           # 5
                websearch_progress,  # 6
                final_answer_box     # 7
            ],
            queue=True
        )

        stop_btn.click(fn=stop_generation_callback, inputs=[], outputs=[websearch_progress])

        def apply_selection(choice, a1, a2, old_rating_1, old_rating_2):
            """
            When the user selects which CoT is best, set that CoT's rating to '1'.
            If the other CoT wasn't marked as '-1', set it to '0'.
            Update final answer to the chosen answer.
            """
            new_rating_1 = old_rating_1
            new_rating_2 = old_rating_2
            final_ans = ""
            if choice == "cot1":
                final_ans = a1
                new_rating_1 = "1"
                if new_rating_2 != "-1":
                    new_rating_2 = "0"
            else:
                final_ans = a2
                new_rating_2 = "1"
                if new_rating_1 != "-1":
                    new_rating_1 = "0"
            return final_ans, new_rating_1, new_rating_2

        apply_selection_btn.click(
            fn=apply_selection,
            inputs=[choose_cot_radio, ans_box_1, ans_box_2, cot1_rating_radio, cot2_rating_radio],
            outputs=[final_answer_box, cot1_rating_radio, cot2_rating_radio]
        )

        def update_chat_state(new_hist):
            """
            Update the global conversation state if the chatbot changes.
            """
            return new_hist

        chatbot.change(fn=update_chat_state, inputs=[chatbot], outputs=[conversation_state])

    # ------------------- TAB: CoT Splitting & Ratings -------------------
    with gr.Tab("CoT Splitting & Ratings"):
        gr.Markdown(
            "### Automatically load/split the generated CoTs from the Chat tab, "
            "then edit step ratings and add supervised feedback."
        )

        split_cot1_btn = gr.Button("Split CoT #1 (Load from Chat)")
        rating_for_cot1_split = gr.Textbox(label="CoT #1 Rating", value="0", interactive=False)
        steps_box_1 = gr.Textbox(label="CoT #1 Steps with Ratings", lines=8)

        split_cot2_btn = gr.Button("Split CoT #2 (Load from Chat)")
        rating_for_cot2_split = gr.Textbox(label="CoT #2 Rating", value="0", interactive=False)
        steps_box_2 = gr.Textbox(label="CoT #2 Steps with Ratings", lines=8)

        rating_steps_1 = gr.Textbox(label="Ratings for CoT #1 Steps (comma-separated)")
        update_steps_rating_1 = gr.Button("Update Ratings (CoT #1 Steps)")

        rating_steps_2 = gr.Textbox(label="Ratings for CoT #2 Steps (comma-separated)")
        update_steps_rating_2 = gr.Button("Update Ratings (CoT #2 Steps)")

        supervised_cot1 = gr.Textbox(label="CoT #1 Supervised Correction", lines=4)
        supervised_ans1 = gr.Textbox(label="Final Answer #1 Supervised Correction", lines=2)

        supervised_cot2 = gr.Textbox(label="CoT #2 Supervised Correction", lines=4)
        supervised_ans2 = gr.Textbox(label="Final Answer #2 Supervised Correction", lines=2)

        save_cot_feedback_btn = gr.Button("Save CoT Feedback")
        cot_feedback_status = gr.Textbox(label="Feedback Save Status", interactive=False)

        def load_cot1_from_chat(cot_text_1, cot1_rating):
            steps = split_cot_into_steps(cot_text_1, initial_rating=cot1_rating)
            auto_step_ratings = parse_line_ratings(steps)
            return cot1_rating, steps, auto_step_ratings

        split_cot1_btn.click(
            fn=load_cot1_from_chat,
            inputs=[cot1_box, cot1_rating_radio],
            outputs=[rating_for_cot1_split, steps_box_1, rating_steps_1]
        )

        def load_cot2_from_chat(cot_text_2, cot2_rating):
            steps = split_cot_into_steps(cot_text_2, initial_rating=cot2_rating)
            auto_step_ratings = parse_line_ratings(steps)
            return cot2_rating, steps, auto_step_ratings

        split_cot2_btn.click(
            fn=load_cot2_from_chat,
            inputs=[cot2_box, cot2_rating_radio],
            outputs=[rating_for_cot2_split, steps_box_2, rating_steps_2]
        )

        def do_update_steps1(steps_text, new_ratings):
            return update_ratings_in_steps(steps_text, new_ratings)

        update_steps_rating_1.click(
            fn=do_update_steps1,
            inputs=[steps_box_1, rating_steps_1],
            outputs=steps_box_1
        )

        def do_update_steps2(steps_text, new_ratings):
            return update_ratings_in_steps(steps_text, new_ratings)

        update_steps_rating_2.click(
            fn=do_update_steps2,
            inputs=[steps_box_2, rating_steps_2],
            outputs=steps_box_2
        )

        def on_save_cot_feedback(
            scot1_steps, scot1_ratings, scot1_supervised, sans1_supervised,
            scot2_steps, scot2_ratings, scot2_supervised, sans2_supervised
        ):
            return save_cot_splitting_feedback(
                scot1_steps,
                scot1_ratings,
                scot1_supervised,
                sans1_supervised,
                scot2_steps,
                scot2_ratings,
                scot2_supervised,
                sans2_supervised
            )

        save_cot_feedback_btn.click(
            fn=on_save_cot_feedback,
            inputs=[
                steps_box_1, rating_steps_1, supervised_cot1, supervised_ans1,
                steps_box_2, rating_steps_2, supervised_cot2, supervised_ans2
            ],
            outputs=[cot_feedback_status]
        )

    # ------------------- TAB: Training Monitor -------------------
    with gr.Tab("Training Monitor"):
        gr.Markdown(
            "### Training Pipeline Diagram (DeepSeek‐R1 Steps)\n"
            "- RLHF\n- Supervised Fine Tuning\n- Teacher LLM\n"
            "We also show the baseline model path, and additional info below.\n"
        )

        model_list_text = gr.Textbox(label="Available Models Info", lines=3, interactive=False)
        base_model_info = (
            f"Base Model: {MODEL_NAME}\n"
            f"Trained Model: {LOCAL_MODEL_DIR}\n"
            f"Teacher Model: {TEACHER_MODEL_NAME}"
        )
        model_list_text.value = base_model_info

        training_status = gr.Textbox(label="Training Status", lines=3, interactive=False)
        training_logs = gr.Textbox(label="Training Logs", lines=10, interactive=False)

        with gr.Row():
            dataset_path_box = gr.Textbox(label="Dataset Path", value=DEFAULT_DATASET)
            output_dir_box = gr.Textbox(label="Output Dir", value=LOCAL_MODEL_DIR)

        with gr.Row():
            lr_box = gr.Number(label="Learning Rate", value=3e-4)
            bs_box = gr.Slider(minimum=1, maximum=32, step=1, value=4, label="Batch Size")
            epochs_box = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Epochs")

        training_mode_radio = gr.Radio(
            choices=["RLHF", "Supervised", "Teacher"],
            value="Supervised",
            label="Training Mode"
        )
        teacher_model_path_box = gr.Textbox(
            label="Teacher Model Path (optional)",
            value=TEACHER_MODEL_NAME,
            lines=1
        )

        training_model_selector = gr.Dropdown(
            choices=list(AVAILABLE_MODELS.keys()),
            value="Base Model",
            label="Model for Training",
            allow_custom_value=True
        )

        start_training_btn = gr.Button("Start Training")

        def train_wrapper(dspath, lr, bs, ep, outdir, modelkey, mode_val, teacher_path):
            """
            Wrap the train_model_advanced generator, yielding stepwise outputs
            for training_status and training_logs.
            """
            final_result = None
            for step in train_model_advanced(dspath, lr, bs, ep, outdir, modelkey, mode_val, teacher_path):
                final_result = step
                yield step
            return final_result

        start_training_btn.click(
            fn=train_wrapper,
            inputs=[
                dataset_path_box,
                lr_box,
                bs_box,
                epochs_box,
                output_dir_box,
                training_model_selector,
                training_mode_radio,
                teacher_model_path_box
            ],
            outputs=[training_status, training_logs]
        )

    # ------------------- TAB: Prompt Engineering -------------------
    with gr.Tab("Prompt Engineering"):
        gr.Markdown("### Custom Prompt Generation with RAG Data")

        model_selector_prompt = gr.Dropdown(
            choices=list(AVAILABLE_MODELS.keys()),
            value="Base Model",
            label="Select Model"
        )
        prompt_box = gr.Textbox(
            label="Custom Prompt",
            lines=10,
            value="You are a wise AI. Q: What is the capital of France?"
        )
        rag_load_dropdown = gr.Dropdown(label="Select RAG Instruction", choices=[], value="")
        load_rag_btn = gr.Button("Load RAG Data")
        combined_prompt_box = gr.Textbox(label="Combined Prompt (with RAG)", lines=10, interactive=False)
        generate_btn = gr.Button("Generate")
        raw_output_box = gr.Textbox(label="LLM Output", lines=10)

        def load_rag_choices():
            rag_list = load_rag_data()
            return rag_list if rag_list else ["(No RAG instructions saved yet)"]

        load_rag_btn.click(fn=load_rag_choices, outputs=rag_load_dropdown)

        def combine_prompt_with_rag(prompt_text, rag_choice):
            if rag_choice and "(No RAG instructions" not in rag_choice:
                return prompt_text + "\n\nRAG Instruction:\n" + rag_choice
            else:
                return prompt_text

        rag_load_dropdown.change(
            fn=combine_prompt_with_rag,
            inputs=[prompt_box, rag_load_dropdown],
            outputs=[combined_prompt_box]
        )

        def custom_prompt_infer(prompt_text, model_key):
            try:
                model, tokenizer = load_model(model_key)
                inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
                outputs = model.generate(**inputs, max_new_tokens=512, num_beams=1, do_sample=True)
                return tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                return f"Error: {e}"

        generate_btn.click(
            fn=custom_prompt_infer,
            inputs=[combined_prompt_box, model_selector_prompt],
            outputs=raw_output_box
        )

    # ------------------- TAB: RAG Instructions -------------------
    with gr.Tab("RAG Instructions"):
        gr.Markdown("#### Manage Additional RAG/Prompt Engineering Instructions (in rag_data.json)")
        rag_instructions_display = gr.Textbox(
            label="Saved RAG Instructions (two newlines separated)",
            lines=10,
            interactive=False
        )
        load_rag_instructions_btn = gr.Button("Load All RAG Instructions")
        new_rag_box = gr.Textbox(label="New RAG Instruction to Save", lines=3)
        save_new_rag_btn = gr.Button("Save New RAG Instruction")
        rag_status = gr.Textbox(label="RAG Save Status", interactive=False)

        def load_all_rag_instructions():
            rag_list = load_rag_data()
            return "\n\n".join(rag_list)

        load_rag_instructions_btn.click(fn=load_all_rag_instructions, outputs=rag_instructions_display)

        def save_new_rag_instruction(rag_text):
            if not rag_text.strip():
                return "Error: Instruction is empty."
            return add_rag_instruction(rag_text)

        save_new_rag_btn.click(fn=save_new_rag_instruction, inputs=[new_rag_box], outputs=rag_status)

    # ------------------- TAB: Dashboard -------------------
    with gr.Tab("Dashboard"):
        dash_box = gr.Textbox(label="Dashboard Info", lines=4, elem_id="dashboard-container")
        dash_btn = gr.Button("Refresh Dashboard")
        dash_btn.click(fn=get_dashboard_data, outputs=dash_box)

    # ------------------- TAB: Feedback History -------------------
    with gr.Tab("Feedback History"):
        gr.Markdown("### View / Edit Saved Feedback")
        feedback_df = gr.Dataframe(
            label="Feedback History",
            headers=[
                "chat_history", "chosen_cot", "cot1_rating", "cot2_rating",
                "step_rankings", "step_supervised", "full_replacement1",
                "full_replacement2", "rag_instruction"
            ],
            interactive=True,
            row_count=(0, "dynamic"),
            col_count=(9, "fixed")
        )
        load_feedback_btn = gr.Button("Load Feedback")
        del_index_box = gr.Textbox(label="Index to Delete", placeholder="Enter index #", lines=1)
        delete_fb_btn = gr.Button("Delete Row")
        fb_edit_status = gr.Textbox(label="Status", interactive=False)
        save_fb_changes_btn = gr.Button("Save Feedback Changes")

        def update_feedback_display():
            fb_list = load_feedback()
            rows = []
            for entry in fb_list:
                rows.append([
                    entry.get("chat_history", ""),
                    entry.get("chosen_cot", ""),
                    entry.get("cot1_rating", ""),
                    entry.get("cot2_rating", ""),
                    entry.get("step_rankings", ""),
                    entry.get("step_supervised", ""),
                    entry.get("full_replacement1", ""),
                    entry.get("full_replacement2", ""),
                    entry.get("rag_instruction", ""),
                ])
            return rows

        load_feedback_btn.click(fn=update_feedback_display, outputs=feedback_df)

        def delete_fb(feedback_data, idx_str):
            updated_data, status = delete_feedback_row(feedback_data, idx_str)
            return updated_data, status

        delete_fb_btn.click(fn=delete_fb, inputs=[feedback_df, del_index_box], outputs=[feedback_df, fb_edit_status])

        def save_fb_changes_fn(feedback_data):
            updated, status = save_feedback_changes(feedback_data)
            return updated, status

        save_fb_changes_btn.click(
            fn=save_fb_changes_fn,
            inputs=[feedback_df],
            outputs=[feedback_df, fb_edit_status]
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    create_default_dataset()
    download_model()
    app.launch()
