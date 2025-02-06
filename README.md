
# localtrainer: No-Code AI Trainer

This repository hosts a comprehensive Gradio-based application that allows you to:
- Chat with an AI model (supporting both Simple and "Teaching" mode)
- Perform web searches to inject external knowledge into your prompts
- Generate multi-step Chain-of-Thought (CoT) reasoning
- Collect and store user feedback (including step-level ratings for CoT)
- Conduct advanced training modes (RLHF, Supervised Fine Tuning, Teacher Fine Tuning)
- Manage RAG (Retrieval-Augmented Generation) instructions
- Maintain a feedback history and a training dashboard



## Features

1. **Model Chat**  
   - **Simple Conversation** mode provides concise answers.  
   - **Teaching** mode uses multi-step reasoning, optionally including parallel CoTs and web search.

2. **Parallel or Sequential CoTs**  
   - If system memory/VRAM allows, generate two Chains-of-Thought in parallel (one from a teacher model if configured).

3. **Feedback Storage**  
   - All feedback and CoT step-level ratings can be saved for later analysis or fine-tuning.

4. **Training Monitor**  
   - Supports advanced fine-tuning modes: RLHF, supervised fine-tuning, or teacher-based fine-tuning.

5. **Prompt Engineering**  
   - Combine arbitrary prompts with RAG instructions, then run them through any loaded model.

6. **RAG Management**  
   - Easily create, load, and update RAG instructions stored in `rag_data.json`.

7. **Feedback History**  
   - View, edit, and delete feedback entries stored in `feedback_dataset.json`.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/stacker00/localtrainer.git
   cd localtrainer
   ```
   
   

2. **Install Dependencies**
    - Create or activate a virtual environment (recommended), then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```  

## Usage
1. **Run the Application**
   ```bash
   python main.py
   ```
   - This will launch a Gradio interface at a local URL (e.g., http://127.0.0.1:7860).

2. **Tabs Overview**
- Chat with Model: Enter text, select your mode (Simple Conversation / Teaching), optionally include a web search, choose a model, and click Submit.
- CoT Splitting & Ratings: Fetch the generated CoTs from the chat tab to split and rate them step-by-step, then save.
- Training Monitor: Perform advanced training steps (RLHF, Supervised, Teacher).
- Prompt Engineering: Combine custom prompts with optional RAG instructions, then generate outputs.
- RAG Instructions: Manage your RAG instructions stored in rag_data.json.
- Dashboard: Quickly view memory usage and feedback statistics.
- Feedback History: Edit or delete prior feedback rows stored in feedback_dataset.json.

## Testing

```bash
   pytest test_main.py
   ```
or
```bash
   python -m unittest test_main.py
   ```

## Issues
- Web search sometimes works when duckduckgo search page is opened on a browser. good luck.
- Metamask not working yet as of first commit.
- It is full of other bugs since I just wrote it. Hope to improve them soon.
