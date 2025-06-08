# Fine-tuning Mistral 7B with QLoRA for iPad Mini 6 Deployment

This project provides scripts and instructions to fine-tune the Mistral 7B model using QLoRA, quantize it to GGUF format, and prepare it for deployment on an iPad Mini 6 (or other devices compatible with llama.cpp).

The process follows these main steps:

1.  **Environment Setup**: Prepare your Python environment with necessary libraries.
2.  **PDF to Q&A Dataset**: Extract text from a PDF and convert it into Question & Answer pairs.
3.  **Prepare Dataset for Training**: Convert the Q&A pairs into a Hugging Face `datasets` object.
4.  **Fine-tune Mistral 7B with QLoRA**: Run the fine-tuning script.
5.  **Merge LoRA Weights (Optional)**: Merge the trained LoRA adapters into the base model.
6.  **Quantize and Deploy**: Convert the model to GGUF format and quantize it for use with `llama.cpp`.

## 1. Environment Setup

This section provides guidance for setting up your environment on Ubuntu and Windows.
**A CUDA-compatible GPU with at least 16GB VRAM is highly recommended for running the fine-tuning script.**

### General Setup (Applicable to both Ubuntu and Windows)

1.  **Install Miniconda/Anaconda**:
    *   Download and install Miniconda (recommended for a minimal installation) or Anaconda from [https://docs.conda.io/projects/miniconda/en/latest/](https://docs.conda.io/projects/miniconda/en/latest/) or [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution). Follow the instructions for your operating system.

2.  **Create a Conda Environment**:
    Open a terminal (or Anaconda Prompt on Windows) and run:
    ```bash
    conda create -n mistral-qlora python=3.10 -y
    conda activate mistral-qlora
    ```

### Ubuntu Setup Details

1.  **NVIDIA CUDA Toolkit**:
    *   Ensure you have NVIDIA drivers installed.
    *   Install the NVIDIA CUDA Toolkit that matches the PyTorch CUDA version you intend to use (e.g., CUDA 11.8 or 12.1). It's often best to install this system-wide via NVIDIA's official repositories or installers.
        *   Check PyTorch installation instructions for compatible CUDA versions: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
        *   NVIDIA CUDA Toolkit Archive: [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)
    *   Verify `nvcc --version` in your terminal.

2.  **Install PyTorch**:
    *   Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select your preferences (e.g., Stable, Linux, Pip, Python, desired CUDA version).
    *   Install PyTorch using the generated command. For example, for CUDA 12.1:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

3.  **Install Other Dependencies**:
    *   Install `build-essential` for compiling some packages if not already present:
        ```bash
        sudo apt-get update
        sudo apt-get install build-essential
        ```
    *   Install the remaining packages from `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```
        This will also install `google-generativeai` for Gemini API access and `python-dotenv` for managing API keys. Note: `bitsandbytes` should compile smoothly on most Linux distributions with the necessary build tools.

### Windows Setup Details

1.  **NVIDIA CUDA Toolkit**:
    *   Ensure you have NVIDIA drivers installed (Game Ready or Studio drivers).
    *   Install the NVIDIA CUDA Toolkit. Download it from the NVIDIA website: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads). Choose the version compatible with the PyTorch version you plan to install.
    *   During installation, it's recommended to use the "Express" setting or ensure that `nvcc` is added to your system's PATH.
    *   Verify `nvcc --version` in Command Prompt or PowerShell.

2.  **Microsoft Visual Studio (Build Tools)**:
    *   Some Python packages, including `bitsandbytes`, may require C++ build tools for compilation on Windows.
    *   Install "Build Tools for Visual Studio" from [https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/) (under "Tools for Visual Studio").
    *   During installation, select "Desktop development with C++".

3.  **Install PyTorch**:
    *   Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select your preferences (e.g., Stable, Windows, Pip, Python, desired CUDA version).
    *   Install PyTorch using the generated command from an Anaconda Prompt or PowerShell (within your conda environment). For example, for CUDA 12.1:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

4.  **Install `bitsandbytes` (Windows Specifics)**:
    *   `bitsandbytes` has historically had some challenges with Windows. As of recent versions, pre-compiled Windows wheels might be available, or it might compile correctly if CUDA and Visual Studio Build Tools are set up properly.
    *   Try installing directly:
        ```bash
        pip install bitsandbytes
        ```
    *   If you encounter issues, refer to the official `bitsandbytes` repository for Windows-specific instructions or precompiled binaries: [https://github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) (check Issues and Releases). Sometimes specific versions are recommended, e.g., `pip install bitsandbytes==0.41.1` (check for the latest compatible version).

5.  **Install Other Dependencies**:
    *   The `requirements.txt` file also includes `google-generativeai` for Gemini API access and `python-dotenv` for managing API keys.
    *   Install the remaining packages from `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```

### Verifying the Setup

After installation, you can verify PyTorch and CUDA:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
```
Save this as a Python script (e.g., `check_env.py`) and run `python check_env.py`.

This updated section provides more detailed, OS-specific instructions for setting up the Conda environment, installing CUDA, PyTorch, and other dependencies, including notes on potential issues like `bitsandbytes` on Windows.

### Google API Key for Q&A Generation

The `pdf_to_qa.py` script now uses the Google Gemini API to generate question-answer pairs. To use this feature, you need a Google API Key with access to the Gemini API.

1.  **Obtain an API Key**: Visit [https://ai.google.dev/](https://ai.google.dev/) to get your API key.
2.  **Set up `.env` file**:
    *   In the root of this project, you'll find a file named `.env.example`.
    *   Rename or copy this file to `.env`.
    *   Open `.env` and replace `"YOUR_GEMINI_API_KEY"` with your actual Google API Key.
    ```
    # .env file content
    GOOGLE_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"
    ```
The script will load this key automatically. Keep your `.env` file secure and do not commit it to public repositories.

## 2. PDF to Q&A Dataset

The `pdf_to_qa.py` script extracts text from your PDF and uses the Google Gemini API to automatically generate **both detailed factual question-answer pairs and a few summarization-style Q&A pairs** for each text chunk.

**Prerequisites**: Ensure you have set up your `GOOGLE_API_KEY` in a `.env` file as described in the "Environment Setup" section.

To run the script (defaulting to approximately 10 Q&A pairs per chunk and a max chunk size of 12000 characters):
```bash
python pdf_to_qa.py --pdf_path your_doc.pdf --output_json qa_dataset.json
```
The script now attempts to extract Q&A pairs exhaustively from each text chunk based on the content. The `--num_questions_per_chunk` (default: 10) argument serves as a user guideline for the desired output quantity per chunk. You can also adjust the maximum character size for each chunk using `--max_chunk_chars` (default: 12000). Smaller chunk sizes might yield more focused Q&A for very dense documents.
Example with custom settings:
```bash
python pdf_to_qa.py --pdf_path your_doc.pdf --output_json qa_dataset.json --num_questions_per_chunk 15 --max_chunk_chars 10000
```
While the Gemini API provides a strong starting point, it's highly recommended to review and curate the generated Q&A pairs for quality and relevance. The script includes an enhanced retry mechanism with increasing backoff times to handle transient API issues, attempting up to 9 times before failing on a specific text chunk.
The script uses a very detailed internal prompt to instruct Gemini to be as exhaustive and literal as possible in extracting factual Q&A pairs from each text chunk. While this aims for maximum detail, you can further refine this internal prompt (located in the `generate_qa_pairs` function within `pdf_to_qa.py`) if you have very specific Q&A style requirements or observe particular patterns in Gemini's output for your documents. Experimenting with `--num_questions_per_chunk` and `--max_chunk_chars` can also help optimize the results for your needs.
Generating both types of Q&A pairs (factual and summarization) aims to create a richer dataset. This helps the fine-tuned model to not only recall specific details but also to provide summaries when explicitly prompted (e.g., 'Summarize the key points of section X').
**Crucially, the quality, accuracy, and relevance of your Q&A pairs will significantly impact the fine-tuned model's performance.** Refer to the comments within `pdf_to_qa.py` for more detailed advice on dataset creation.

## 3. Prepare Dataset for Training

Use the `prepare_dataset.py` script to convert the `qa_dataset.json` into the format required for training and save it to disk.

```bash
python prepare_dataset.py --input_json qa_dataset.json --output_dir mistral_qa_dataset
```

**Note for CVs/Specific Document Fine-tuning**: If your goal is for the model to learn the entire content of a specific document (like a CV), it's recommended to train on all generated Q&A pairs. You can achieve this by setting `--test_size 0` when running the script:
```bash
python prepare_dataset.py --input_json qa_dataset.json --output_dir mistral_qa_dataset --test_size 0
```
This ensures all extracted information is used for training, maximizing the model's knowledge of that specific document.
This will create a directory named `mistral_qa_dataset` containing the processed dataset.

## 4. Fine-tune Mistral 7B with QLoRA

Use the `train_mistral_qlora.py` script to fine-tune the model.

```bash
python train_mistral_qlora.py \
    --model_id "mistralai/Mistral-7B-v0.1" \
    --dataset_path "mistral_qa_dataset" \
    --output_dir "mistral-qlora-output" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --fp16 \
    --logging_steps 10
```
The training script defaults to evaluating the model on the test set each epoch (`--evaluation_strategy "epoch"`). **It is highly recommended to experiment with hyperparameters** like learning rate (`--learning_rate`), LoRA r (`--lora_r`) and alpha (`--lora_alpha`), number of epochs (`--num_train_epochs`), and maximum sequence length (`--max_length`) to achieve optimal results for your specific dataset and task. Training checkpoints will be saved in `mistral-qlora-output`.

## 5. Merge LoRA Weights (Optional)

If you want to merge the LoRA adapter weights with the base model to create a single model directory, use the `merge_adapters.py` script. Replace `checkpoint-xxx` with the actual checkpoint you want to use from the `mistral-qlora-output` directory (e.g., `checkpoint-100` if you trained for 3 epochs with 100 steps per epoch, it might be the last one).

```bash
python merge_adapters.py \
    --base_model_id "mistralai/Mistral-7B-v0.1" \
    --adapter_path "mistral-qlora-output/checkpoint-xxx" \
    --output_dir "merged-mistral-qlora"
```

This will save the merged model and tokenizer to the `merged-mistral-qlora` directory.

## 6. Quantize and Deploy to iPad Mini 6 (via llama.cpp)

### A. Convert to GGUF for llama.cpp

First, clone the `llama.cpp` repository:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

Next, convert your fine-tuned (and optionally merged) model to GGUF format. If you merged the adapters, `model_dir` will be `merged-mistral-qlora`. If you did not merge and want to convert a specific checkpoint that PEFT can load, you might need to adjust the `convert.py` script or ensure it can load adapter weights (often, conversion scripts expect a fully merged model). The user's instructions imply using the merged model.

Make sure your `merged-mistral-qlora` directory (or the checkpoint directory if not merging and `convert.py` supports it) is accessible. The original instructions used `./merged-mistral-qlora` relative to the `llama.cpp` directory. You might need to adjust paths. For example, if `llama.cpp` is in the same parent directory as your fine-tuning project:

```bash
# Inside llama.cpp directory
python3 convert.py ../merged-mistral-qlora --outfile ../mistral-7b-merged-f16.gguf --outtype f16
```
*(Note: The original instructions had `--model_dir ./merged-mistral-qlora` and `--outfile mistral-7b-gguf`. The `convert.py` script arguments can vary; consult `python3 convert.py --help`. The `--outtype f16` is common for an intermediate float16 GGUF model before quantization.)*

### B. Quantize the GGUF Model

Now, quantize the float16 GGUF model to a smaller format like Q4_K_M. Compile `llama.cpp` if you haven't already (e.g., `make`).

```bash
# Inside llama.cpp directory
./quantize ../mistral-7b-merged-f16.gguf ../mistral-7b-merged.Q4_K_M.gguf Q4_K_M
```
This creates `mistral-7b-merged.Q4_K_M.gguf`, which is the model file you'll use on the iPad.

### C. Use with llama.cpp on iPad

1.  Transfer the `mistral-7b-merged.Q4_K_M.gguf` file to your iPad Mini 6.
2.  Use an application that supports `llama.cpp` models in GGUF format. Examples include:
    *   **MLC Chat**: Developed by the MLC community, often supports various GGUF models.
    *   **LocalAI mobile client**: If available and supports GGUF.
    *   **Custom App**: Compile `llama.cpp` into a custom iOS application. There are Swift wrappers and community projects like `ios-ggml-clients` that can serve as starting points.

Place the `.gguf` model file in the app's accessible storage directory as required by the specific app.

> **Tip**: The Q4_K_M quantization is a good balance for reducing memory usage and maintaining performance for real-time responses on mobile devices.

## Project Structure

```
.
├── .env.example             # Example for API key configuration
├── README.md
├── requirements.txt
├── pdf_to_qa.py
├── prepare_dataset.py
├── train_mistral_qlora.py
├── merge_adapters.py
├── your_doc.pdf             # (You provide this)
├── qa_dataset.json          # (Generated by pdf_to_qa.py)
├── mistral_qa_dataset/      # (Generated by prepare_dataset.py)
├── mistral-qlora-output/    # (Generated by train_mistral_qlora.py)
└── merged-mistral-qlora/    # (Generated by merge_adapters.py)
```
