# Support Ticket Classification

This project demonstrates a support ticket classification workflow using a local dataset and a large language model (LLaMA 2) to categorize customer issues, infer urgency, and structure outputs for downstream processing.

The notebook in this repository is designed as a practical example of how an LLM can be used to:

- classify support tickets into categories such as technical issues, hardware issues, and data recovery
- generate structured JSON outputs
- assign a priority and estimated time to resolution (ETA)
- help triage incoming support requests more consistently

## Project Structure

- `support_ticket.ipynb` — notebook containing the full data loading, model setup, prompt engineering, categorization, and analysis workflow
- `support_ticket_data.csv` — sample support ticket dataset used in the project
- `LICENSE` — repository license

## Dataset

The dataset contains a small set of support tickets with the following columns:

- `support_tick_id` — unique identifier for each ticket
- `support_ticket_text` — text description of the support issue

Example records include issues such as:

- slow or unstable internet connections
- laptop startup failures
- accidental file deletion / data recovery requests
- weak Wi-Fi signal complaints
- battery drain problems
- account access / password reset requests
- slow computer performance

## Notebook Workflow

The notebook walks through a typical LLM-based classification pipeline:

1. Install required dependencies
   - `llama-cpp-python`
   - `huggingface_hub`
2. Load the support ticket dataset from `support_ticket_data.csv`
3. Download the LLaMA model from the Hugging Face model hub
4. Initialize the model using `llama_cpp.Llama`
5. Define a prompt template that instructs the model to classify the ticket
6. Run the model against each ticket text
7. Extract and normalize labels from the model output
8. Build a final categorized dataset with fields such as category, priority, and ETA

## Model and Prompting Approach

The workflow uses a local GGUF model from the LLaMA 2 family, specifically:

- `TheBloke/Llama-2-13B-chat-GGUF`
- `llama-2-13b-chat.Q5_K_M.gguf`

The notebook prompts the model with ticket text and instructions such as:

- classify the ticket into one of the supported categories
- return a structured JSON object
- produce a response suitable for support triage workflows

This is an example of prompt-based classification and is useful for experimentation, prototyping, and learning rather than as a fully production-hardened deployment.

## Requirements

To run the notebook successfully, you should have:

- Python 3.9+
- Jupyter Notebook or JupyterLab
- internet access to download the model and dependencies
- a machine with enough RAM/VRAM for running a LLaMA model
- optional GPU support for better performance

## Setup

From the project root, install the required packages:

```bash
pip install pandas huggingface_hub llama-cpp-python
```

Then open the notebook and run the cells in order.

If you are using a GPU-enabled environment, the notebook includes a CUDA installation command for `llama-cpp-python`:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir -q
```

## Example Usage

Once the notebook is running, it can process ticket text like this:

> "My computer is running very slowly and I need help optimizing performance."

The model is expected to classify it into a relevant support category and return a structured result that can be used in triage or automation workflows.

## Notes and Limitations

- The dataset is intentionally small and is best suited for demonstration purposes.
- The model output can vary depending on prompt wording, model version, temperature, and runtime environment.
- Output parsing can require careful validation in production systems.
- The notebook is a prototype, not a full production support orchestration system.
- Some notebook cells reference Google Colab-specific paths and may need adjustment for a local environment.

## Intended Use

This repository is useful for:

- learning prompt-based classification with LLMs
- experimenting with ticket triage patterns
- prototyping text classification pipelines for customer support
- understanding how structured LLM outputs can be integrated into business workflows

## License

This project is licensed under the terms of the included [LICENSE](LICENSE).
