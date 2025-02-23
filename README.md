![Weight & Biases Visualization](Screenshot_6.png)

This repository contains code and resources for fine-tuning the Deepseek AI model (specifically `deepseek-ai/deepseek-coder-6.7b-instruct`) on a medical dataset (`FreedomIntelligence/medical-o1-reasoning-SFT`) to enhance its performance in medical reasoning tasks. The project uses advanced techniques like 4-bit quantization, LoRA adapters, and libraries such as Transformers, Datasets, and PEFT for efficient training on GPU hardware.

## Overview

The goal of this project is to adapt the Deepseek language model to provide accurate and structured responses to medical queries by fine-tuning it on a specialized medical dataset. The fine-tuning process leverages efficient techniques to minimize memory usage while maintaining performance, making it suitable for resource-constrained environments like Google Colab with a T4 GPU.

## Features

- Fine-tunes the Deepseek model using 4-bit quantization for memory efficiency.
- Implements LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
- Uses a medical dataset to train the model for medical question-answering tasks.
- Supports training on GPU with FP16 precision for faster computation.
- Includes preprocessing, tokenization, and training pipelines using Hugging Face libraries.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.11 or later
- PyTorch (with CUDA support, version 2.5.1 or later)
- Hugging Face Transformers (`transformers>=4.48.3`)
- Hugging Face Datasets (`datasets>=3.3.2`)
- PEFT (`peft>=0.14.0`)
- BitsAndBytes (`bitsandbytes>=0.45.2`)
- Accelerate (`accelerate>=1.3.0`)
- Hugging Face Hub (`huggingface_hub>=0.28.1`)

### Installation

Install the required packages using pip:

```bash
pip install transformers datasets torch accelerate huggingface_hub peft bitsandbytes
```

Ensure you have a Hugging Face token for accessing the Deepseek model and dataset. Log in using:

```python
from huggingface_hub import login
login(token="your-huggingface-token")
```

Replace `"your-huggingface-token"` with your actual Hugging Face access token.

## Dataset

This project uses the `FreedomIntelligence/medical-o1-reasoning-SFT` dataset, which contains medical questions and responses in English. The dataset is subsetted to 4,000 training samples and 400 validation samples for efficient fine-tuning.

## Usage

### 1. Clone the Repository

Clone this repository to your local machine or set up a Colab notebook with the provided code.

### 2. Prepare the Environment

Run the following command in your notebook or terminal to install dependencies:

```bash
pip install transformers datasets torch accelerate huggingface_hub peft bitsandbytes
```

### 3. Fine-Tune the Model

The main script (`Finetuning_Deepseek_R1.ipynb`) contains the fine-tuning pipeline. Follow these steps:

1. Load the dataset and subset it as needed:
   ```python
   dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")
   train_dataset = dataset["train"].shuffle(seed=42).select(range(4000))
   eval_dataset = dataset["test"].shuffle(seed=42).select(range(400))
   ```

2. Preprocess and tokenize the data using the Deepseek tokenizer:
   ```python
   tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
   ```

3. Load the base model with 4-bit quantization and add LoRA adapters:
   ```python
   from transformers import BitsAndBytesConfig
   from peft import LoraConfig, get_peft_model

   quant_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.float16,
       bnb_4bit_use_double_quant=True,
   )
   model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", quantization_config=quant_config)
   lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
   model = get_peft_model(model, lora_config)
   ```

4. Set up training arguments and train the model:
   ```python
   from transformers import TrainingArguments, Trainer

   training_args = TrainingArguments(
       output_dir="./fine_tuned_model",
       num_train_epochs=1,
       per_device_train_batch_size=2,
       per_device_eval_batch_size=2,
       gradient_accumulation_steps=8,
       save_strategy="steps",
       save_steps=500,
       learning_rate=3e-5,
       fp16=True,
       optim="paged_adamw_8bit",
   )

   trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train, eval_dataset=None)
   trainer.train()
   ```

5. Save the fine-tuned model and tokenizer:
   ```python
   model.save_pretrained("fine-tuned-deepseek-r1-1.5b")
   tokenizer.save_pretrained("fine-tuned-deepseek-r1-1.5b")
   ```

### 4. Inference

Use the fine-tuned model to generate responses to medical queries. Example:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

model_path = "fine-tuned-deepseek-r1-1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", device_map="auto")
model = PeftModel.from_pretrained(model, model_path)

prompt = "What are the symptoms and treatment options for Type 2 Diabetes?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Hardware Requirements

- GPU with at least 16GB of VRAM (e.g., NVIDIA T4 or better) for efficient training with 4-bit quantization.
- 32GB of system RAM recommended for handling large models and datasets.

## Results

After fine-tuning, the model achieves a training loss reduction (e.g., from 2.8102 to 1.3631 over 250 steps), indicating improved performance on the medical dataset. The fine-tuned model can generate structured, step-by-step medical responses.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Ensure you follow the project's coding standards and include tests for new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Deepseek AI for providing the base model (`deepseek-ai/deepseek-coder-6.7b-instruct`).
- Hugging Face for the Transformers, Datasets, and PEFT libraries.
- FreedomIntelligence for the `medical-o1-reasoning-SFT` dataset.
