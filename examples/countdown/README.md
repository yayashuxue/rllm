# CountDown Agent Examples

This directory contains examples for training and running countdown task agents using the RLLM framework. The countdown agent solves mathematical puzzles where it must use given numbers with basic arithmetic operations to reach a target number.

Our examples use the following:
* Qwen3-0.6B as the base model
* Countdown-Tasks-3to4 dataset for training and evaluation
* 1024 examples as test set, remaining as training set

## Task Description

The countdown task involves using a set of given numbers (3-4 numbers) to reach a target number through basic arithmetic operations (+, -, *, /). Each number can only be used once. For example:

- **Target**: 98
- **Numbers**: [44, 19, 35]
- **Solution**: 44 + 19 + 35 = 98

## Model Hosting

### Option 1: Using vLLM

Start a vLLM server with OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 
```

### Option 2: Using SGLang

```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-0.6B \ 
    --dp-size 1 \
    --dtype bfloat16
# increase dp_size to enable data-parallel processing on multi-GPU 
```

The server should be accessible at `http://localhost:30000/v1`

## Dataset Preparation

Prepare the countdown task dataset from HuggingFace:

```bash
cd examples/countdown
python prepare_countdown_data.py
```

This will:
- Download Countdown-Tasks-3to4 dataset from HuggingFace
- Split into 1024 test examples and remaining training examples
- Register both datasets with the RLLM DatasetRegistry
- Convert the raw data into question-answer format suitable for training

## Running Inference

Once your model server is running and datasets are prepared, you can run inference:

```bash
cd examples/countdown
python run_countdown.py
```

### Configuration Options

You can modify the inference script parameters:

- `n_parallel_agents`: Number of parallel agents (default: 64)
- `model_name`: Model to use (default: "Qwen/Qwen3-0.6B")
- `base_url`: API server URL (default: "http://localhost:30000/v1")
- `max_response_length`: Maximum response length (default: 8192)
- `max_prompt_length`: Maximum prompt length (default: 2048)
- `temperature`: Sampling temperature (default: 0.6)
- `top_p`: Top-p sampling (default: 0.95)

The script will:
1. Load the countdown test dataset (1024 examples)
2. Repeat each problem 16 times for Pass@K evaluation
3. Run parallel and async trajectory collection using the agent execution engine
4. Evaluate results and report Pass@1 and Pass@K accuracy

## Training

### Basic Training

To train the countdown agent with 8K context:

```bash
bash examples/countdown/train_countdown_8k.sh
```

### Training Configuration

The training script uses the following key configurations:
- **Algorithm**: GRPO (Generalized Reward-based Policy Optimization)
- **Batch Size**: 128 (train), 30 (validation)
- **Context Length**: 2048 (prompt), 8192 (response)
- **Learning Rate**: 1e-6
- **Total Epochs**: 100
- **Model**: Qwen3-0.6B

### Extending Context Length

To train with longer context lengths, you can create additional training scripts following the pattern:
- `train_countdown_16k.sh` - for 16K context
- `train_countdown_24k.sh` - for 24K context

## Dataset Statistics

- **Total Examples**: ~490K
- **Training Set**: ~489K examples (after removing 1024 for test)
- **Test Set**: 1024 examples
- **Number Range**: Target numbers from 10-100
- **Input Numbers**: 3-4 numbers per problem
- **Task Type**: Mathematical reasoning with arithmetic operations

## Performance Evaluation

The evaluation uses Pass@K metrics:
- **Pass@1**: Percentage of problems solved correctly on first attempt
- **Pass@16**: Percentage of problems solved correctly in at least one of 16 attempts

## Example Usage

```bash
# 1. Prepare data
cd examples/countdown
python prepare_countdown_data.py

# 2. Start model server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 30000

# 3. Run inference
python run_countdown.py

# 4. Train model (optional)
bash train_countdown_8k.sh
```

## Notes

- The countdown agent inherits from the base agent architecture and can be extended for more complex mathematical reasoning tasks
- The reward function uses the existing math reward system to evaluate correctness
- The task format converts countdown problems into natural language instructions for better model understanding 