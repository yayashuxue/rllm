# Solver-Judge Workflow for Countdown Tasks

This directory contains an implementation of a solver-judge workflow for countdown tasks, following the pattern established in the critique agent examples.

## Overview

The solver-judge workflow consists of two phases:
1. **Solver Phase**: A solver agent generates multiple solutions (default: 4) to the countdown problem
2. **Judge Phase**: A judge agent evaluates all solutions and selects the best one

This approach can potentially improve performance by:
- Generating diverse solution attempts
- Using verification to select the highest quality solution
- Training both generation and evaluation capabilities

## Files

- `train_solver_judge_flow.py`: Training script for the solver-judge workflow
- `train_solver_judge_flow.sh`: Bash script with training configuration
- `solver_judge_flow.py`: Solver-judge workflow implementation

## Training Configuration

The training uses the following key settings:
- **Model**: Qwen/Qwen3-0.6B
- **Dataset**: Countdown task dataset 
- **Batch Size**: 8
- **Context Length**: 4096 tokens (prompt + response)
- **Solutions per Problem**: 4
- **GPUs**: 8 (configurable via CUDA_VISIBLE_DEVICES)

## Usage

### Prerequisites

1. Ensure the countdown dataset is prepared:
```bash
cd examples/countdown
python prepare_countdown_data.py
```

2. Verify the environment is set up correctly with required dependencies.

### Training

Run the training script:
```bash
cd examples/solver_judge
bash train_solver_judge_flow.sh
```

The script will:
- Set up the required environment variables for vLLM
- Launch distributed training using Ray
- Log progress to console and Weights & Biases
- Save checkpoints every 10 epochs
- Run validation every 10 epochs

### Configuration

Key parameters can be modified in the bash script:
- `data.train_batch_size`: Batch size for training
- `data.max_prompt_length`: Maximum prompt length
- `data.max_response_length`: Maximum response length  
- `actor_rollout_ref.rollout.n`: Number of solutions per problem
- `trainer.total_epochs`: Total training epochs
- `trainer.project_name`: W&B project name

## Workflow Architecture

The `SolverJudgeWorkflow` implements a two-phase approach:

1. **Initialization**: Creates solver and judge components with the rollout engine
2. **Execution**: 
   - Solver generates N solutions to the problem
   - Judge evaluates all solutions and selects the best one
   - Reward function is applied to evaluate the selected solution
3. **Training**: The workflow creates episodes with trajectories from both solver and judge

## Expected Benefits

This workflow architecture may provide improvements over single-agent approaches:
- **Diversity**: Multiple solution attempts increase chance of finding correct answers
- **Quality Control**: Judge can filter out poor solutions
- **Robustness**: Verification step reduces impact of occasional poor generations
- **Learning**: Both generation and evaluation skills are trained simultaneously

## Monitoring

Training progress can be monitored via:
- Console logs showing loss, rewards, and validation metrics
- Weights & Biases dashboard (project: 'solver-judge-workflow')
- Checkpoint files saved every 10 epochs

## Troubleshooting

Common issues and solutions:
- **Memory errors**: Reduce batch size or context length
- **Ray errors**: Check CUDA_VISIBLE_DEVICES and available GPUs
- **Dataset errors**: Ensure countdown dataset is properly prepared
- **Import errors**: Verify all dependencies are installed and paths are correct 