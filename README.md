# Large Language Models Training Strategies

Over the last few months as I've been reading up on LLMs, I have realised that deep within the actual papers lie some great advice on training strategies, tweaks made to data, choice of hyperparamters and such which get lost in the noise of hype. 

This repository will serve as an index detailing various strategies, tweaks, and recent architectural or hyperparameter choices made by different large language models. 


## Data Preparation Strategies

- MinHashLSH with Jaccard Similarity for De-duplicating Dataset
- Dataset Distillation: A process where a smaller 'student' dataset is created that can be used to train a model to similar performance as the original 'teacher' dataset.
- Dynamic Mixing: Mixing data from different datasets or domains at different ratios during training.

## Model Architectural Choices

- Transformer Architecture: The fundamental building block of most modern LLMs, consisting of self-attention and feed-forward layers.
- Sparse and Dense Attention Layers in Alternation: A combination of sparse attention (for computational efficiency) and dense attention (for representational power).
- Flash Attention Layer: A method to increase the efficiency of attention mechanisms.
- Rotary Embeddings for Positional Embeddings: A new technique that injects positional information more effectively into transformer models.
- LayerNorm: A type of normalization technique often used in transformer models to stabilize the inputs to each layer.

## Training Strategies

- Unpadding: A technique to reduce the computational cost of training by removing unnecessary padding tokens.
- Knowledge Distillation: Training a smaller student model to imitate the behavior of a larger teacher model.

## Optimization Strategies

- AdamW Optimizer: A variant of the Adam optimizer that decouples weight decay from the rest of the optimization algorithm.
- Gradient Clipping: A method to prevent gradients from becoming too large and causing numerical instability during training.


## Parallelism Strategies 
- Tensor Parallelism: Splitting the model's parameters across multiple GPUs to allow for larger models.
- Data Parallelism (e.g. PyTorch FSDP and DDP, ZERO sharding strategy): Splitting the input data across multiple GPUs to allow for larger batch sizes.
- Pipeline Parallelism: Splitting the model into several stages that are each run on different GPUs, allowing for larger models and better hardware utilization.


### Contributions

Contributions are welcome. 