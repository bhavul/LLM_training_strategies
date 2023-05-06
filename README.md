# Large Language Models Training Strategies

Over the last few months as I've been reading up on LLMs, I have realised that deep within the actual papers lie some great advice on training strategies, tweaks made to data, choice of hyperparamters and such which get lost in the noise of hype. 

This repository will serve as an index detailing various strategies, tweaks, and recent architectural or hyperparameter choices made by different large language models. 

# Strategies, tweaks, and techniques

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

## Pre-training Tasks

- Token Masking: Token deletion or masking is the most common strategy. In autoregressive models like GPT, it masks next token, while in encoder-decoder models like BERT it randomly masks some percentage of tokens (MLM).
- Next Sentence Prediction (NSP) : the model is given two sentences and must predict whether the second sentence follows the first in the original document
- Text infilling : parts of the text are removed, and the model is tasked with filling in the missing parts
- Sentence permutation : another form of sequence corruption where the sentences in a document are permuted, and the model is tasked with sorting them back into the original order
- Document Rotation :the document is rotated, and the model's task is to understand the correct ordering of the document

## Hyperparameter Choices

- TBA

## Optimization Strategies

- AdamW Optimizer: A variant of the Adam optimizer that decouples weight decay from the rest of the optimization algorithm.
- Gradient Clipping: A method to prevent gradients from becoming too large and causing numerical instability during training.

## Parallelism Strategies 

- Tensor Parallelism: Splitting the model's parameters across multiple GPUs to allow for larger models.
- Data Parallelism (e.g. PyTorch FSDP and DDP, ZERO sharding strategy): Splitting the input data across multiple GPUs to allow for larger batch sizes.
- Pipeline Parallelism: Splitting the model into several stages that are each run on different GPUs, allowing for larger models and better hardware utilization.

## Fine-tuning Approaches

- P-tuning: A method of parameter efficient fine-tuning where additional positional parameters are introduced and learned.
- Adapter Tuning: A method of fine-tuning where additional, smaller layers (adapters) are added to the model and trained, while the original model parameters are frozen.


## Contributions

Contributions are welcome. 