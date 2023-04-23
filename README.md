# Large Language Models Training Strategies

Over the last few months as I've been reading up on LLMs, I have realised that deep within the actual papers lie some great advice on training strategies, tweaks made to data, choice of hyperparamters and such which get lost in the noise of hype. 

This repository will serve as an index detailing various strategies, tweaks, and recent architectural or hyperparameter choices made by different large language models. 


## Training Strategies

- Unpadding: A technique to reduce the computational cost of training by removing unnecessary padding tokens.
- Gradient Clipping: A method to prevent gradients from becoming too large and causing numerical instability during training.
- AdamW Optimizer: A variant of the Adam optimizer that decouples weight decay from the rest of the optimization algorithm.
- Tensor Parallelism: Splitting the model's parameters across multiple GPUs to allow for larger models.
- Data Parallelism (e.g. PyTorch FSDP and DDP, ZERO sharding strategy): Splitting the input data across multiple GPUs to allow for larger batch sizes.
- LayerNorm: A type of normalization technique often used in transformer models to stabilize the inputs to each layer.


### Contributions

Contributions are welcome. 