# Training Details - Fluid Antenna System Paper

## Training objective

Unsupervised. No optimal-beamforming labels are provided. The loss is the negative average weighted sum rate (WSR), so minimizing the loss is equivalent to maximizing WSR. Parameter updates follow this loss via stochastic gradient descent.

## Optimizer

The methodology section uses generic SGD language, but Section VI specifies Adam as the concrete optimizer. This is not a contradiction: Adam is a stochastic gradient-based method. Configuration:

- Initial learning rate: 0.001
- LR decay factor: 0.995 every 100 training steps
- Samples per epoch: 10,000 (synthetic, randomly generated)
- Batch size: 200 -> 50 updates per epoch
- Test set: 2,000 samples
- Stopping: predefined epoch limit or convergence, whichever comes first

## Input and output representation

Complex CSI is converted to a real-valued matrix by concatenating real and imaginary parts. For each cell, the node-feature submatrix stacks the CSI from that BS to its assigned users. The output follows the same format, representing [Re(w), Im(w)] of the beamforming vectors.

## Network architecture (per-cell GNN)

Each per-cell GNN contains:

1. Input MLP: expands from 2N -> 1024 -> 512
2. Two successive GNN layers: MLP blocks centered at 512-dimensional embeddings, with concatenation-induced expansion to 1024
3. Final FC layer: projects back to 2N

ReLU is used in all FC layers except the final output layer, which is linear.

## Global training vs distributed deployment

Global training does not mean federated learning, parameter averaging, or distributed backpropagation. It means the full multi-GNN model is trained in one centralized setup with all cells' CSI available jointly, allowing the model to learn inter-cell interference coupling. After training, the model is split into independent per-BS GNN instances for distributed deployment.
