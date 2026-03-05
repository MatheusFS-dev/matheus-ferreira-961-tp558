# Related work section

The author organizes related work into three main areas:

## A. Beamforming and Port Selection in FASs

- This part reviews studies focused specifically on beamforming and port selection.
- Most methods use alternating optimization (AO) to maximize energy efficiency by jointly optimizing beamforming and fluid antenna (FA) positioning.

### Identified gap

Although existing beamforming and port selection methods can solve optimization problems in their specific scenarios, they often have high computational complexity and do not adequately consider hardware execution efficiency. This motivates a hardware-software co-optimization approach to reduce computational latency.

## B. GNN-Based Beamforming Optimization

- This part reviews prior work that applies GNNs to beamforming in different wireless settings (for example, MIMO-, RIS-, and MISO-related scenarios).

### Identified gap

Even with strong results from GNN-based beamforming in several systems, its use for joint beamforming and port selection in FASs is still largely unexplored.

## C. FPGA-Based Deep Learning Acceleration

- This part reviews prior efforts to accelerate deep learning inference on FPGA.
- Two common architecture paradigms are highlighted:
	- **Fully pipelined:** processes layers in sequence to improve throughput and resource usage.
	- **Non-pipelined (universal compute unit):** reuses one compute structure across all DNN layers.

### Identified gap

Existing FPGA accelerators can support general DNN inference, but they do not fully exploit the specific characteristics of this GNN-based optimization algorithm. As a result, computational efficiency is limited, motivating a tailored hardware design.

## Overall synthesis

Current literature treats these themes mostly in isolation:

- beamforming and port selection,
- GNN-based optimization,
- FPGA acceleration.

What is still missing is a unified hardware-software co-optimization perspective. The core contribution is therefore an end-to-end system that integrates all three dimensions, instead of optimizing only one part in isolation.


# System model section

The author considers an FA-enhanced downlink multi-cell MIMO network.
Each base station (BS) serves multiple users simultaneously, and each user has a fixed-position antenna.

## 1) Network architecture and FA structure

- Each BS contains $N$ fluid antennas (FAs).
- Each FA has $L$ selectable ports, and the RF chain can switch among them.
- Ports within each FA are uniformly distributed along a linear span of length $W\lambda$.
- Different FAs are separated by at least $\lambda/2$, so inter-FA correlation and mutual coupling are negligible.

Conceptually:

BS (with N fluid antennas)
   ├── FA1: ports {1..L}
   ├── FA2: ports {1..L}
   ...
   └── FAN: ports {1..L}

This switchable-port structure increases the BS-UE spatial DoF compared with fixed-position antennas.

## 2) Channel assumptions

- Rich scattering is assumed (typical dense urban propagation).
- Small-scale fading is Rayleigh distributed.
- Because ports in the same FA are close, their channels are spatially correlated.
- Port-level spatial correlation inside each FA is modeled using Jakes' model.
- The overall channel is therefore correlated Rayleigh fading.
- Perfect CSI is assumed at all BSs.

## 3) Signal transmission model

Each BS transmits $K$ data streams (one per served user) using beamforming.

For each user, the received signal contains:

- Desired signal (its own stream).
- Intra-cell interference (from other users served by the same BS).
- Inter-cell interference (from neighboring BSs).
- Thermal noise.

## 4) Joint port-selection and beamforming idea

Because each FA has multiple candidate ports, selecting different ports creates different channel realizations.
By choosing suitable port combinations, the system can:

- improve desired signal strength,
- suppress intra/inter-cell interference,
- and improve beamforming effectiveness.

Therefore, port selection and beamforming are optimized jointly.

## 5) Optimization objective

The objective is to maximize network communication performance, measured by the sum rate across all users.
The optimization jointly determines:

- which FA ports are activated, and
- which beamforming vectors are used.

This joint design is the key mechanism for exploiting FA-enabled spatial flexibility.


# Beamforming and Port Selection

The system maximizes weighted sum rate under BS transmit-power constraints.
To do so, it jointly optimizes:

- beamforming vectors (continuous variables), and
- FA port selection (discrete variables).

Because this combines discrete and continuous decisions, the original formulation is mixed-integer and non-convex, which becomes intractable as antenna/port counts increase.

## 1) Core strategy: GNN-RPS

The paper decomposes the joint problem into two coordinated steps:

- **GNN-based beamforming** for a fixed port configuration.
- **Random Port Selection (RPS)** to explore candidate configurations.

Together, these steps form the GNN-RPS algorithm, which trades exhaustive search for efficient sampling plus fast neural inference.

## 2) Beamforming subproblem (fixed ports)

If ports are fixed, only beamforming remains.
Instead of iterative optimization each time CSI changes, a GNN is trained to map channel information directly to beamforming weights.

Why GNN:

- Wireless interference has graph-like dependencies.
- User representations should include information from interfering users.
- After training, inference is fast enough for real-time operation.

## 3) Port-selection subproblem (RPS)

Exhaustive port search is combinatorial.
RPS approximates the search by evaluating only a random subset of candidate port configurations.

Per candidate configuration:

1. Build the graph input from current CSI.
2. Run the GNN to predict beamforming vectors.
3. Compute communication performance.

Then select the candidate with the best achieved rate.

## 4) End-to-end algorithm flow

1. Randomly sample multiple port configurations.
2. For each configuration, infer beamforming with the GNN.
3. Evaluate objective value (network rate).
4. Keep the best port + beamforming pair.

This yields near-optimal performance with much lower complexity than joint exhaustive optimization.

## 5) Graph representation (input and output)

### 5.1 What the graph represents

Within each cell, users form a fully connected interaction graph.
This models the assumption that all users can potentially interfere with one another.

### 5.2 Node definition

- One node corresponds to one UE served by one BS.
- Node features are the BS-to-UE channel over $N$ FAs.
- Complex CSI is split into real and imaginary parts.
- Therefore each node feature vector has $2N$ real values.

### 5.3 Edge interpretation

- Edges represent user-user interaction potential (interference coupling).
- The implementation emphasis is on neighborhood aggregation (not rich edge features).

### 5.4 Multi-cell tensor construction

- Build one node-feature submatrix per cell.
- Concatenate submatrices across cells into a global input tensor.

### 5.5 Output structure

- Output mirrors input organization, but values are beamforming weights.
- Per user, the output is a beamforming vector over $N$ FAs.
- Complex weights are split into real and imaginary parts, giving $2N$ outputs per user.
- Per-cell outputs are concatenated into a global output tensor.

Practical mapping:

- Input per user: CSI over antennas ($2N$ real values).
- Output per user: beamforming weights over antennas ($2N$ real values).

## 6) Multi-GNN architecture in multi-cell deployment

The design is not one global monolithic GNN at inference.
Instead, it uses $I$ parallel per-cell GNNs (one per cell).

- Each BS runs only its local GNN with local CSI.
- GNNs share the same architecture and operational rule.
- Parameters are optimized centrally, then models are deployed per BS.

This supports **centralized training, distributed inference**, reducing signaling overhead and inference latency.

## 7) Per-cell GNN module (layer stack)

Each per-cell model follows:

1. Input MLP: $2N \rightarrow$ embedding.
2. GNN layer 1: message passing update.
3. GNN layer 2: second interaction update.
4. Final FC layer: embedding $\rightarrow 2N$ beamforming outputs.

So the pipeline is: **MLP → GNN → GNN → FC**.

## 8) Message-passing rule inside one GNN layer

For user $k$, the updated embedding is built from:

- its own current embedding, and
- an aggregated summary of other users in the same cell.

Operational steps:

1. **Neighbor transform:** apply an MLP to each neighbor embedding.
2. **Aggregation:** max-pool transformed neighbors element-wise.
3. **Combination:** concatenate self embedding with pooled neighbor summary.
4. **Update transform:** apply a second MLP to produce the new embedding.

Design implications:

- No pair-specific message function is used, which keeps computation symmetric and efficient.
- Max pooling emphasizes dominant interferers (worst-case-style interference summary).
- Two stacked GNN layers improve interaction modeling depth.

## 9) Why this scales

- Pooling over a set of neighbors yields fixed-size aggregated vectors.
- This supports varying UE counts without changing output dimensionality.
- The architecture balances interaction modeling quality and computational practicality.