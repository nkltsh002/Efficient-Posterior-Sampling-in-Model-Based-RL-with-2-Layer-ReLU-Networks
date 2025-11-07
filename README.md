# Efficient Posterior Sampling in Model-Based RL with 2-Layer ReLU Networks

This repository contains the implementation and experimental code for the paper **"Efficient Posterior Sampling in Model-Based RL with 2-Layer ReLU Networks"**.

## Overview

**Convex-PSRL** is a novel model-based reinforcement learning algorithm that leverages the convex dual formulation of 2-layer ReLU networks to perform exact MAP (Maximum A Posteriori) inference for posterior sampling. Unlike standard deep PSRL methods that require computationally expensive MCMC sampling, Convex-PSRL solves a tractable convex quadratic program, enabling efficient and provably optimal posterior sampling.

### Key Contributions

- **Exact MAP Inference**: Reformulates 2-layer ReLU network training as a convex optimization problem
- **Tractable Posterior Sampling**: Eliminates the need for MCMC, achieving polynomial-time complexity
- **Sample Efficiency**: Demonstrates superior performance compared to MPC-PSRL, LaPSRL, KSRL, and random baselines
- **Theoretical Guarantees**: Provable convergence and optimality properties

## Repository Structure

```
.
├── main.py                      # Main entry point for running experiments
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── src/
│   ├── convex_psrl.py          # Convex-PSRL implementation
│   ├── baselines.py            # Baseline algorithms (MPC-PSRL, LaPSRL, KSRL)
│   ├── environments.py         # RL environments (CartPole, Pendulum, etc.)
│   ├── utils.py                # Utility functions for training and evaluation
│   ├── generate_figure1.py     # Generate conceptual contrast diagram
│   ├── generate_figure2.py     # Generate pipeline flowchart
│   └── run_experiments.py      # Run experiments and generate learning curves
├── figures/                     # Generated figures (PDF)
│   ├── figure1.pdf             # Conceptual contrast: Deep PSRL vs Convex-PSRL
│   ├── figure2.pdf             # Convex optimization pipeline flowchart
│   └── figure3.pdf             # Sample efficiency comparison
└── results/                     # Experimental data (pickle files)
    └── cartpole_results.pkl    # Raw experimental results
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**

```bash
git clone https://github.com/nkltsh002/Efficient-Posterior-Sampling-in-Model-Based-RL-with-2-Layer-ReLU-Networks.git
cd Efficient-Posterior-Sampling-in-Model-Based-RL-with-2-Layer-ReLU-Networks
```

2. **Create a virtual environment (recommended):**

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.10.0` - Scientific computing
- `cvxpy>=1.4.0` - Convex optimization (CRITICAL for Convex-PSRL)
- `matplotlib>=3.7.0` - Plotting and visualization
- `gymnasium>=0.29.0` - RL environments
- `torch>=2.0.0` - Neural networks (for baselines)
- `tqdm>=4.65.0` - Progress bars
- `pandas>=2.0.0` - Data manipulation
- `seaborn>=0.12.0` - Statistical plotting

## Running Experiments

### Quick Start: Full Pipeline

To run the complete experimental pipeline (generate all figures and run experiments):

```bash
python main.py
```

This will:
1. Generate Figure 1 (Conceptual Contrast Diagram)
2. Generate Figure 2 (Convex Optimization Pipeline)
3. Run experiments with all methods (Convex-PSRL, MPC-PSRL, LaPSRL, KSRL, Random)
4. Generate Figure 3 (Sample Efficiency Comparison)
5. Save all results to `figures/` and `results/` directories

**Expected runtime:** 10-30 minutes (depending on your system)

### Generate Only Conceptual Figures

To quickly generate Figure 1 and Figure 2 without running experiments:

```bash
python main.py --figures-only
```

**Expected runtime:** < 10 seconds

### Run Only Experiments

To run experiments and generate Figure 3 (skipping conceptual figures):

```bash
python main.py --experiments-only
```

**Expected runtime:** 10-30 minutes

## Reproducing Individual Figures

### Figure 1: Conceptual Contrast Diagram

Shows the comparison between Standard Deep PSRL (MCMC-based, intractable) and Convex-PSRL (convex program, tractable).

```bash
python src/generate_figure1.py
```

**Output:** `figures/figure1.pdf`

### Figure 2: Convex Optimization Pipeline

Illustrates the step-by-step Convex-PSRL process: data collection → matrix formulation → convex QP → MAP weights → policy.

```bash
python src/generate_figure2.py
```

**Output:** `figures/figure2.pdf`

### Figure 3: Sample Efficiency Comparison

Learning curves comparing Convex-PSRL against baselines on CartPole environment.

```bash
python src/run_experiments.py
```

**Output:** `figures/figure3.pdf` and `results/cartpole_results.pkl`

**Parameters to modify** (edit `src/run_experiments.py`):
- `n_seeds`: Number of random seeds (default: 5)
- `n_episodes`: Training episodes per seed (default: 100)
- `environments`: List of environments to test (default: `['cartpole']`)

## Understanding the Results

### Figure 1: Conceptual Contrast

- **Left side:** Standard Deep PSRL requires MCMC sampling (computationally intractable)
- **Right side:** Convex-PSRL uses convex optimization (tractable, polynomial time)
- **Key insight:** 2-layer ReLU networks enable exact inference via convex reformulation

### Figure 2: Pipeline Flowchart

Shows the 5-step Convex-PSRL process with annotations:
1. **Collect Data:** Gather trajectories (s, a, r, s')
2. **Formulate Matrices:** Construct X (inputs) and Y (targets)
3. **Set Up Convex Program:** min ||Y - f(X)||² + λ||W||² (convex dual formulation)
4. **Solve QP:** Obtain MAP weights w* using CVXPY/OSQP
5. **Use Model:** Plan actions via MPC, repeat (posterior sampling loop)

### Figure 3: Learning Curves

- **X-axis:** Training episodes
- **Y-axis:** Average return (reward)
- **Lines:** Mean performance across seeds
- **Shaded regions:** ±1 standard deviation (uncertainty bands)
- **Expected result:** Convex-PSRL should outperform baselines in sample efficiency

**Baseline descriptions:**
- **MPC-PSRL**: Linear models with Bayesian updates (Fan & Ming 2021)
- **LaPSRL**: Neural network with Laplace approximation
- **KSRL**: Gaussian Process-based Thompson Sampling
- **Random**: Random action selection (lower bound)

## Customization

### Running on Different Environments

To test on additional environments, edit `src/run_experiments.py`:

```python
environments = ['cartpole', 'pendulum', 'mountaincar']
```

Available environments:
- `cartpole`: Discrete control (balance pole)
- `pendulum`: Continuous control (swing-up task)
- `mountaincar`: Continuous control (reach goal)

### Adjusting Hyperparameters

Edit `src/run_experiments.py` to modify agent hyperparameters:

```python
agent_params = {
    'state_dim': state_dim,
    'action_dim': action_dim,
    'hidden_dim': 32,        # Hidden layer size
    'l2_reg': 0.01,          # L2 regularization
    'gamma': 0.99            # Discount factor
}
```

### Quick Testing Mode

For rapid iteration during development, reduce computational load:

```python
# In src/run_experiments.py, modify:
n_seeds = 2              # Fewer seeds
n_episodes = 50          # Fewer episodes
```

## Troubleshooting

### Common Issues

1. **CVXPY solver fails:**
   - Ensure `cvxpy>=1.4.0` is installed
   - The code includes a fallback to gradient descent if convex solver fails
   - Check solver status in console output

2. **Import errors:**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Ensure you're in the correct virtual environment

3. **Slow experiments:**
   - Reduce `n_seeds` or `n_episodes` in `run_experiments.py`
   - Use `--figures-only` to skip experiments
   - Consider using a GPU if available (install `torch` with CUDA support)

4. **Memory errors:**
   - Reduce `hidden_dim` in agent parameters
   - Run experiments sequentially (one environment at a time)
   - Reduce `n_seeds`

### Verifying Installation

Test that all modules can be imported:

```bash
python -c "import cvxpy; import gymnasium; import matplotlib; print('All imports successful!')"
```

## Paper Integration

### Using Figures in LaTeX

Include the generated PDFs in your paper:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/figure1.pdf}
    \caption{Conceptual comparison between Standard Deep PSRL (requiring 
             intractable MCMC sampling) and our Convex-PSRL method (using 
             tractable convex optimization for 2-layer ReLU networks).}
    \label{fig:conceptual_contrast}
\end{figure}
```

### Suggested Captions

**Figure 1:**
> "Conceptual comparison between Standard Deep PSRL and Convex-PSRL. Left: Deep networks require MCMC for posterior sampling (intractable). Right: 2-layer ReLU networks enable exact MAP inference via convex optimization (tractable)."

**Figure 2:**
> "Convex-PSRL optimization pipeline. The algorithm iteratively collects trajectories, formulates convex programs, solves for MAP weights, and uses the learned model for planning."

**Figure 3:**
> "Sample efficiency comparison on CartPole environment. Convex-PSRL (green) achieves superior performance compared to MPC-PSRL, LaPSRL, KSRL, and random baselines. Shaded regions represent ±1 standard deviation across 5 random seeds."

## Citation

If you use this code in your research, please cite:

```bibtex
@article{convex_psrl_2025,
  title={Efficient Posterior Sampling in Model-Based RL with 2-Layer ReLU Networks},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Built on top of [Gymnasium](https://gymnasium.farama.org/) for RL environments
- Uses [CVXPY](https://www.cvxpy.org/) for convex optimization
- Baselines inspired by Fan & Ming (2021), and related PSRL literature

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Email: nkltsh002@myuct.ac.za (replace with your email)

---

**Last updated:** November 7, 2025
