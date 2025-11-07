"""
Quick experimental run with reduced parameters for fast testing.
This generates Figure 3 with fewer seeds and episodes.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.run_experiments import run_all_experiments, create_figure3

# Set random seed
np.random.seed(42)

print("\n" + "="*60)
print("QUICK EXPERIMENT RUN (Reduced Parameters)")
print("="*60)
print("\nRunning with:")
print("  - 3 seeds (instead of 5)")
print("  - 50 episodes (instead of 100)")
print("  - Should complete in ~5-10 minutes")
print("="*60 + "\n")

# Run experiments with reduced parameters
results = run_all_experiments(
    env_name='cartpole',
    n_seeds=3,
    n_episodes=50,
    save_data=True
)

# Generate figure
figures_dir = Path(__file__).parent / 'figures'
figures_dir.mkdir(exist_ok=True)
save_path = figures_dir / 'figure3.pdf'

create_figure3(results, 'cartpole', str(save_path))

print("\n" + "="*60)
print("QUICK EXPERIMENT COMPLETE!")
print("="*60)
print(f"\nFigure 3 saved to: {save_path}")
print("\nTo run full experiments (5 seeds, 100 episodes):")
print("  python main.py --experiments-only")
print("="*60 + "\n")
