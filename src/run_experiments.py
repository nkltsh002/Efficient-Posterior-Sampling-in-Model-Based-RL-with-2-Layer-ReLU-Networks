"""
Run experiments and generate Figure 3: Sample Efficiency Comparison
Compares Convex-PSRL against baselines (MPC-PSRL, LaPSRL, KSRL, Random)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from convex_psrl import ConvexPSRL
from baselines import MPCPSRLAgent, LaPSRLAgent, KSRLAgent, RandomAgent
from environments import get_environment
from utils import run_multiple_seeds, save_results, smooth_curve


def run_all_experiments(env_name: str = 'cartpole', 
                        n_seeds: int = 5, 
                        n_episodes: int = 100,
                        save_data: bool = True):
    """
    Run experiments with all methods on specified environment.
    
    Args:
        env_name: Environment name ('cartpole', 'pendulum', 'mountaincar')
        n_seeds: Number of random seeds
        n_episodes: Number of training episodes per seed
        save_data: Whether to save raw data
        
    Returns:
        Dictionary with results for all methods
    """
    print(f"\n{'='*60}")
    print(f"Running experiments on {env_name.upper()}")
    print(f"Seeds: {n_seeds}, Episodes: {n_episodes}")
    print(f"{'='*60}\n")
    
    # Get environment info
    env = get_environment(env_name)
    state_dim = env.state_dim
    action_dim = env.action_dim
    env.close()
    
    results = {}
    
    # 1. Convex-PSRL
    print("Running Convex-PSRL...")
    results['Convex-PSRL'] = run_multiple_seeds(
        agent_class=ConvexPSRL,
        agent_params={
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_dim': 32,
            'l2_reg': 0.01,
            'gamma': 0.99
        },
        env_name=env_name,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        max_steps=200,
        n_action_samples=10,
        update_freq=5,
        posterior_sample_freq=10
    )
    print(f"  Final reward: {results['Convex-PSRL']['mean'][-1]:.2f} ± {results['Convex-PSRL']['std'][-1]:.2f}")
    
    # 2. MPC-PSRL
    print("Running MPC-PSRL...")
    results['MPC-PSRL'] = run_multiple_seeds(
        agent_class=MPCPSRLAgent,
        agent_params={
            'state_dim': state_dim,
            'action_dim': action_dim,
            'gamma': 0.99,
            'horizon': 5
        },
        env_name=env_name,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        max_steps=200,
        n_action_samples=10,
        update_freq=5,
        posterior_sample_freq=10
    )
    print(f"  Final reward: {results['MPC-PSRL']['mean'][-1]:.2f} ± {results['MPC-PSRL']['std'][-1]:.2f}")
    
    # 3. LaPSRL
    print("Running LaPSRL...")
    results['LaPSRL'] = run_multiple_seeds(
        agent_class=LaPSRLAgent,
        agent_params={
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_dim': 32,
            'gamma': 0.99
        },
        env_name=env_name,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        max_steps=200,
        n_action_samples=10,
        update_freq=5,
        posterior_sample_freq=10
    )
    print(f"  Final reward: {results['LaPSRL']['mean'][-1]:.2f} ± {results['LaPSRL']['std'][-1]:.2f}")
    
    # 4. KSRL
    print("Running KSRL...")
    results['KSRL'] = run_multiple_seeds(
        agent_class=KSRLAgent,
        agent_params={
            'state_dim': state_dim,
            'action_dim': action_dim,
            'gamma': 0.99
        },
        env_name=env_name,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        max_steps=200,
        n_action_samples=10,
        update_freq=5,
        posterior_sample_freq=10
    )
    print(f"  Final reward: {results['KSRL']['mean'][-1]:.2f} ± {results['KSRL']['std'][-1]:.2f}")
    
    # 5. Random Baseline
    print("Running Random Agent...")
    results['Random'] = run_multiple_seeds(
        agent_class=RandomAgent,
        agent_params={
            'state_dim': state_dim,
            'action_dim': action_dim,
            'gamma': 0.99
        },
        env_name=env_name,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        max_steps=200,
        n_action_samples=10,
        update_freq=5,
        posterior_sample_freq=10
    )
    print(f"  Final reward: {results['Random']['mean'][-1]:.2f} ± {results['Random']['std'][-1]:.2f}")
    
    # Save results
    if save_data:
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        save_path = results_dir / f'{env_name}_results.pkl'
        save_results(results, str(save_path))
    
    return results


def create_figure3(results_dict: dict, env_name: str, 
                   save_path: str = '../figures/figure3.pdf'):
    """
    Create learning curve comparison plot (Figure 3).
    
    Args:
        results_dict: Dictionary with results from all methods
        env_name: Environment name for plot title
        save_path: Path to save figure
    """
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Colors and styles for each method
    colors = {
        'Convex-PSRL': '#27ae60',
        'MPC-PSRL': '#3498db',
        'LaPSRL': '#e74c3c',
        'KSRL': '#f39c12',
        'Random': '#95a5a6'
    }
    
    linestyles = {
        'Convex-PSRL': '-',
        'MPC-PSRL': '--',
        'LaPSRL': '-.',
        'KSRL': ':',
        'Random': '-'
    }
    
    linewidths = {
        'Convex-PSRL': 3,
        'MPC-PSRL': 2,
        'LaPSRL': 2,
        'KSRL': 2,
        'Random': 1.5
    }
    
    # Plot each method
    for method_name, results in results_dict.items():
        mean_rewards = results['mean']
        std_rewards = results['std']
        episodes = np.arange(len(mean_rewards))
        
        # Smooth curves for better visualization
        mean_smooth = smooth_curve(mean_rewards, window=5)
        std_smooth = smooth_curve(std_rewards, window=5)
        
        # Plot mean line
        ax.plot(episodes, mean_smooth, 
               label=method_name,
               color=colors[method_name],
               linestyle=linestyles[method_name],
               linewidth=linewidths[method_name],
               alpha=0.9)
        
        # Plot confidence band (mean ± std)
        ax.fill_between(episodes,
                       mean_smooth - std_smooth,
                       mean_smooth + std_smooth,
                       color=colors[method_name],
                       alpha=0.2)
    
    # Formatting
    ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Return', fontsize=14, fontweight='bold')
    
    env_display = env_name.replace('_', ' ').title()
    ax.set_title(f'Sample Efficiency Comparison on {env_display}',
                fontsize=16, fontweight='bold', pad=15)
    
    ax.legend(loc='lower right', fontsize=11, frameon=True, 
             fancybox=True, shadow=True, ncol=1)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set reasonable y-limits based on data
    all_means = [results['mean'] for results in results_dict.values()]
    y_min = min([np.min(m) for m in all_means])
    y_max = max([np.max(m) for m in all_means])
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure 3 saved to {save_path}")
    plt.close()


def generate_all_environment_figures():
    """
    Generate Figure 3 for multiple environments.
    """
    environments = ['cartpole']  # Can add: 'pendulum', 'mountaincar'
    
    for env_name in environments:
        print(f"\n{'#'*60}")
        print(f"# Processing environment: {env_name}")
        print(f"{'#'*60}")
        
        # Run experiments
        results = run_all_experiments(
            env_name=env_name,
            n_seeds=5,
            n_episodes=100,
            save_data=True
        )
        
        # Generate figure
        figures_dir = Path(__file__).parent.parent / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        if env_name == 'cartpole':
            save_path = figures_dir / 'figure3.pdf'
        else:
            save_path = figures_dir / f'figure3_{env_name}.pdf'
        
        create_figure3(results, env_name, str(save_path))
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate all figures
    generate_all_environment_figures()
