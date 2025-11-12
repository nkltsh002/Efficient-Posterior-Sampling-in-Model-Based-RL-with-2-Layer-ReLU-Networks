"""
Generate Figure 3: Sample Efficiency Comparison across all 6 environments.
Shows learning curves for all methods with 10-seed means and error bands.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from typing import Dict


def smooth_curve(rewards: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving average smoothing to learning curve."""
    if len(rewards) < window:
        return rewards
    return np.convolve(rewards, np.ones(window)/window, mode='valid')


def create_figure3_all_envs(results_file: str = 'results/section_4.2_sample_efficiency.pkl',
                            save_path: str = 'figures/figure3.pdf'):
    """
    Create Figure 3 with all 6 environments and all baselines.
    
    Args:
        results_file: Path to results pickle file
        save_path: Path to save figure
    """
    # Load results
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)
    
    # Setup plot style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")
    
    # Create 2x3 subplot grid for 6 environments
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Environment names
    env_names = {
        'cartpole': 'CartPole',
        'pendulum': 'Pendulum',
        'mountaincar': 'MountainCar',
        'walker2d': 'Walker2d',
        'hopper': 'Hopper',
        'halfcheetah': 'HalfCheetah'
    }
    
    # Method colors and styles
    method_styles = {
        'Convex-PSRL': {'color': '#2ecc71', 'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.2},
        'PETS': {'color': '#3498db', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.15},
        'Deep-Ensemble-VI': {'color': '#9b59b6', 'linestyle': '-.', 'linewidth': 2, 'alpha': 0.15},
        'LaPSRL': {'color': '#e74c3c', 'linestyle': ':', 'linewidth': 2, 'alpha': 0.15},
        'MPC-PSRL': {'color': '#f39c12', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.1},
        'KSRL': {'color': '#1abc9c', 'linestyle': '-.', 'linewidth': 1.5, 'alpha': 0.1}
    }
    
    # Plot each environment
    for idx, (env_key, env_name) in enumerate(env_names.items()):
        ax = axes[idx]
        
        if env_key not in all_results:
            ax.text(0.5, 0.5, f'{env_name}\n(No data)', 
                   ha='center', va='center', fontsize=14)
            ax.set_title(env_name, fontsize=14, fontweight='bold')
            continue
        
        env_results = all_results[env_key]
        
        # Plot each method
        for method, style in method_styles.items():
            if method not in env_results:
                continue
            
            results = env_results[method]
            
            if 'error' in results:
                continue
            
            mean_rewards = results['mean']
            std_rewards = results['std']
            
            # Smooth curves
            smoothed_mean = smooth_curve(mean_rewards, window=5)
            smoothed_std = smooth_curve(std_rewards, window=5)
            
            episodes = np.arange(len(smoothed_mean))
            
            # Plot mean with error band
            ax.plot(episodes, smoothed_mean, 
                   color=style['color'], 
                   linestyle=style['linestyle'],
                   linewidth=style['linewidth'],
                   label=method)
            
            ax.fill_between(episodes, 
                           smoothed_mean - smoothed_std,
                           smoothed_mean + smoothed_std,
                           color=style['color'],
                           alpha=style['alpha'])
        
        ax.set_title(env_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Episodes', fontsize=12)
        ax.set_ylabel('Cumulative Reward', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # Overall title
    fig.suptitle('Figure 3: Sample Efficiency Comparison (10 seeds, mean ± std)',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    save_dir = Path(save_path).parent
    save_dir.mkdir(exist_ok=True)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\n✓ Figure 3 saved to {save_path}")
    
    plt.close()


def verify_cartpole_performance(results_file: str = 'results/section_4.2_sample_efficiency.pkl'):
    """
    Verify CartPole performance claim: "hits 195/200 in 3 episodes".
    
    Args:
        results_file: Path to results pickle file
    """
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)
    
    if 'cartpole' not in all_results:
        print("✗ CartPole results not found")
        return
    
    cartpole_results = all_results['cartpole']
    
    if 'Convex-PSRL' not in cartpole_results:
        print("✗ Convex-PSRL results not found for CartPole")
        return
    
    rewards = cartpole_results['Convex-PSRL']['mean']
    
    # Check if hits 195/200 in first 3 episodes
    if len(rewards) >= 3:
        early_rewards = rewards[:3]
        max_early = np.max(early_rewards)
        
        print(f"\nCartPole Performance Verification:")
        print(f"  Episode 1: {rewards[0]:.1f}")
        print(f"  Episode 2: {rewards[1]:.1f}")
        print(f"  Episode 3: {rewards[2]:.1f}")
        print(f"  Best in first 3 episodes: {max_early:.1f}/200")
        
        if max_early >= 195:
            print(f"  ✓ Claim verified: Convex-PSRL hits ≥195/200 in first 3 episodes")
        else:
            print(f"  ⚠ Claim not met: Best score {max_early:.1f} < 195")
            print(f"     Consider updating text or running more seeds")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate Figure 3: Sample Efficiency Comparison'
    )
    parser.add_argument('--results', type=str, 
                       default='results/section_4.2_sample_efficiency.pkl',
                       help='Path to results file')
    parser.add_argument('--output', type=str, default='figures/figure3.pdf',
                       help='Output path for figure')
    parser.add_argument('--verify', action='store_true',
                       help='Verify CartPole performance claim')
    
    args = parser.parse_args()
    
    # Check if results exist
    if not Path(args.results).exists():
        print(f"✗ Results file not found: {args.results}")
        print("  Run experiments first: python run_all_experiments.py --section 4.2")
        return
    
    # Generate figure
    create_figure3_all_envs(args.results, args.output)
    
    # Verify performance claim
    if args.verify:
        verify_cartpole_performance(args.results)


if __name__ == '__main__':
    main()
