"""
Generate partial Figure 3 from recovered experimental data.
This creates plots for the 3 environments that completed: CartPole, Pendulum, MountainCar.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_partial_figure3(results_file: str = 'results/section_4.2_partial_recovered.pkl',
                           output_file: str = 'figures/figure3_partial.pdf'):
    """Create figure with 3 completed environments"""
    
    # Load recovered results
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    # Setup plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    environments = ['CARTPOLE', 'PENDULUM', 'MOUNTAINCAR']
    env_titles = ['CartPole', 'Pendulum', 'MountainCar']
    
    # Method colors and styles
    method_styles = {
        'Convex-PSRL': {'color': 'blue', 'linestyle': '-', 'marker': 'o', 'label': 'Convex-PSRL (Ours)'},
        'PETS': {'color': 'red', 'linestyle': '--', 'marker': 's', 'label': 'PETS'},
        'Deep-Ensemble-VI': {'color': 'green', 'linestyle': '-.', 'marker': '^', 'label': 'Deep Ensemble VI'},
        'LaPSRL': {'color': 'orange', 'linestyle': ':', 'marker': 'D', 'label': 'LaPSRL'},
        'MPC-PSRL': {'color': 'purple', 'linestyle': '-', 'marker': 'v', 'label': 'MPC-PSRL'},
        'KSRL': {'color': 'brown', 'linestyle': '--', 'marker': 'p', 'label': 'KSRL'},
    }
    
    for idx, (env_name, title) in enumerate(zip(environments, env_titles)):
        ax = axes[idx]
        
        if env_name not in results:
            ax.text(0.5, 0.5, f'{title}\nNo data available', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        env_results = results[env_name]
        
        # Plot each method
        for method, data in env_results.items():
            if 'mean' in data:
                style = method_styles.get(method, {'color': 'gray', 'linestyle': '-'})
                
                mean_rewards = data['mean']
                std_rewards = data['std']
                episodes = np.arange(len(mean_rewards))
                
                # Plot mean line
                ax.plot(episodes, mean_rewards, 
                       color=style['color'], 
                       linestyle=style['linestyle'],
                       marker=style.get('marker', None),
                       markevery=len(episodes)//5,  # Show ~5 markers
                       linewidth=2,
                       label=style.get('label', method),
                       alpha=0.9)
                
                # Plot error band
                ax.fill_between(episodes,
                               mean_rewards - std_rewards,
                               mean_rewards + std_rewards,
                               color=style['color'],
                               alpha=0.2)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Reward', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first plot
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to {output_file}")
    
    # Also save as PNG for easy viewing
    png_file = str(output_path).replace('.pdf', '.png')
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    print(f"✅ PNG version saved to {png_file}")
    
    plt.close()
    
    return fig

def print_summary_statistics(results_file: str = 'results/section_4.2_partial_recovered.pkl'):
    """Print detailed statistics from recovered results"""
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print("\n" + "="*70)
    print("RECOVERED EXPERIMENTAL RESULTS SUMMARY")
    print("="*70)
    
    for env_name in ['CARTPOLE', 'PENDULUM', 'MOUNTAINCAR']:
        if env_name not in results:
            continue
            
        print(f"\n{env_name}")
        print("-" * 70)
        
        env_results = results[env_name]
        
        # Create table
        print(f"{'Method':<25} {'Final Reward':<20} {'Time (min)':<15}")
        print("-" * 70)
        
        # Sort by final reward (best to worst)
        sorted_methods = sorted(env_results.items(),
                               key=lambda x: x[1].get('final_reward', -np.inf),
                               reverse=True)
        
        for method, data in sorted_methods:
            if 'final_reward' in data:
                reward = data['final_reward']
                std = data['final_std']
                time_min = data['wall_clock_time'] / 60
                print(f"{method:<25} {reward:8.2f} ± {std:6.2f}      {time_min:10.1f}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    
    # Analyze CartPole performance
    if 'CARTPOLE' in results:
        cartpole = results['CARTPOLE']
        convex_reward = cartpole.get('Convex-PSRL', {}).get('final_reward', 0)
        convex_time = cartpole.get('Convex-PSRL', {}).get('wall_clock_time', 1) / 60
        
        print(f"\n📊 CartPole Analysis:")
        print(f"  - Convex-PSRL achieved {convex_reward:.1f} reward in {convex_time:.1f} minutes")
        print(f"  - Best performer: KSRL with {cartpole['KSRL']['final_reward']:.1f} reward")
        print(f"  - Fastest method: MPC-PSRL ({cartpole['MPC-PSRL']['wall_clock_time']/60:.1f} min)")
        
        # Check if hit 195/200 threshold
        if convex_reward >= 195:
            print(f"  ✅ Convex-PSRL EXCEEDS 195/200 threshold!")
        else:
            print(f"  ⚠️  Convex-PSRL below 195/200 threshold (got {convex_reward:.1f})")
            print(f"     Note: This is QUICK MODE (3 seeds, 50 episodes) - full run may perform better")
    
    # Analyze Pendulum
    if 'PENDULUM' in results:
        pendulum = results['PENDULUM']
        convex_reward = pendulum.get('Convex-PSRL', {}).get('final_reward', 0)
        best_method = max(pendulum.items(), key=lambda x: x[1].get('final_reward', -np.inf))
        
        print(f"\n📊 Pendulum Analysis:")
        print(f"  - Convex-PSRL achieved {convex_reward:.1f} reward")
        print(f"  - Best performer: {best_method[0]} with {best_method[1]['final_reward']:.1f} reward")
        print(f"  - Deep-Ensemble-VI took {pendulum['Deep-Ensemble-VI']['wall_clock_time']/3600:.1f} hours!")
    
    # Analyze MountainCar
    if 'MOUNTAINCAR' in results:
        mountaincar = results['MOUNTAINCAR']
        convex_reward = mountaincar.get('Convex-PSRL', {}).get('final_reward', 0)
        
        print(f"\n📊 MountainCar Analysis:")
        print(f"  - Convex-PSRL achieved {convex_reward:.1f} reward")
        print(f"  - PETS achieved {mountaincar['PETS']['final_reward']:.1f} (best)")
        print(f"  - Deep-Ensemble-VI took {mountaincar['Deep-Ensemble-VI']['wall_clock_time']/3600:.1f} hours!")
        print(f"  ⚠️  KSRL did not complete (interrupted)")
    
    print("\n" + "="*70)
    print("⚡ COMPUTATIONAL COST:")
    print("="*70)
    total_time_hours = sum(
        data.get('wall_clock_time', 0) 
        for env_results in results.values() 
        for data in env_results.values()
    ) / 3600
    
    print(f"  Total computation time: {total_time_hours:.1f} hours (≈ {total_time_hours/24:.1f} days)")
    print(f"  Completed environments: {len(results)}/3 classic control")
    print(f"  Remaining: 3 MuJoCo environments (Walker2d, Hopper, HalfCheetah)")
    print(f"  Estimated time for MuJoCo: {total_time_hours:.1f} hours (similar complexity)")

if __name__ == '__main__':
    print("Generating partial Figure 3 from recovered data...")
    create_partial_figure3()
    print_summary_statistics()
    
    print("\n" + "="*70)
    print("✅ RECOVERY COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - figures/figure3_partial.pdf (3-environment comparison)")
    print("  - figures/figure3_partial.png (easy viewing)")
    print("\nYour options now:")
    print("  1. Use these 3 environments for your paper (may be sufficient)")
    print("  2. Continue experiments for MuJoCo environments only")
    print("  3. Re-run with optimizations to speed up (reduce hidden_dim, fewer gradients)")
    print("="*70)
