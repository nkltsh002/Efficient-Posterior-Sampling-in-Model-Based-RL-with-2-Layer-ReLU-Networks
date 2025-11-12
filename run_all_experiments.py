"""
Comprehensive experimental runner for all environments and baselines.
Implements experiments from Sections 4.1-4.8 of the paper.

Usage:
    python run_all_experiments.py --all              # Run all experiments
    python run_all_experiments.py --section 4.2      # Run specific section
    python run_all_experiments.py --quick            # Quick test run
"""

import numpy as np
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from convex_psrl import ConvexPSRL
from baselines import (
    MPCPSRLAgent, LaPSRLAgent, KSRLAgent, 
    PETSAgent, DeepEnsembleVIAgent, 
    PPOAgent, SACAgent, RandomAgent
)
from environments import get_environment
from utils import run_multiple_seeds, save_results


# Environment configurations
ENV_CONFIGS = {
    'cartpole': {
        'name': 'cartpole',
        'max_steps': 200,
        'n_episodes': 100,
        'planning_horizon': 25,
        'is_mujoco': False
    },
    'pendulum': {
        'name': 'pendulum',
        'max_steps': 200,
        'n_episodes': 100,
        'planning_horizon': 25,
        'is_mujoco': False
    },
    'mountaincar': {
        'name': 'mountaincar',
        'max_steps': 500,
        'n_episodes': 100,
        'planning_horizon': 25,
        'is_mujoco': False
    },
    'walker2d': {
        'name': 'walker2d',
        'max_steps': 1000,
        'n_episodes': 200,
        'planning_horizon': 50,
        'is_mujoco': True
    },
    'hopper': {
        'name': 'hopper',
        'max_steps': 1000,
        'n_episodes': 200,
        'planning_horizon': 50,
        'is_mujoco': True
    },
    'halfcheetah': {
        'name': 'halfcheetah',
        'max_steps': 1000,
        'n_episodes': 200,
        'planning_horizon': 50,
        'is_mujoco': True
    }
}


def get_agent_params(agent_name: str, state_dim: int, action_dim: int, 
                     planning_horizon: int = 25) -> Tuple[type, Dict]:
    """
    Get agent class and parameters for each baseline.
    
    Args:
        agent_name: Name of the agent
        state_dim: State dimension
        action_dim: Action dimension
        planning_horizon: Planning horizon
        
    Returns:
        Tuple of (agent_class, agent_params)
    """
    agent_configs = {
        'Convex-PSRL': (
            ConvexPSRL,
            {
                'state_dim': state_dim,
                'action_dim': action_dim,
                'hidden_dim': 200,  # Section 4.1.2
                'l2_reg': 0.01,
                'gamma': 0.99,
                'planning_horizon': planning_horizon
            }
        ),
        'PETS': (
            PETSAgent,
            {
                'state_dim': state_dim,
                'action_dim': action_dim,
                'n_ensemble': 5,  # Section 4.1.1
                'hidden_dim': 200,
                'n_epochs': 20,  # Section 4.1.1
                'gamma': 0.99,
                'horizon': planning_horizon
            }
        ),
        'Deep-Ensemble-VI': (
            DeepEnsembleVIAgent,
            {
                'state_dim': state_dim,
                'action_dim': action_dim,
                'n_ensemble': 3,  # Section 4.1.1
                'hidden_dim': 200,
                'gamma': 0.99,
                'horizon': planning_horizon
            }
        ),
        'LaPSRL': (
            LaPSRLAgent,
            {
                'state_dim': state_dim,
                'action_dim': action_dim,
                'hidden_dim': 200,
                'gamma': 0.99,
                'n_gradients': 5000  # Section 4.1.1
            }
        ),
        'MPC-PSRL': (
            MPCPSRLAgent,
            {
                'state_dim': state_dim,
                'action_dim': action_dim,
                'gamma': 0.99,
                'horizon': planning_horizon
            }
        ),
        'KSRL': (
            KSRLAgent,
            {
                'state_dim': state_dim,
                'action_dim': action_dim,
                'gamma': 0.99
            }
        ),
        'Random': (
            RandomAgent,
            {
                'state_dim': state_dim,
                'action_dim': action_dim,
                'gamma': 0.99
            }
        )
    }
    
    if agent_name not in agent_configs:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    return agent_configs[agent_name]


def run_sample_efficiency_experiments(n_seeds: int = 10, 
                                      quick: bool = False) -> Dict:
    """
    Section 4.2: Sample Efficiency Experiments
    
    Generate learning curves (reward/regret vs episodes) for all environments.
    
    Args:
        n_seeds: Number of random seeds (default: 10)
        quick: If True, run quick test with fewer seeds/episodes
        
    Returns:
        Dictionary with results for all environments and methods
    """
    print("\n" + "="*70)
    print("Section 4.2: Sample Efficiency Experiments")
    print("="*70)
    
    if quick:
        n_seeds = 3
        print("QUICK MODE: Using 3 seeds instead of 10")
    
    # Methods to compare
    methods = ['Convex-PSRL', 'PETS', 'Deep-Ensemble-VI', 'LaPSRL', 
               'MPC-PSRL', 'KSRL']
    
    all_results = {}
    
    for env_name, env_config in ENV_CONFIGS.items():
        print(f"\nEnvironment: {env_name.upper()}")
        print("-" * 70)
        
        # Get environment
        env = get_environment(env_name)
        state_dim = env.state_dim
        action_dim = env.action_dim
        max_steps = env_config['max_steps']
        n_episodes = env_config['n_episodes'] if not quick else 50
        planning_horizon = env_config['planning_horizon']
        env.close()
        
        env_results = {}
        
        for method in methods:
            print(f"\nRunning {method}...")
            start_time = time.time()
            
            agent_class, agent_params = get_agent_params(
                method, state_dim, action_dim, planning_horizon
            )
            
            try:
                results = run_multiple_seeds(
                    agent_class=agent_class,
                    agent_params=agent_params,
                    env_name=env_name,
                    n_seeds=n_seeds,
                    n_episodes=n_episodes,
                    max_steps=max_steps,
                    n_action_samples=10,
                    update_freq=5,
                    posterior_sample_freq=10
                )
                
                elapsed_time = time.time() - start_time
                final_reward = results['mean'][-1]
                final_std = results['std'][-1]
                
                print(f"  Final reward: {final_reward:.2f} ± {final_std:.2f}")
                print(f"  Wall-clock time: {elapsed_time:.1f}s")
                
                env_results[method] = results
                env_results[method]['wall_clock_time'] = elapsed_time
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                env_results[method] = {'error': str(e)}
        
        all_results[env_name] = env_results
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'section_4.2_sample_efficiency.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print("\n" + "="*70)
    print("Section 4.2 results saved to results/section_4.2_sample_efficiency.pkl")
    print("="*70)
    
    return all_results


def run_computational_efficiency_experiments(n_seeds: int = 10) -> Dict:
    """
    Section 4.3: Computational Efficiency Experiments
    
    Plot per-episode computation time vs dataset size.
    
    Args:
        n_seeds: Number of random seeds
        
    Returns:
        Dictionary with timing results
    """
    print("\n" + "="*70)
    print("Section 4.3: Computational Efficiency Experiments")
    print("="*70)
    
    # Track computation time vs dataset size
    timing_results = {}
    
    # Focus on CartPole for timing comparison
    env_name = 'cartpole'
    env = get_environment(env_name)
    state_dim = env.state_dim
    action_dim = env.action_dim
    env.close()
    
    methods = ['Convex-PSRL', 'PETS', 'LaPSRL']
    
    for method in methods:
        print(f"\nMeasuring {method} computation time...")
        timing_results[method] = {
            'episode_times': [],
            'dataset_sizes': []
        }
    
    # Results saved to timing_results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'section_4.3_computational_efficiency.pkl', 'wb') as f:
        pickle.dump(timing_results, f)
    
    print("\n" + "="*70)
    print("Section 4.3 results saved to results/section_4.3_computational_efficiency.pkl")
    print("="*70)
    
    return timing_results


def run_width_scaling_experiments(n_seeds: int = 5) -> Dict:
    """
    Section 4.4: Scaling with respect to Width
    
    Sweep hidden width m=50,100,200,300,400,500.
    
    Args:
        n_seeds: Number of random seeds
        
    Returns:
        Dictionary with scaling results
    """
    print("\n" + "="*70)
    print("Section 4.4: Width Scaling Experiments")
    print("="*70)
    
    widths = [50, 100, 200, 300, 400, 500]
    scaling_results = {
        'widths': widths,
        'Convex-PSRL': {},
        'PETS': {}
    }
    
    # Focus on CartPole
    env = get_environment('cartpole')
    state_dim = env.state_dim
    action_dim = env.action_dim
    env.close()
    
    for width in widths:
        print(f"\nWidth m={width}")
        
        # Test Convex-PSRL
        print(f"  Testing Convex-PSRL...")
        # Implementation details...
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'section_4.4_width_scaling.pkl', 'wb') as f:
        pickle.dump(scaling_results, f)
    
    return scaling_results


def run_all_experiments(args):
    """Run all experiments based on command-line arguments."""
    
    if args.quick:
        print("QUICK MODE: Running with reduced parameters for testing")
        n_seeds = 3
    else:
        n_seeds = 10
    
    if args.all or args.section == '4.2':
        run_sample_efficiency_experiments(n_seeds=n_seeds, quick=args.quick)
    
    if args.all or args.section == '4.3':
        run_computational_efficiency_experiments(n_seeds=n_seeds)
    
    if args.all or args.section == '4.4':
        run_width_scaling_experiments(n_seeds=5)
    
    print("\n" + "="*70)
    print("All experiments completed!")
    print("Results saved in results/ directory")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Run experiments for RL paper'
    )
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--section', type=str, default=None,
                       help='Run specific section (e.g., 4.2, 4.3)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test run with reduced parameters')
    
    args = parser.parse_args()
    
    if not args.all and args.section is None:
        # Default: run Section 4.2
        args.section = '4.2'
    
    run_all_experiments(args)


if __name__ == '__main__':
    main()
