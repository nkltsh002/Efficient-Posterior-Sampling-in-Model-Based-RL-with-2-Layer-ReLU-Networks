"""
Continue experiments for MuJoCo environments ONLY.
This script:
1. Runs ONLY Walker2d, Hopper, HalfCheetah (skips CartPole, Pendulum, MountainCar)
2. Uses OPTIMIZED settings to reduce runtime
3. Saves checkpoints after each environment
4. Merges results with recovered data at the end
"""

import sys
import time
import pickle
import numpy as np
from pathlib import Path

sys.path.append('src')

from utils import run_multiple_seeds
from environments import get_environment
from convex_psrl import ConvexPSRL
from baselines import (
    PETSAgent, DeepEnsembleVIAgent, LaPSRLAgent,
    MPCPSRLAgent, KSRLAgent, RandomAgent
)

# OPTIMIZED CONFIGURATIONS for faster runtime
ENV_CONFIGS = {
    'WALKER2D': {
        'max_steps': 500,
        'n_episodes': 40,  # Reduced from 50 for quick mode
        'planning_horizon': 40,  # Reduced from 50
        'is_mujoco': True
    },
    'HOPPER': {
        'max_steps': 500,
        'n_episodes': 40,
        'planning_horizon': 40,
        'is_mujoco': True
    },
    'HALFCHEETAH': {
        'max_steps': 500,
        'n_episodes': 40,
        'planning_horizon': 40,
        'is_mujoco': True
    }
}

def get_agent_params_optimized(env_name: str, agent_name: str, planning_horizon: int):
    """Get OPTIMIZED agent parameters for faster runtime"""
    
    if agent_name == 'Convex-PSRL':
        return ConvexPSRL, {
            'state_dim': get_environment(env_name).state_dim,
            'action_dim': get_environment(env_name).action_dim,
            'hidden_dim': 150,  # REDUCED from 200 for speed
            'learning_rate': 0.001,
            'regularization': 0.01,
            'planning_horizon': planning_horizon,
            'solver': 'mosek',
            'solver_timeout': 45,  # REDUCED from 60
            'solver_tol': 1e-5,  # RELAXED from 1e-6
            'planning_method': 'cem'
        }
    
    elif agent_name == 'PETS':
        return PETSAgent, {
            'state_dim': get_environment(env_name).state_dim,
            'action_dim': get_environment(env_name).action_dim,
            'hidden_dim': 150,  # REDUCED from 200
            'n_networks': 4,  # REDUCED from 5
            'n_epochs': 15,  # REDUCED from 20
            'batch_size': 64
        }
    
    elif agent_name == 'Deep-Ensemble-VI':
        return DeepEnsembleVIAgent, {
            'state_dim': get_environment(env_name).state_dim,
            'action_dim': get_environment(env_name).action_dim,
            'hidden_dim': 150,  # REDUCED from 200
            'n_networks': 3,
            'n_epochs': 15,  # REDUCED from 20
            'batch_size': 64
        }
    
    elif agent_name == 'LaPSRL':
        return LaPSRLAgent, {
            'state_dim': get_environment(env_name).state_dim,
            'action_dim': get_environment(env_name).action_dim,
            'hidden_dim': 150,  # REDUCED from 200
            'learning_rate': 0.001,
            'n_gradients': 3000,  # REDUCED from 5000
            'optimizer': 'sarah_ld'
        }
    
    elif agent_name == 'MPC-PSRL':
        return MPCPSRLAgent, {
            'state_dim': get_environment(env_name).state_dim,
            'action_dim': get_environment(env_name).action_dim,
            'hidden_dim': 100,
            'planning_horizon': planning_horizon
        }
    
    elif agent_name == 'KSRL':
        return KSRLAgent, {
            'state_dim': get_environment(env_name).state_dim,
            'action_dim': get_environment(env_name).action_dim,
            'kernel_bandwidth': 1.0,
            'n_samples': 100
        }
    
    else:  # Random
        return RandomAgent, {
            'action_dim': get_environment(env_name).action_dim
        }

def run_mujoco_experiments_only(n_seeds: int = 3):
    """Run experiments ONLY on MuJoCo environments"""
    
    print("\n" + "="*70)
    print("MUJOCO ENVIRONMENTS ONLY - OPTIMIZED MODE")
    print("="*70)
    print(f"Settings:")
    print(f"  - Seeds: {n_seeds}")
    print(f"  - Episodes per env: 40 (reduced from 50)")
    print(f"  - Hidden dim: 150 (reduced from 200)")
    print(f"  - PETS: 4 networks, 15 epochs (down from 5 nets, 20 epochs)")
    print(f"  - LaPSRL: 3000 gradients (down from 5000)")
    print(f"  - Solver timeout: 45s (down from 60s)")
    print(f"\nEstimated time savings: ~30-40% faster than original")
    print("="*70)
    
    all_results = {}
    
    # Methods to run
    methods = ['Convex-PSRL', 'PETS', 'Deep-Ensemble-VI', 'LaPSRL', 'MPC-PSRL', 'KSRL']
    
    for env_name in ['WALKER2D', 'HOPPER', 'HALFCHEETAH']:
        print(f"\nEnvironment: {env_name}")
        print("-" * 70)
        
        config = ENV_CONFIGS[env_name]
        env_results = {}
        
        for method in methods:
            print(f"\nRunning {method}...")
            
            agent_class, agent_params = get_agent_params_optimized(
                env_name, method, config['planning_horizon']
            )
            
            start_time = time.time()
            
            try:
                results = run_multiple_seeds(
                    agent_class=agent_class,
                    agent_params=agent_params,
                    env_name=env_name,
                    n_seeds=n_seeds,
                    n_episodes=config['n_episodes'],
                    max_steps=config['max_steps'],
                    n_action_samples=10,
                    update_freq=5,
                    posterior_sample_freq=10
                )
                
                elapsed_time = time.time() - start_time
                final_reward = results['mean'][-1]
                final_std = results['std'][-1]
                
                print(f"  Final reward: {final_reward:.2f} ± {final_std:.2f}")
                print(f"  Wall-clock time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
                
                env_results[method] = results
                env_results[method]['wall_clock_time'] = elapsed_time
                env_results[method]['final_reward'] = final_reward
                env_results[method]['final_std'] = final_std
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                env_results[method] = {'error': str(e)}
        
        all_results[env_name] = env_results
        
        # SAVE CHECKPOINT after each environment!
        checkpoint_file = Path('results') / f'checkpoint_{env_name.lower()}.pkl'
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({env_name: env_results}, f)
        print(f"\n✅ Checkpoint saved to {checkpoint_file}")
    
    return all_results

def merge_with_recovered_results(mujoco_results: dict):
    """Merge new MuJoCo results with recovered classic control results"""
    
    print("\n" + "="*70)
    print("MERGING RESULTS")
    print("="*70)
    
    # Load recovered results
    recovered_file = Path('results/section_4.2_partial_recovered.pkl')
    if recovered_file.exists():
        with open(recovered_file, 'rb') as f:
            classic_results = pickle.load(f)
        print(f"✅ Loaded recovered results: {list(classic_results.keys())}")
    else:
        print("⚠️  No recovered results found, using MuJoCo only")
        classic_results = {}
    
    # Merge
    complete_results = {**classic_results, **mujoco_results}
    
    # Save complete results
    complete_file = Path('results/section_4.2_sample_efficiency_COMPLETE.pkl')
    with open(complete_file, 'wb') as f:
        pickle.dump(complete_results, f)
    
    print(f"✅ Complete results saved to {complete_file}")
    print(f"   Environments: {list(complete_results.keys())}")
    
    return complete_results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=3,
                       help='Number of random seeds (default: 3)')
    parser.add_argument('--skip-merge', action='store_true',
                       help='Skip merging with recovered results')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CONTINUING EXPERIMENTS - MUJOCO ONLY")
    print("="*70)
    print(f"\nThis will run experiments on:")
    print("  1. Walker2d")
    print("  2. Hopper")
    print("  3. HalfCheetah")
    print(f"\nWith {args.seeds} seeds per environment")
    print(f"Estimated time: {args.seeds * 3 * 6 * 15/60:.1f} hours (conservative estimate)")
    print("\nOptimizations applied:")
    print("  - Reduced hidden dim: 200 → 150")
    print("  - Reduced episodes: 50 → 40")
    print("  - Reduced planning horizon: 50 → 40")
    print("  - Reduced PETS: 5 nets/20 epochs → 4 nets/15 epochs")
    print("  - Reduced LaPSRL: 5000 → 3000 gradients")
    print("  - Reduced solver timeout: 60s → 45s")
    print("  - Relaxed tolerance: 1e-6 → 1e-5")
    print("\nCheckpoints will be saved after EACH environment!")
    
    response = input("\nContinue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # Run experiments
    start_time = time.time()
    mujoco_results = run_mujoco_experiments_only(n_seeds=args.seeds)
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("MUJOCO EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/3600:.1f} hours")
    
    # Merge with recovered results
    if not args.skip_merge:
        complete_results = merge_with_recovered_results(mujoco_results)
        print("\n✅ You can now generate the COMPLETE Figure 3 with all 6 environments!")
        print("   Run: python scripts/generate_figure3.py --results results/section_4.2_sample_efficiency_COMPLETE.pkl")
    else:
        print("\n✅ MuJoCo results saved to checkpoints")
        print("   To merge later: python -c \"from continue_mujoco_only import merge_with_recovered_results; merge_with_recovered_results({})\"")
