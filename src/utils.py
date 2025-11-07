"""
Utility functions for experiments, evaluation, and data management.
"""

import numpy as np
import pickle
from typing import List, Dict, Any, Tuple
from pathlib import Path


def run_episode(agent, env, max_steps: int = 200, 
                n_action_samples: int = 10, update_freq: int = 1) -> Dict[str, Any]:
    """
    Run a single episode with an agent in an environment.
    
    Args:
        agent: RL agent with add_experience, update_models, plan_action methods
        env: Environment with reset, step methods
        max_steps: Maximum steps per episode
        n_action_samples: Number of action candidates for planning
        update_freq: How often to update models (in steps)
        
    Returns:
        Dictionary with episode statistics
    """
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < max_steps:
        # Get action samples for planning
        action_samples = env.get_action_samples(n_action_samples)
        
        # Plan action
        action = agent.plan_action(state, action_samples)
        
        # Take step in environment
        next_state, reward, done, _ = env.step(action)
        
        # Add experience to agent
        agent.add_experience(state, action, next_state, reward)
        
        # Update models periodically
        if steps % update_freq == 0:
            agent.update_models()
        
        total_reward += reward
        state = next_state
        steps += 1
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'success': done
    }


def evaluate_agent(agent, env, n_episodes: int = 10, max_steps: int = 200,
                   n_action_samples: int = 10) -> Dict[str, float]:
    """
    Evaluate an agent over multiple episodes.
    
    Args:
        agent: RL agent
        env: Environment
        n_episodes: Number of evaluation episodes
        max_steps: Max steps per episode
        n_action_samples: Action samples for planning
        
    Returns:
        Dictionary with mean and std of rewards
    """
    rewards = []
    
    for _ in range(n_episodes):
        result = run_episode(agent, env, max_steps, n_action_samples, update_freq=1000)
        rewards.append(result['total_reward'])
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards)
    }


def train_agent(agent, env, n_episodes: int = 100, max_steps: int = 200,
                n_action_samples: int = 10, update_freq: int = 1,
                posterior_sample_freq: int = 10, verbose: bool = True) -> Dict[str, List]:
    """
    Train an agent and collect learning curve data.
    
    Args:
        agent: RL agent
        env: Environment
        n_episodes: Number of training episodes
        max_steps: Max steps per episode
        n_action_samples: Action samples for planning
        update_freq: Model update frequency
        posterior_sample_freq: How often to resample posterior
        verbose: Whether to print progress
        
    Returns:
        Dictionary with episode rewards and other metrics
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        # Resample from posterior (Thompson Sampling)
        if episode % posterior_sample_freq == 0 and episode > 0:
            if hasattr(agent, 'sample_posterior'):
                agent.sample_posterior()
        
        # Run episode
        result = run_episode(agent, env, max_steps, n_action_samples, update_freq)
        
        episode_rewards.append(result['total_reward'])
        episode_lengths.append(result['steps'])
        
        if verbose and (episode + 1) % 10 == 0:
            recent_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{n_episodes}, "
                  f"Recent Avg Reward: {recent_reward:.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def run_multiple_seeds(agent_class, agent_params: Dict, env_name: str,
                       n_seeds: int = 5, n_episodes: int = 100,
                       **train_kwargs) -> Dict[str, np.ndarray]:
    """
    Run experiments with multiple random seeds.
    
    Args:
        agent_class: Agent class to instantiate
        agent_params: Parameters for agent initialization
        env_name: Name of environment
        n_seeds: Number of random seeds
        n_episodes: Episodes per seed
        **train_kwargs: Additional arguments for train_agent
        
    Returns:
        Dictionary with results across seeds
    """
    import environments
    
    all_rewards = []
    
    for seed in range(n_seeds):
        # Set random seed
        np.random.seed(seed)
        
        # Create agent and environment
        agent = agent_class(**agent_params)
        env = environments.get_environment(env_name)
        
        # Train agent
        results = train_agent(agent, env, n_episodes=n_episodes, 
                            verbose=False, **train_kwargs)
        
        all_rewards.append(results['episode_rewards'])
        
        env.close()
    
    # Stack results
    rewards_array = np.array(all_rewards)  # (n_seeds, n_episodes)
    
    return {
        'rewards': rewards_array,
        'mean': np.mean(rewards_array, axis=0),
        'std': np.std(rewards_array, axis=0),
        'median': np.median(rewards_array, axis=0)
    }


def save_results(results: Dict, filepath: str):
    """
    Save experimental results to file.
    
    Args:
        results: Dictionary of results
        filepath: Path to save file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """
    Load experimental results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Dictionary of results
    """
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    return results


def compute_regret(rewards: np.ndarray, optimal_reward: float) -> np.ndarray:
    """
    Compute cumulative regret.
    
    Args:
        rewards: Array of episode rewards
        optimal_reward: Optimal reward per episode
        
    Returns:
        Cumulative regret array
    """
    regret = optimal_reward - rewards
    cumulative_regret = np.cumsum(regret)
    return cumulative_regret


def smooth_curve(values: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Smooth a curve using moving average.
    
    Args:
        values: Array of values
        window: Window size for smoothing
        
    Returns:
        Smoothed array
    """
    if len(values) < window:
        return values
    
    smoothed = np.convolve(values, np.ones(window) / window, mode='valid')
    # Pad to maintain original length
    pad_size = len(values) - len(smoothed)
    if pad_size > 0:
        smoothed = np.concatenate([values[:pad_size], smoothed])
    
    return smoothed
