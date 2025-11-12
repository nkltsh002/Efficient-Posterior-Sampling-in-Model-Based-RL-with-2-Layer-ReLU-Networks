"""
RL environments for experimental evaluation.
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Optional


class DiscreteCartPoleWrapper:
    """
    Wrapper for CartPole with discrete actions converted to continuous for compatibility.
    """
    
    def __init__(self, max_steps: int = 200):
        """Initialize CartPole environment."""
        self.env = gym.make('CartPole-v1')
        self.max_steps = max_steps
        self.state_dim = 4
        self.action_dim = 1  # Treating as continuous in [-1, 1]
        self.current_step = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment."""
        state, _ = self.env.reset()
        self.current_step = 0
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Continuous action in range [-1, 1]
            
        Returns:
            next_state, reward, done, info
        """
        # Convert continuous action to discrete (0 or 1)
        discrete_action = 0 if action[0] < 0 else 1
        
        next_state, reward, terminated, truncated, info = self.env.step(discrete_action)
        done = terminated or truncated
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        return next_state, reward, done, info
    
    def get_action_samples(self, n_samples: int = 10) -> np.ndarray:
        """Get random action samples for planning."""
        return np.random.uniform(-1, 1, size=(n_samples, 1))
    
    def close(self):
        """Close environment."""
        self.env.close()


class ContinuousMountainCar:
    """
    Continuous control version of Mountain Car environment.
    """
    
    def __init__(self, max_steps: int = 500):
        """Initialize Mountain Car environment."""
        self.env = gym.make('MountainCarContinuous-v0')
        self.max_steps = max_steps
        self.state_dim = 2  # position, velocity
        self.action_dim = 1  # force
        self.current_step = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment."""
        state, _ = self.env.reset()
        self.current_step = 0
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step.
        
        Args:
            action: Continuous action (force)
            
        Returns:
            next_state, reward, done, info
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        return next_state, reward, done, info
    
    def get_action_samples(self, n_samples: int = 10) -> np.ndarray:
        """Get random action samples for planning."""
        return np.random.uniform(-1, 1, size=(n_samples, 1))
    
    def close(self):
        """Close environment."""
        self.env.close()


class Pendulum:
    """
    Inverted Pendulum continuous control environment.
    """
    
    def __init__(self, max_steps: int = 200):
        """Initialize Pendulum environment."""
        self.env = gym.make('Pendulum-v1')
        self.max_steps = max_steps
        self.state_dim = 3  # cos(theta), sin(theta), angular_velocity
        self.action_dim = 1  # torque
        self.current_step = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment."""
        state, _ = self.env.reset()
        self.current_step = 0
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step.
        
        Args:
            action: Continuous action (torque)
            
        Returns:
            next_state, reward, done, info
        """
        # Clip action to valid range
        action = np.clip(action, -2.0, 2.0)
        
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        return next_state, reward, done, info
    
    def get_action_samples(self, n_samples: int = 10) -> np.ndarray:
        """Get random action samples for planning."""
        return np.random.uniform(-2, 2, size=(n_samples, 1))
    
    def close(self):
        """Close environment."""
        self.env.close()


class NoisyMuJoCoWrapper:
    """
    Wrapper for MuJoCo environments with Gaussian process noise on state transitions.
    Adds σ=0.1 Gaussian noise to state transitions as per Section 4.1.1
    """
    
    def __init__(self, env_id: str, max_steps: int = 1000, noise_std: float = 0.1):
        """
        Initialize MuJoCo environment with noise.
        
        Args:
            env_id: Gymnasium environment ID (e.g., 'Walker2d-v4')
            max_steps: Maximum steps per episode
            noise_std: Standard deviation of Gaussian process noise
        """
        self.env = gym.make(env_id)
        self.max_steps = max_steps
        self.noise_std = noise_std
        self.current_step = 0
        
        # Get dimensions from environment
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Action bounds
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        
    def reset(self) -> np.ndarray:
        """Reset environment."""
        state, _ = self.env.reset()
        self.current_step = 0
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step with Gaussian process noise on state transitions.
        
        Args:
            action: Continuous action
            
        Returns:
            next_state, reward, done, info
        """
        # Clip action to valid range
        action = np.clip(action, self.action_low, self.action_high)
        
        # Take step in environment
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        # Add Gaussian process noise to state transition (σ=0.1)
        noise = np.random.normal(0, self.noise_std, size=next_state.shape)
        next_state = next_state + noise
        
        done = terminated or truncated
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        return next_state, reward, done, info
    
    def get_action_samples(self, n_samples: int = 10) -> np.ndarray:
        """Get random action samples for planning."""
        return np.random.uniform(
            self.action_low, 
            self.action_high, 
            size=(n_samples, self.action_dim)
        )
    
    def close(self):
        """Close environment."""
        self.env.close()


class Walker2d(NoisyMuJoCoWrapper):
    """Walker2d environment with Gaussian process noise."""
    
    def __init__(self, max_steps: int = 1000, noise_std: float = 0.1):
        super().__init__('Walker2d-v4', max_steps, noise_std)


class Hopper(NoisyMuJoCoWrapper):
    """Hopper environment with Gaussian process noise."""
    
    def __init__(self, max_steps: int = 1000, noise_std: float = 0.1):
        super().__init__('Hopper-v4', max_steps, noise_std)


class HalfCheetah(NoisyMuJoCoWrapper):
    """HalfCheetah environment with Gaussian process noise."""
    
    def __init__(self, max_steps: int = 1000, noise_std: float = 0.1):
        super().__init__('HalfCheetah-v4', max_steps, noise_std)


def get_environment(env_name: str, add_noise: bool = True):
    """
    Factory function to get environment by name.
    
    Args:
        env_name: Name of environment ('cartpole', 'mountaincar', 'pendulum', 
                  'walker2d', 'hopper', 'halfcheetah')
        add_noise: Whether to add Gaussian process noise (for MuJoCo envs)
        
    Returns:
        Environment instance
    """
    env_map = {
        'cartpole': DiscreteCartPoleWrapper,
        'mountaincar': ContinuousMountainCar,
        'pendulum': Pendulum,
        'walker2d': Walker2d,
        'hopper': Hopper,
        'halfcheetah': HalfCheetah
    }
    
    if env_name.lower() not in env_map:
        raise ValueError(f"Unknown environment: {env_name}. Choose from {list(env_map.keys())}")
    
    # For MuJoCo envs, pass noise parameter
    if env_name.lower() in ['walker2d', 'hopper', 'halfcheetah']:
        return env_map[env_name.lower()](noise_std=0.1 if add_noise else 0.0)
    else:
        return env_map[env_name.lower()]()
