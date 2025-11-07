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


def get_environment(env_name: str):
    """
    Factory function to get environment by name.
    
    Args:
        env_name: Name of environment ('cartpole', 'mountaincar', 'pendulum')
        
    Returns:
        Environment instance
    """
    env_map = {
        'cartpole': DiscreteCartPoleWrapper,
        'mountaincar': ContinuousMountainCar,
        'pendulum': Pendulum
    }
    
    if env_name.lower() not in env_map:
        raise ValueError(f"Unknown environment: {env_name}. Choose from {list(env_map.keys())}")
    
    return env_map[env_name.lower()]()
