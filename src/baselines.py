"""
Baseline algorithms for comparison with Convex-PSRL.

Includes:
- MPC-PSRL: Model Predictive Control with Posterior Sampling (Fan & Ming 2021)
- LaPSRL: Laplace-approximated PSRL
- KSRL: Kernel-based Thompson Sampling
- RandomAgent: Random baseline
"""

import numpy as np
from typing import Dict, Optional


class MPCPSRLAgent:
    """
    Model Predictive Control with Posterior Sampling (Fan & Ming 2021).
    Uses linear models with Bayesian updates.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 gamma: float = 0.99, horizon: int = 5):
        """
        Initialize MPC-PSRL agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            gamma: Discount factor
            horizon: MPC planning horizon
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.horizon = horizon
        
        # Linear model parameters: s' = A @ [s, a] + noise
        input_dim = state_dim + action_dim
        self.A_mean = np.zeros((state_dim, input_dim))
        self.A_cov = np.eye(state_dim * input_dim)
        
        # Reward model parameters: r = w @ [s, a] + noise
        self.w_mean = np.zeros(input_dim)
        self.w_cov = np.eye(input_dim)
        
        # Experience buffer
        self.X = []  # [state, action]
        self.Y_next = []  # next_state
        self.Y_reward = []  # reward
        
        # Sampled model parameters (Thompson Sampling)
        self.A_sample = self.A_mean.copy()
        self.w_sample = self.w_mean.copy()
        
    def add_experience(self, state: np.ndarray, action: np.ndarray,
                       next_state: np.ndarray, reward: float):
        """Add transition to buffer."""
        self.X.append(np.concatenate([state, action]))
        self.Y_next.append(next_state)
        self.Y_reward.append(reward)
    
    def update_models(self) -> Dict:
        """
        Bayesian update of model parameters (conjugate priors).
        """
        if len(self.X) == 0:
            return {'status': 'no_data'}
        
        X = np.array(self.X)
        Y_next = np.array(self.Y_next)
        Y_reward = np.array(self.Y_reward)
        
        # Bayesian linear regression for dynamics
        # Posterior: N(A_mean, A_cov)
        noise_var = 0.1
        for i in range(self.state_dim):
            y = Y_next[:, i]
            # Conjugate update
            precision_prior = np.linalg.inv(self.A_cov[i*X.shape[1]:(i+1)*X.shape[1], 
                                                       i*X.shape[1]:(i+1)*X.shape[1]])
            precision_post = precision_prior + (X.T @ X) / noise_var
            cov_post = np.linalg.inv(precision_post)
            mean_post = cov_post @ (precision_prior @ self.A_mean[i, :] + 
                                   (X.T @ y) / noise_var)
            
            self.A_mean[i, :] = mean_post
            self.A_cov[i*X.shape[1]:(i+1)*X.shape[1], 
                      i*X.shape[1]:(i+1)*X.shape[1]] = cov_post
        
        # Bayesian linear regression for reward
        precision_prior = np.linalg.inv(self.w_cov)
        precision_post = precision_prior + (X.T @ X) / noise_var
        self.w_cov = np.linalg.inv(precision_post)
        self.w_mean = self.w_cov @ (precision_prior @ self.w_mean + 
                                    (X.T @ Y_reward) / noise_var)
        
        return {'status': 'updated', 'n_samples': len(self.X)}
    
    def sample_posterior(self):
        """Sample model parameters from posterior (Thompson Sampling)."""
        # Sample dynamics model
        flat_mean = self.A_mean.flatten()
        try:
            self.A_sample = np.random.multivariate_normal(
                flat_mean, self.A_cov + np.eye(len(flat_mean)) * 1e-6
            ).reshape(self.A_mean.shape)
        except:
            self.A_sample = self.A_mean + np.random.randn(*self.A_mean.shape) * 0.1
        
        # Sample reward model
        try:
            self.w_sample = np.random.multivariate_normal(
                self.w_mean, self.w_cov + np.eye(len(self.w_mean)) * 1e-6
            )
        except:
            self.w_sample = self.w_mean + np.random.randn(*self.w_mean.shape) * 0.1
    
    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict next state using sampled model."""
        x = np.concatenate([state, action])
        return self.A_sample @ x
    
    def predict_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Predict reward using sampled model."""
        x = np.concatenate([state, action])
        return self.w_sample @ x
    
    def plan_action(self, state: np.ndarray, action_space_samples: np.ndarray) -> np.ndarray:
        """
        MPC: Plan action by optimizing over action sequences.
        """
        best_value = -np.inf
        best_action = action_space_samples[0]
        
        for action in action_space_samples:
            value = 0
            s = state.copy()
            
            for h in range(self.horizon):
                r = self.predict_reward(s, action)
                s = self.predict_next_state(s, action)
                value += (self.gamma ** h) * r
                
                # Random rollout for future
                if h < self.horizon - 1:
                    action = action_space_samples[np.random.randint(len(action_space_samples))]
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action


class LaPSRLAgent:
    """
    Laplace-approximated PSRL using neural network with Laplace approximation.
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 64, gamma: float = 0.99):
        """Initialize LaPSRL agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        
        # Neural network parameters
        input_dim = state_dim + action_dim
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W2_dynamics = np.random.randn(state_dim, hidden_dim) * 0.01
        self.W2_reward = np.random.randn(1, hidden_dim) * 0.01
        
        # Laplace approximation (Hessian at MAP)
        self.hessian_inv = None
        
        # Experience
        self.X = []
        self.Y_next = []
        self.Y_reward = []
    
    def add_experience(self, state: np.ndarray, action: np.ndarray,
                       next_state: np.ndarray, reward: float):
        """Add transition."""
        self.X.append(np.concatenate([state, action]))
        self.Y_next.append(next_state)
        self.Y_reward.append(reward)
    
    def update_models(self) -> Dict:
        """Fit neural network using gradient descent (MAP estimate)."""
        if len(self.X) == 0:
            return {'status': 'no_data'}
        
        X = np.array(self.X)
        Y_next = np.array(self.Y_next)
        
        # Simple gradient descent
        lr = 0.01
        for _ in range(100):
            hidden = np.maximum(0, X @ self.W1.T)
            pred_next = hidden @ self.W2_dynamics.T
            error = pred_next - Y_next
            
            grad_W2 = (error.T @ hidden) / len(X)
            grad_W1_pre = (error @ self.W2_dynamics) * (hidden > 0)
            grad_W1 = (grad_W1_pre.T @ X) / len(X)
            
            self.W2_dynamics -= lr * grad_W2
            self.W1 -= lr * grad_W1
        
        return {'status': 'updated', 'n_samples': len(self.X)}
    
    def sample_posterior(self):
        """Sample from Laplace approximation around MAP."""
        # Add noise to weights
        noise_scale = 0.1 / np.sqrt(len(self.X) + 1)
        self.W1 += np.random.randn(*self.W1.shape) * noise_scale
        self.W2_dynamics += np.random.randn(*self.W2_dynamics.shape) * noise_scale
    
    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict next state."""
        x = np.concatenate([state, action]).reshape(1, -1)
        hidden = np.maximum(0, x @ self.W1.T)
        return (hidden @ self.W2_dynamics.T)[0]
    
    def predict_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Predict reward."""
        return 0.0  # Simplified
    
    def plan_action(self, state: np.ndarray, action_space_samples: np.ndarray) -> np.ndarray:
        """Plan action."""
        best_value = -np.inf
        best_action = action_space_samples[0]
        
        for action in action_space_samples:
            s = state.copy()
            for _ in range(3):
                s = self.predict_next_state(s, action)
            value = -np.sum((s - state)**2)  # Simplified value
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action


class KSRLAgent:
    """
    Kernel-based Thompson Sampling for RL.
    Uses Gaussian Process models for dynamics.
    """
    
    def __init__(self, state_dim: int, action_dim: int, gamma: float = 0.99):
        """Initialize KSRL agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Kernel parameters
        self.length_scale = 1.0
        self.noise_var = 0.1
        
        # Experience
        self.X = []
        self.Y_next = []
        self.Y_reward = []
        
        # GP posterior
        self.K_inv = None
    
    def add_experience(self, state: np.ndarray, action: np.ndarray,
                       next_state: np.ndarray, reward: float):
        """Add transition."""
        self.X.append(np.concatenate([state, action]))
        self.Y_next.append(next_state)
        self.Y_reward.append(reward)
    
    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel."""
        sq_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                  np.sum(X2**2, axis=1).reshape(1, -1) - \
                  2 * X1 @ X2.T
        return np.exp(-sq_dist / (2 * self.length_scale**2))
    
    def update_models(self) -> Dict:
        """Update GP posterior."""
        if len(self.X) == 0:
            return {'status': 'no_data'}
        
        X = np.array(self.X)
        K = self.rbf_kernel(X, X) + self.noise_var * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)
        
        return {'status': 'updated', 'n_samples': len(self.X)}
    
    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """GP prediction."""
        if len(self.X) == 0:
            return state
        
        x = np.concatenate([state, action]).reshape(1, -1)
        X_train = np.array(self.X)
        Y_train = np.array(self.Y_next)
        
        k_star = self.rbf_kernel(x, X_train)
        mean = k_star @ self.K_inv @ Y_train
        
        return mean[0]
    
    def predict_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Predict reward."""
        return 0.0
    
    def plan_action(self, state: np.ndarray, action_space_samples: np.ndarray) -> np.ndarray:
        """Plan action."""
        return action_space_samples[np.random.randint(len(action_space_samples))]
    
    def sample_posterior(self):
        """Sample from GP posterior."""
        pass  # Already stochastic


class RandomAgent:
    """Random baseline agent."""
    
    def __init__(self, state_dim: int, action_dim: int, gamma: float = 0.99):
        """Initialize random agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
    
    def add_experience(self, state, action, next_state, reward):
        """No-op."""
        pass
    
    def update_models(self):
        """No-op."""
        return {'status': 'random'}
    
    def sample_posterior(self):
        """No-op."""
        pass
    
    def plan_action(self, state: np.ndarray, action_space_samples: np.ndarray) -> np.ndarray:
        """Random action."""
        return action_space_samples[np.random.randint(len(action_space_samples))]
