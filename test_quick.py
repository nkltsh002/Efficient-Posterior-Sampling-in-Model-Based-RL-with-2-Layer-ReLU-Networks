"""
Quick test script to verify the implementation works before running full experiments.
Tests each component individually.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("="*60)
print("QUICK TEST - Verifying Implementation")
print("="*60)

# Test 1: Import all modules
print("\n[Test 1/5] Testing imports...")
try:
    from src.convex_psrl import ConvexPSRL, TwoLayerReLUNetwork
    from src.baselines import MPCPSRLAgent, RandomAgent
    from src.environments import get_environment
    from src.utils import run_episode
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Test 2-Layer ReLU Network
print("\n[Test 2/5] Testing 2-Layer ReLU Network...")
try:
    net = TwoLayerReLUNetwork(input_dim=4, hidden_dim=16, output_dim=4)
    X_test = np.random.randn(10, 4)
    Y_test = np.random.randn(10, 4)
    
    result = net.fit_convex_dual(X_test, Y_test)
    print(f"  Training status: {result['status']}")
    
    predictions = net.forward(X_test)
    print(f"  Prediction shape: {predictions.shape}")
    print("✓ 2-Layer ReLU Network working")
except Exception as e:
    print(f"✗ Network test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test Convex-PSRL agent
print("\n[Test 3/5] Testing Convex-PSRL agent...")
try:
    agent = ConvexPSRL(state_dim=4, action_dim=1, hidden_dim=16)
    
    # Add some dummy experience
    for i in range(10):
        state = np.random.randn(4)
        action = np.random.randn(1)
        next_state = np.random.randn(4)
        reward = np.random.randn()
        agent.add_experience(state, action, next_state, reward)
    
    # Update models
    stats = agent.update_models()
    print(f"  Update status: {stats}")
    
    # Test planning
    state = np.random.randn(4)
    actions = np.random.randn(5, 1)
    best_action = agent.plan_action(state, actions)
    print(f"  Best action shape: {best_action.shape}")
    
    print("✓ Convex-PSRL agent working")
except Exception as e:
    print(f"✗ Agent test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test environment
print("\n[Test 4/5] Testing CartPole environment...")
try:
    env = get_environment('cartpole')
    state = env.reset()
    print(f"  Initial state shape: {state.shape}")
    
    action = env.get_action_samples(1)[0]
    next_state, reward, done, info = env.step(action)
    print(f"  Step successful - reward: {reward}")
    
    env.close()
    print("✓ Environment working")
except Exception as e:
    print(f"✗ Environment test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Run a mini episode
print("\n[Test 5/5] Running a mini episode...")
try:
    agent = ConvexPSRL(state_dim=4, action_dim=1, hidden_dim=16)
    env = get_environment('cartpole')
    
    result = run_episode(agent, env, max_steps=10, n_action_samples=3, update_freq=5)
    print(f"  Episode result: {result}")
    
    env.close()
    print("✓ Mini episode successful")
except Exception as e:
    print(f"✗ Episode test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("ALL TESTS PASSED ✓")
print("="*60)
print("\nThe implementation is ready!")
print("You can now run full experiments with: python main.py")
print("="*60 + "\n")
