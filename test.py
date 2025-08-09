import os
import torch
import numpy as np
from src.environment import CartPoleEnv
from src.dqn_agent import DQNAgent
from config import Config

def test_agent(model_path, num_episodes=5, render=True):
    env = CartPoleEnv(render_mode="human" if render else None)

    agent = DQNAgent(
        Config.STATE_SIZE,
        Config.ACTION_SIZE,
        Config.LEARNING_RATE,
        Config.GAMMA,
        0.0,
        0.0,
        1.0
    )

    agent.q_network.load_state_dict(torch.load(model_path))
    agent.epsilon = 0.0

    scores = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        for step in range(Config.MAX_STEPS):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                break
        
        scores.append(total_reward)
        print(f"Episode {episode + 1}: Score {total_reward}, Steps {steps}")
    
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)

    print(f"\n Test Results:")
    print(f"Average Score: {avg_score:.1f}")
    print(f"Best Score: {max_score}")
    print(f"Worst Score: {min_score}")
    
    return scores

def main():
    model_path = 'model/cartpole_model.pth'

    print("CartPole DQN Tester")
    print("=" * 30)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model from: {model_path}")

    print("Testing with visual rendering...")
    test_agent(model_path, num_episodes=5, render=True)

    print("Testing without rendering...")
    test_agent(model_path, num_episodes=10, render=False)

if __name__ == "__main__":
    main()