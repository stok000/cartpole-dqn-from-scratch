import numpy as np
import matplotlib.pyplot as plt
import torch
from src.environment import CartPoleEnv
from src.dqn_agent import DQNAgent
from config import Config

def train():
    env = CartPoleEnv()
    agent = DQNAgent(
        Config.STATE_SIZE,
        Config.ACTION_SIZE,
        Config.LEARNING_RATE,
        Config.GAMMA,
        Config.EPSILON_START,
        Config.EPSILON_END,
        Config.EPSILON_DECAY
    )

    best_score = 0
    solved_episodes = 0
    scores = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    for episode in range(Config.NUM_EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(Config.MAX_STEPS):
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)

            agent.learn()

            state = next_state
            total_reward += reward

            if step % Config.TARGET_UPDATE == 0:
                agent.update_target_network()

            if done:
                break
        
        scores.append(total_reward)

        if total_reward > best_score:
            best_score = total_reward
        
        if len(scores) >= 100:
            recent_avg = sum(scores[-100:]) / 100
            if recent_avg >= 195:
                solved_episodes += 1
                if solved_episodes >= 10:
                    print(f"CartPole SOLVED at episode {episode}!")
                    print(f"Average score over last 100 episodes: {recent_avg:.2f}")
                    break

        if episode % 10 == 0:
            update_real_time_plot(ax, scores, episode)

        if episode % 100 == 0:
            recent_avg = sum(scores[-100:]) / 100 if len(scores) >= 100 else sum(scores) / len(scores)
            print(f"Episode {episode}, Score: {total_reward}, Best: {best_score}, Avg(100): {recent_avg:.2f}, Epsilon: {agent.epsilon:.3f}")

    plt.ioff()
    plt.show()

    if Config.SAVE_MODEL:
        torch.save(agent.q_network.state_dict(), 'model/cartpole_model.pth')
        print("Saved as 'model/cartpole_model.pth'")

    return scores

def update_real_time_plot(ax, scores, episode):
    ax.clear()
    
    ax.plot(scores, alpha=0.3, label='Episode Score', color='blue')
    
    window_size = 100
    if len(scores) >= window_size:
        moving_avg = []
        for i in range(window_size - 1, len(scores)):
            avg = sum(scores[i - window_size + 1:i + 1]) / window_size
            moving_avg.append(avg)
        
        x_avg = range(window_size - 1, len(scores))
        ax.plot(x_avg, moving_avg, label=f'Moving Average ({window_size})', color='red', linewidth=2)
    
    ax.axhline(y=195, color='green', linestyle='--', label='Solved Threshold')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title(f'CartPole DQN Training Progress - Episode {episode}')
    ax.legend()
    ax.grid(True)
    
    plt.pause(0.01)

def test_trained_agent():
    env = CartPoleEnv(render_mode="human")
    agent = DQNAgent(
        Config.STATE_SIZE,
        Config.ACTION_SIZE,
        Config.LEARNING_RATE,
        Config.GAMMA,
        Config.EPSILON_START,
        Config.EPSILON_END,
        Config.EPSILON_DECAY
    )
    agent.q_network.load_state_dict(torch.load('model/cartpole_model.pth'))
    agent.epsilon = 0.0

    for episode in range(5):
        state = env.reset()
        total_reward = 0

        for step in range(Config.MAX_STEPS):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        print(f"Test Episode {episode+1}: Score {total_reward}")

if __name__ == "__main__":
    scores = train()
    print("\nTesting trained agent...")
    test_trained_agent()