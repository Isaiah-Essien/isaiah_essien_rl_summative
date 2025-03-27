from stable_baselines3 import DQN, PPO
from gymnasium.wrappers import RecordVideo
from environment.custom_env import EpilepsyDetectionEnv

model_dqn = DQN.load("epilepsy_dqn_model.zip")
model_ppo = PPO.load("epilepsy_ppo_model.zip")

env_dqn = RecordVideo(
    EpilepsyDetectionEnv(render_mode='rgb_array'),
    video_folder='rl_agent_videos_dqn',
    episode_trigger=lambda x: True
)

env_ppo = RecordVideo(
    EpilepsyDetectionEnv(render_mode='rgb_array'),
    video_folder='rl_agent_videos_ppo',
    episode_trigger=lambda x: True
)

# Evaluation function (no change needed)


def evaluate_and_record(model, env, episodes=3, model_name='model'):
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        print(f"{model_name} Episode {episode+1}: Total Reward = {total_reward:.2f}")
    env.close()


# Record videos clearly
evaluate_and_record(model_dqn, env_dqn, model_name='DQN')
evaluate_and_record(model_ppo, env_ppo, model_name='PPO')
