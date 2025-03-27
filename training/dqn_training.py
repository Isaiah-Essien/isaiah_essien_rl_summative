from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from CustomLogger import EpilepsyLoggerCallback
from environment.custom_env import EpilepsyDetectionEnv

# No modifications to your original env
env = DummyVecEnv([lambda: Monitor(EpilepsyDetectionEnv())])

logger_callback = EpilepsyLoggerCallback(verbose=1)

model_dqn = DQN("MlpPolicy", env, verbose=1,
                learning_rate=0.001,
                gamma=0.95,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=64,
                exploration_fraction=0.1,
                exploration_final_eps=0.02)

model_dqn.learn(total_timesteps=50000, callback=logger_callback)
model_dqn.save("epilepsy_dqn_model")
env.close()
