from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from CustomLogger import EpilepsyLoggerCallback
from environment.custom_env import EpilepsyDetectionEnv

env = DummyVecEnv([lambda: Monitor(EpilepsyDetectionEnv())])
logger_callback = EpilepsyLoggerCallback(verbose=1)

model = PPO("MlpPolicy", env, verbose=1,
            learning_rate=0.0003,
            gamma=0.95,
            batch_size=32,
            n_steps=2048)

model.learn(total_timesteps=50000, callback=logger_callback)
model.save("epilepsy_ppo_model")

env.close()
