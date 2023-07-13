# import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

models_dir = "models"

env = make_vec_env('LunarLander-v2')
env.reset()

MODEL_NAME = "PPO4-600000-1689278114.zip"
model_path = f"{models_dir}/{MODEL_NAME}"

model = PPO.load(model_path)  # type: ignore


episodes = 10
for episode in range(episodes):
    obs = env.reset()
    terminated = False
    while not terminated:
        env.render("human")
        action, _ = model.predict(obs)  # type: ignore
        obs, reward, terminated, info = env.step(action)

env.close()
