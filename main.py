import gym
from stable_baselines3 import PPO
import os
import time

models_dir = "models"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = gym.make('LunarLander-v2')
env.reset()

MODEL_NAME = "PPO4"
model = PPO('MlpPolicy', env, verbose=1,  # type: ignore
            tensorboard_log=log_dir)

TIMESTEPS = 100_000
EPISODES = 10

for i in range(1, EPISODES):
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                tb_log_name=f"{MODEL_NAME}-{int(time.time())}")
    model.save(f"{models_dir}/{MODEL_NAME}-{TIMESTEPS*i}-{int(time.time())}")


# episodes = 10
# for episode in range(episodes):
#     obs = env.reset()
#     terminated = False

#     while not terminated:
#         env.render()
#         observation, reward, terminated, truncated, info = env.step(
#             env.action_space.sample())
#         # print(reward)

env.close()
