import gym
from stable_baselines3 import A2C


env = gym.make('LunarLander-v2', render_mode='human')
env.reset()
model = A2C('MlpPolicy', env, verbose=1)  # type: ignore
model.learn(total_timesteps=100000)

episodes = 10

for episode in range(episodes):
    obs = env.reset()
    terminated = False

    while not terminated:
        env.render()
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample())
        # print(reward)

env.close()
