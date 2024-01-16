
import gym
import panda_gym
import numpy as np
from stable_baselines3 import DDPG, HerReplayBuffer, SAC, TD3, PPO
from numpngw import write_apng




env = gym.make("PandaPush-v2", render=True)
# model = DDPG("MultiInputPolicy", env, verbose=1, replay_buffer_class=HerReplayBuffer)
# model.learn(total_timesteps=1000000)
# # print("action space:", env.action_space)
# # model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1)
# # model.learn(total_timesteps=100)
# model.save("ddpg_panda_pickanplace")
# model = DDPG.load("ddpg_panda_pick", env=env)

images = [env.render("rgb_array")]
model = SAC.load("SAC_Panda_Push", env=env )
obs = env.reset()
episodes = 0
success = 0
total_reward = 0
count = 0
steps = []
i = 0
while True:
    action = model.predict(obs)
    obs, reward, done, info = env.step(action[0])
    count = count + 1
    images.append(env.render("rgb_array"))
    if reward==0:
        success = success + 1
        episodes = episodes + 1
        steps.append(count)
        count = 0
        obs = env.reset()
    elif done==True:
        count = 0
        episodes = episodes+1
        obs = env.reset()
    if episodes==10:
        break
print("Episodes ",episodes)
print("Success ",success)
print("Rate ",success/episodes)
print("Average steps taken ", sum(steps)/len(steps))
# animation = env.get_animation()
# save_path = "reach.mp4"
env.close()


write_apng("SACPandaPushSphere.png", images, delay=50)  # real-time rendering = 40 ms between frames
