import torch
from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import Agent
from utils import read_session_info
try:
    # 加载 Unity 环境
    env = UnityEnvironment(file_name="D:\Code\dl_navigation\Banana_Windows_x86_64\Banana_Windows_x86_64\Banana.exe",base_port=65454)

    # 重置环境
    env.reset()
    print("环境加载成功！")

    # 获取智能体的行为名称
    behavior_names = env.brain_names
    print(f"行为名称: {behavior_names}")

    # 与环境交互
    brain_name = behavior_names[0]
    brain = env.brains[brain_name]
    print(f"智能体的观察空间大小: {brain.vector_observation_space_size}")
    print(f"智能体的动作空间大小: {brain.vector_action_space_size}")


except Exception as e:
    print(f"加载环境时出错: {e}")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


env_info = env.reset(train_mode=True)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
   action = np.random.randint(action_size)        # select an action
   env_info = env.step(action)[brain_name]        # send the action to the environment
   next_state = env_info.vector_observations[0]   # get the next state
   reward = env_info.rewards[0]                   # get the reward
   done = env_info.local_done[0]                  # see if episode has finished
   score += reward                                # update the score
   state = next_state                             # roll over the state to next time step
   if done:                                       # exit loop if episode finished
       break
    
print("Score: {}".format(score))

def checkWeights(env, train_n, test, fc1_n, fc2_n, eps_s, episodes):
    agent = Agent(state_size=37, action_size=4, seed=17, fc1_units=fc1_n, fc2_units=fc2_n)  
    file_weights = 'weights_'+str(train_n)+'.trn'
    agent.qnetwork_local.load_state_dict(torch.load(file_weights))

    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state,.05)                  # select an action
        action = int(action)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
    
    print('Train: {}, Test: {}, Episode: {}, fc1_units: {}, fc2_units: {}, eps_start: {}, Score: {}'\
          .format(train_n, test, episodes, fc1_n, fc2_n, eps_s, score))
    return score


# meta_path="./session.txt"

# loaded_data=read_session_info(meta_path)
# list_fc1_units = loaded_data["list_fc1_units"]
# list_fc2_units = loaded_data["list_fc2_units"]
# list_eps_start = loaded_data["list_eps_start"]
# list_episodes = loaded_data["list_episodes"]
# numb_of_trains = loaded_data["numb_of_trains"]

#mannual copy from train output
list_fc1_units = [64]
list_fc2_units = [72]
list_eps_start = [0.989]
list_episodes  = [1]

numb_of_trains = 1 # 10 
for i in range(0, numb_of_trains):
    fc1_nodes = list_fc1_units[i]
    fc2_nodes = list_fc2_units[i]
    eps_start = list_eps_start[i]
    episodes  = list_episodes[i]
    list_scores = []
    for test in range(0,6):        
        score = checkWeights(env=env, train_n=i, test=test, fc1_n=fc1_nodes, fc2_n=fc2_nodes, eps_s=eps_start,episodes=episodes)
        list_scores.append(score)
    avg_score =  np.mean(list_scores)
    print('       Average Score: ', avg_score)
    print('=========================================================')