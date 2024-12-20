from unityagents import UnityEnvironment
import numpy as np
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
env.close()
