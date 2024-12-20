import time
import math
import datetime
import torch                       
from dqn_agent import Agent
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

def dqn(env,agent,brain_name,n_episodes=2000, eps_start=.99, eps_end=0.01, eps_decay = .996, train_numb = 0):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    start_time = time.time()                               # start time 
    scores = []                                            # list of scores for each episode
    scores_window = deque(maxlen=100)                      # last 100 scores
    eps = eps_start                                        # initialize epsilon

    for i_episode in range(1, n_episodes+1):               # loop by episodes
        env_info = env.reset(train_mode = True)[brain_name]
        state = env_info.vector_observations[0]            # get the current state 
        score = 0                                          # reset the score counter
        done = False                                       # are we done yet?
        while not done:                                    # internal loop in the episode
            action = agent.act(state,eps)                  # next action from the agent 
            action = int(action)                           # cast to int
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # done is true if episode has finished
            agent.step(state,action,reward,next_state, done) # next learning step by state and reward
            score += reward                                # count our rewards
            state = next_state                             # update state
            if done:                                       # done ?
                break                                      # save score
        scores_window.append(score)                        # save score in the deque with 100 or less elements
        scores.append(score)
        
        eps = max(eps_end,eps_decay*eps)                   # make epsilon a bit smaller
        
        count = 0                                          # how many times we've reached 13
        for j in range(len(scores_window)):                
            if scores_window[j] >= 13:
                count+=1
                
        elapsed = datetime.timedelta(seconds = time.time()-start_time)  # elapsed time
        
        print('\rEpisode: {}, elapsed: {}, Avg.Score: {:.2f},  score {}, How many scores >= 13: {}, eps.: {:.2f}'. \
            format(i_episode, elapsed, np.mean(scores_window), score, count, eps), end="")
        
        if np.mean(scores_window) >=13:  # check completion criteria.
            print("\n terminating at episode :", i_episode, "ave reward reached +13 over 100 episodes")
            break
            
    torch.save(agent.qnetwork_local.state_dict(), 'weights_'+str(train_numb)+'.trn') # save the weights into the file 
    return scores, i_episode

