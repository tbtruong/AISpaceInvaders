import gym
import numpy as np
import tensorflow as tf
from env import *
import random
from collections import deque, Counter
import time
import os
import pandas as pd


model_path = '/usr4/cs440/bvu/SpaceInvaderNew/Models/'

tf.compat.v1.reset_default_graph()
#Reset is technically not necessary if variables done  in TF2
#https://github.com/ageron/tf2_course/issues/8


#Define our model's approach 
    #Epsilon-Greedy Policy, Loss Function, Epsilon etc...
    #Depending on what learning we do, we will need helper functions for that. 
        #Github and Colab one uses Q-learning
 

class DQN:
    def __init__(self, env, single_frame_dim, num_frames_to_stack, old_model_filepath=None):
        self.env = env
        self.memory = deque(maxlen=100000) #5000
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9999997
        self.learning_rate = 0.00003
        self.burnin = 20000 #2000
        self.update_target_step_count = 4000 #3000
        self.num_steps_since_last_update = 0
        self.single_frame_dim = single_frame_dim
        self.num_frames_to_stack = num_frames_to_stack
       
        self.model = self.create_model()
        self.target_model = self.create_model()

        if(old_model_filepath==None):
            self.model = self.create_model()  # Will do the actual predictions
            self.target_model = self.create_model()  # Will compute what action we DESIRE from our model
        else:
            self.model = tf.keras.models.load_model(old_model_filepath)
            self.target_model = tf.keras.models.load_model(old_model_filepath)
        # Otherwise we are changing the goal at each timestep.

    #Defining the model
    def create_model(self):
        
        model = tf.keras.Sequential()
        #Layers (3)
        model.add(tf.keras.layers.Conv2D(32, (8, 8), strides=4, input_shape=(self.single_frame_dim[0], self.single_frame_dim[1], self.num_frames_to_stack), activation="relu"))
        model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu"))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu"))
        #Flatten Image
        model.add(tf.keras.layers.Flatten())
        #Connect Layers
        model.add(tf.keras.layers.Dense(128, activation="relu")) #adds the fully connected layer
        model.add(tf.keras.layers.Dense(env.action_space.n)) #env.action_space.n returns the number of actions in the action space #adds the final output layer
        #Configuration for training/Weights (Loss and Optimizzer)
        model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        # model.summary()
        return model

    def update_target_model(self, current_episode, current_step, update_by_episode_count):

        self.num_steps_since_last_update +=1 #just did a step, update the counter
        #update if new episode and parameter set or by episode count
        if (update_by_episode_count and current_step == 0) or (not update_by_episode_count and (self.num_steps_since_last_update == self.update_target_step_count)): # for smaller problems you might want to update just by episode. For larger problems, usually cap it at 5k or max for episode or whatever heuristic you prefer.
            print("Updating_target_model at episode/step: " + str(current_episode) + " / " +str(current_step))
            self.target_model.set_weights(self.model.get_weights())
            self.num_steps_since_last_update = 0

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, batch_size=64):
        if len(self.memory) < self.burnin:
            return
        samples = random.sample(self.memory, batch_size)
        all_states = np.reshape([np.squeeze(x[0]) for x in samples], (batch_size, self.single_frame_dim[0],  self.single_frame_dim[1], num_frames_to_stack))
        all_actions = np.reshape([x[1] for x in samples], (batch_size,))
        all_rewards = np.reshape([x[2] for x in samples], (batch_size,))
        all_new_states = np.reshape([np.squeeze(x[3]) for x in samples], (batch_size, self.single_frame_dim[0], self.single_frame_dim[1], num_frames_to_stack))
        all_dones = np.reshape([x[4] for x in samples], (batch_size,))

        all_targets = np.array(self.model.predict_on_batch(all_states.astype('float16')))  # this is what we will update
        Q_0 = np.array(self.model.predict_on_batch(all_new_states.astype('float16')))  # This is what we use to find what max action we should take
        Q_target = np.array(self.target_model.predict_on_batch(all_new_states.astype('float16')))  # This is the values we will combine with max action to update the target
        max_actions = np.argmax(Q_0, axis=1)  # This is the index we will use to take from Q_target
        max_Q_target_values = Q_target[np.arange(len(Q_target)), np.array(max_actions)]  # The target will be updated with this.
        all_targets[np.arange(len(all_targets)), np.array(all_actions)] = all_rewards + self.gamma * max_Q_target_values * (~all_dones)  # Actually due the update

        self.model.train_on_batch(all_states.astype('float16'), all_targets)  # reweight network to get new targets    

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            reshaped_state = np.expand_dims(state, axis=0).astype('float16')
            return np.argmax(self.model.predict(reshaped_state)[0]) 

    def save_model(self, fn):
        self.model.save(fn)


#######################################################################################
# Initialize environment and parameters
#######################################################################################
env = gym.make('SpaceInvaders-v0')
raw_image_dim = preprocess_image(env.reset()).shape
num_episodes = 3000
num_frames_to_collapse = 3
num_frames_to_stack = 3
my_agent = DQN(env=env,single_frame_dim=raw_image_dim,num_frames_to_stack=num_frames_to_stack)
 
totalreward = []
steps = []


for episode in range(num_episodes):
    print("---------------------------------------------------")
    print("Episode: " + str(episode) + " started")
    time_start = time.time()
    current_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=3)
    cur_state = np.repeat(preprocess_image(env.reset())[:, :, np.newaxis], stack_size, axis=2)
    episode_reward = 0
    step = 0
    done = False
    

    while not done:
        if(step % 100 == 0):
            print("At step: " + str(step))    
                    
        action = my_agent.act(cur_state)
        my_agent.update_target_model(episode, step, False)  # self.update_target_step_count steps for every episode. Here every episode    

        new_state, reward, done, info = env.step(action)
        updated_frames, cur_state = update_current_frames(current_frames, new_state, done) #the 4 original stackked frames as well if 
        step += 1

        #Add to memory
        my_agent.remember(current_frames, action, reward, updated_frames, done)

        #Fit the model
        my_agent.replay()

        #Set the current_state to the new state
        current_frames = updated_frames
        
        episode_reward += reward

        if info['ale.lives'] == 0:
            print("Breaking due to death!")
            break
        # if episode_reward == 1000:
        #     print("Breaking due to goal reached")
        #     break
        if step > 3000:
            print("Breaking due to out of steps.")
            break
        
    totalreward.append(episode_reward)
    steps.append(step)
    if episode_reward >= 1000:
        print("Episode: " + str(episode) + " -- SUCCESS -- with a total reward of: " + str(episode_reward))
        my_agent.save_model(model_path + "episode-{}_model_success.h5".format(episode))
    else:
        print("Episode: " + str(episode) + " -- FAILURE -- with a total reward of: " + str(episode_reward))
        if episode % 10 == 0:
            my_agent.save_model(model_path + "episode-{}_model_failure.h5".format(episode))

    time_end = time.time()
    tf.keras.backend.clear_session()

    print("Episode: " + str(int(episode)) + " completed in steps/time/avg_running_reward: " + str(steps[-1]) + " / " + str(int(time_end - time_start)) + " / " + str(np.array(totalreward)[-100:].mean()))
    print("This took: " + str(int(time_end - time_start)) + " seconds.")
    print('-----------------------------------------------------------------------------------------------------------')

    if episode % 500 == 0:
        results_df = pd.DataFrame(totalreward, columns=['episode_reward'])
        results_df['steps_taken'] = steps
        results_df['average_running_reward'] = results_df['episode_reward'].rolling(window=10).mean()
        results_df.to_csv(model_path + "training_results" + str(episode) + ".csv")

env.close()

# results_df = pd.DataFrame(totalreward, columns=['episode_reward'])
# results_df['steps_taken'] = steps
# results_df['average_running_reward'] = results_df['episode_reward'].rolling(window=100).mean()

# results_df.to_csv(model_path + "training_results.csv")

 