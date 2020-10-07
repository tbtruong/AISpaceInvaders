import gym
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
from gym import wrappers
from time import time

from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque, Counter

env = gym.make('SpaceInvaders-v0')
# env = wrappers.Monitor(env, './videos/' + str(time()) + '/')

## Finding out what's in the action_space
# print(env.unwrapped.get_action_meanings())
#env.action_space == ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

## Sample render using random actions
# env.reset()
# for _ in range (1000):
#     env.render()
#     env.step(env.action_space.sample())
# env.close()

# for i in range(20):
#     observation = env.reset()
#     print(len(observation[0])) #Obs size 210x160x3
#     for k in range(1000):
#         env.render()
#         # print(observation)
#         action = env.action_space.sample()

#         observation, reward, done, info = env.step(action)
#         # if reward != 0.0:
#         #     print(reward)
#         #     img = Image.fromarray(observation)
#         #     img.save('obs.png')

# observation = env.reset()

#Alternative preprocessing from colab

def preprocess_image(img):
    # Convert the image to greyscale; crop and resize the image to 84x84
    processed_observe = np.uint8(
        resize(rgb2gray(img), (84 , 84), mode='constant') * 255) #mode = Points outside the boundaries of the input are filled according to the given mode where constant --> pads with constant values
    return processed_observe


observation = env.reset()
# og_image = Image.fromarray(observation)
# og_image.save('og_image.png')

# img = preprocess_image(observation)
# image = Image.fromarray(img)
# image.convert('RGB').save('preprocess1.png')

# state_stack = 4
# long_state = np.repeat(preprocess_image(env.reset())[:, :, np.newaxis], state_stack, axis=2)
# # print(long_state)
# image = Image.fromarray(long_state)
# image.save('stacked.png')

#Stack Images
stack_size = 3  # Stacking 4 frames
# Initialize deque with zero-images one array for each image. Deque is a special kind of queue that deletes last entry when new entry comes in
current_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=3)

def update_current_frames(current_frames, observe, is_new_game):
    #If we are just beginning the episode, there is no previous action. Stack the current state 4 times to avoid sub-optimal, and make this the current history (a list of states). 
    #As we progress through the episode drop the first element and append at the last position to keep history continual 
    state = preprocess_image(observe)
    counter = 0

     #stacked frames == current_frames
     #stacked state == 

    if (is_new_game):
        #WIPE CURRENT_FRAMES
       
        updated_current_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=3) #makes deque object
        while counter <= stack_size:
            updated_current_frames.append(state)
            counter += 1 


        #Create new current_frames where state = pre_processing(observe)
        stacked_state = np.stack(updated_current_frames, axis=2)  #makes the stacked state where this is the new environment
        #stacked_state = np.reshape(stacked_state, (1, 84, 84, 4))
    
    else:
        updated_current_frames = current_frames
        updated_current_frames.popleft()
        updated_current_frames.append(state)
        stacked_state = np.stack(updated_current_frames, axis=2)
        #stacked_state = np.reshape(stacked_state, (1, 84, 84, 4))
    
    return updated_current_frames, stacked_state
    #updated_current_frames is the deque object while stacked state can be converted into the picture


"""

#GITHUB WAY

stack_size = 4 # We stack 4 composite frames in total

# Initialize deque with zero-images one array for each image. Deque is a special kind of queue that deletes last entry when new entry comes in
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_image(state)
    
    if is_new_episode:
        # Clear our stacked_frames
         stacked_frames = deque([np.zeros((88,80), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x, apply elementwise maxima
        maxframe = np.maximum(frame,frame)
        stacked_frames.append(maxframe)
        stacked_frames.append(maxframe)
        stacked_frames.append(maxframe)
        stacked_frames.append(maxframe)
        
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        #Since deque append adds t right, we can fetch rightmost element
        maxframe=np.maximum(stacked_frames[-1],frame)
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(maxframe)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames
"""

observe, _, _, _ = env.step(1)

current_frames, stacked_state = update_current_frames(current_frames,observe, False)
image = Image.fromarray(stacked_state.astype(np.uint8))
image.save('stackedFALSENEWMYFUNCTION.png')

"""
episode_reward = 0
step = 0
done = False
"""

#Defining the model
    #Layers (3)
    #Flatten Image
    #Connect Layers
    #Output Layer
    #Weights

#Define our model's approach 
    #Epsilon-Greedy Policy, Loss Function, Episolon etc...
    #Depending on what learning we do, we will need helper functions for that. 
        #Github and Colab one uses Q-learning
 

#Helper functions to train and output data
    #Storage of our agents experience
    #Showing the video
    #Seperating into batches
    #Defining the Q table

#Training Code
#Output Code and Evaluation 


#Hyperparameters 
env.close()
    