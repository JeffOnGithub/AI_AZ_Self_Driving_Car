# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
if K._BACKEND == 'tensorflow':
    import tensorflow as tf
    
#Custom Keras smooth L1 loss function
#https://stackoverflow.com/questions/44130871/keras-smooth-l1-loss
HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    if K._BACKEND == 'tensorflow':
        x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    else:
        x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return  K.sum(x)

#https://stackoverflow.com/questions/34968722/softmax-function-python
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

        
# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        return samples

# Implementing Deep Q Learning

class Dqn():
    
    BATCH_SIZE = 100
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = self.build_model(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.last_state = np.array([np.zeros(input_size)])
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        TEMPERATURE = 10
        #Get model prediction
        actions = self.model.predict(state)
        #Softmaxed distribution 
        actions_softmaxed = softmax(actions[0] * TEMPERATURE)
        #Return an action from the distribution
        action = np.random.multinomial(1, np.array(actions_softmaxed))
        return np.argmax(action)
    
    def learn(self):
        samples = self.memory.sample(self.BATCH_SIZE)
        
        """
        #Item by item training (slow)
        #https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
        for sample in samples:
            state = sample[0]
            action = sample[2]
            reward = sample[3]
            new_state = sample[1]
            
            target = self.model.predict(state)

            Q_future = max(self.model.predict(new_state)[0])
            
            target[0][action] = reward + Q_future * self.gamma
            
            #print(target)
            
            self.model.train_on_batch(state, target)
        """

        #Item by item training (fast)
        
        #Rebuild memory in batches
        #TODO optimize process
        batch_state = []
        batch_next_state = []
        batch_action = []
        batch_reward = []
        for event in samples:
            batch_state.append(event[0][0])
            batch_next_state.append(event[1][0])
            batch_action.append(event[2][0])
            batch_reward.append([event[3][0],event[3][0],event[3][0]])
        batch_state = np.array(batch_state)
        batch_next_state = np.array(batch_next_state)
        batch_action = np.array(batch_action)
        batch_reward = np.array(batch_reward)
            
        #Set target at non-impactful values
        target = self.model.predict(batch_state)

        #Get Q futures
        Q_futures = self.model.predict(batch_next_state)
        Q_future = []
        for i in range(0, self.BATCH_SIZE):
            Q_max = max(Q_futures[i])
            Q_future.append(Q_max)
        Q_future = np.array(Q_future)
        
        #Adjust target with action and reward information
        for i in range(0, self.BATCH_SIZE):
            action = batch_action[i]
            target[i][action] = batch_reward[i][action] + Q_future[i] * self.gamma
            
        #Send batch to training 
        self.model.train_on_batch(batch_state, target)
    
    def update(self, reward, new_signal):
        new_state = np.array([new_signal])
        self.memory.push((self.last_state, new_state, np.array([self.last_action]), np.array([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > self.BATCH_SIZE:
            self.learn()
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        self.model.save_weights('last_brain.keras')
    
    def load(self):
        if os.path.isfile('last_brain.keras'):
            print("=> loading checkpoint... ")
            self.model.load_weights('last_brain.keras')
            print("done !")
        else:
            print("no checkpoint found...")
            
    def build_model(self, input_size, nb_action):
        #Build ANN model
        model = Sequential()
        model.add(Dense(30, input_shape=(input_size,), activation='relu'))
        model.add(Dense(nb_action, activation='linear'))
        
        #Configure optimizer
        optimizer = Adam(lr = 0.0001)
        
        #Compile
        model.compile(optimizer=optimizer,
              loss=smoothL1,
              metrics=['accuracy'])
        return model

