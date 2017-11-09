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
        samples = zip(*random.sample(self.memory, batch_size))
        return samples

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = self.build_model(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.last_state = np.zeros(input_size)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        TEMPERATURE = 100
        #Get model prediction
        actions = self.model.predict(state)
        #Apply temperature factor
        actions = map(lambda x: x * TEMPERATURE, actions)
        #Normalize the distribution
        actions = map(lambda x: x / sum(actions), actions)
        #Draw an action from the distribution
        return np.random.multinomial(1)
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #https://keon.io/deep-q-learning/
        for i in range(0,100):
            target = batch_reward[i] + self.gamma * np.amax(self.model.predict(batch_next_state[i])[0])
            target_f = self.model.predict(batch_state[i])
            target_f[0][batch_action[i]] = target
            self.model.train_on_batch(batch_state, target_f)
    
    def update(self, reward, new_signal):
        new_state = np.array([new_signal])
        self.memory.push((self.last_state, new_state, np.array([self.last_action]), np.array([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
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
        self.model.save('last_brain.keras')
    
    def load(self):
        if os.path.isfile('last_brain.keras'):
            print("=> loading checkpoint... ")
            self.model = load_model('last_brain.keras')
            print("done !")
        else:
            print("no checkpoint found...")
            
    def build_model(self, input_size, nb_action):
        #Build ANN model
        model = Sequential()
        model.add(Dense(30, input_shape=(input_size,), activation='relu'))
        model.add(Dense(nb_action, activation='softmax'))
        
        #Configure optimizer
        optimizer = Adam(lr = 0.001)
        
        #Compile
        model.compile(optimizer=optimizer,
              loss=smoothL1,
              metrics=['accuracy'])
        return model

