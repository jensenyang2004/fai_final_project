import numpy as np
import random
import tensorflow as tf
from collections import deque

from keras import layers, models, optimizers

class DQN:
    def __init__(self, state_size, action_size):
        self.model = self._build_model(state_size, action_size)

    def _build_model(self, state_size, action_size):
        model = models.Sequential()
        model.add(layers.Dense(100, input_dim=state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam())
        return model

    def predict(self, state):
        return self.model.predict(state)

    def fit(self, state, target, verbose=0):
        self.model.fit(state, target, epochs=1, verbose=verbose)

    def load(self, name):
        self.model = models.load_model(name)

    def save(self, name):
        self.model.save(name)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.model.set_weights(self.model.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return  # Not enough samples to perform replay
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, name = "Model01"):
        self.model.save(name)
        # Save additional attributes
        np.savez(name + "_attributes", epsilon=self.epsilon, memory=list(self.memory))

    def load(self, name):
        self.model.load(name)
        print(f"\033[31mload successfully!!! \033[0m")
        # Load additional attributes
        data = np.load(name + "_attributes.npz", allow_pickle=True)
        self.epsilon = data['epsilon']
        self.memory = deque(data['memory'], maxlen=2000)
        print(f"\033[31mload successfully!!! \033[0m")
