import numpy as np
import random
from collections import deque
import tensorflow as tf
from keras import layers
from game.players import BasePokerPlayer
from game.engine import hand_evaluator
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'


class DQNPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.state_size = 10  # Example state size, you need to adjust based on your state representation
        self.action_size = 3  # Number of actions: fold, call, raise
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        act_values = self.model.predict(state)
        return valid_actions[np.argmax(act_values[0])]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self.get_state(hole_card, round_state)

        print(state)

        state = np.reshape(state, [1, self.state_size])
        action = self.act(state, valid_actions)

        self.last_state = state
        self.last_action = action
        self.last_round_state = round_state
        return action['action'], action['amount']
    
    def encode_card(cards):
        card_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
        encoding = [0] * (13 * 4)
        if(cards == None):
            return encoding
        for card in cards:
            print(card)
            rank, suit = card[0], card[1]
            index = suit_map[rank] * 4 + card_map[suit]
            encoding[index] = 1
        return encoding


    def get_state(self, hole_card, round_state):
        # Encode state
        hole_card_encoding = self.encode_cards(hole_card)
        community_card_encoding = self.encode_cards(round_state['community_card'])
        state = [
            round_state['pot']['main']['amount'],  # Pot size
            round_state['seats'][round_state['next_player']]['stack'],  # Player's stack
            len(round_state['community_card'])  # Number of community cards
        ] + hole_card_encoding + community_card_encoding
        return np.array(state, dtype=int)

        # self: hole cards
        # pot: main, side
        # comminity cards
        # next player: stack, state
    
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return DQNPlayer()
