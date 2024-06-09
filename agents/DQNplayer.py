from game.players import BasePokerPlayer
from agents.DQNAgent import DQNAgent
import numpy as np
import random

class PokerDQNPlayer(BasePokerPlayer):
    def __init__(self, state_size, action_size):
        self.agent = DQNAgent(state_size, action_size)
        self.last_state = None
        self.last_action = None

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self.get_state(hole_card, round_state)
        action_idx = self.agent.act(state)
        print("shit")
        self.last_state = state
        self.last_action = action_idx
        
        if valid_actions[action_idx]['action'] == 'raise':
            amount = random.randint(valid_actions[action_idx]['amount']['min'], valid_actions[action_idx]['amount']['max'])
        else:
            amount = valid_actions[action_idx]['amount']
        
        return valid_actions[action_idx]['action'], amount

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

    def encode_cards(self, cards):
        card_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
        encoding = [0] * (13 * 4)
        for card in cards:
            rank, suit = card[0], card[1]
            index = suit_map[rank] * 4 + card_map[suit]
            encoding[index] = 1
        return encoding

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        if self.last_state is not None and self.last_action is not None:
            reward = self.calculate_reward(winners, hand_info)
            next_state = self.get_state(None, round_state)
            self.agent.remember(self.last_state, self.last_action, reward, next_state, True)
            self.agent.replay(32)  # Batch size for replay
            self.last_state = None
            self.last_action = None

    def calculate_reward(self, winners, hand_info):
        # Define a reward function based on the game result
        return 1 if self in winners else -1

def setup_ai():
    state_size = 128  # Adjusted state size to include the encoded state
    action_size = 3  # Assuming 3 possible actions: fold, call, raise
    return PokerDQNPlayer(state_size, action_size)
