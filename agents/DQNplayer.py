from game.players import BasePokerPlayer
from agents.DQNAgent import DQNAgent
from game.engine.hand_evaluator import HandEvaluator as evaluator
# from agents.DQNAgent_torch import DQNAgent
import numpy as np
import random
from collections import namedtuple


Card = namedtuple('Card', ['rank', 'suit'])

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'


class PokerDQNPlayer(BasePokerPlayer):
    def __init__(self, state_size, action_size, training, model_file = None):
        self.agent = DQNAgent(state_size, action_size)
        if model_file:
            self.load_model(model_file)
        self.last_state = None
        self.last_action = None
        self.training = training
        self.gameWins = 0
        self.gamePlay = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        if self.last_state is not None and self.last_action is not None and self.training:
            reward = 0
            for player in round_state["seats"]:
                if(player["name"] == "RL"):
                    if(player["state"] == "participating"):
                        reward = 0
                    else:
                        reward = -1
            next_state = self.get_state(None, round_state)
            self.agent.remember(self.last_state, self.last_action, reward, next_state, True)
            self.agent.replay(32)  # Batch size for replay
            self.last_state = None
            self.last_action = None

        state = self.get_state(hole_card, round_state)
        action_idx = self.agent.act(state)
        self.last_state = state
        self.last_action = action_idx
        actions_choices = [
            {"action" : "fold", "amount" : 0},
            {"action" : "call", "amount" : valid_actions[1]['amount']},
            {"action" : "raise", "amount" : valid_actions[2]['amount']['min']},
            {"action" : "raise", "amount" : (valid_actions[2]['amount']['max'] + valid_actions[2]['amount']['min'])/3},
            {"action" : "raise", "amount" : (valid_actions[2]['amount']['max'] + valid_actions[2]['amount']['min'])*2/3},
            {"action" : "raise", "amount" : valid_actions[2]['amount']['max']}
        ]
        if actions_choices[action_idx]["amount"] < actions_choices[1]['amount']:
            return actions_choices[1]["action"], actions_choices[1]["amount"]
        return actions_choices[action_idx]["action"], actions_choices[action_idx]["amount"]

    def get_state(self, hole_card, round_state, valid_actions = None):
        hole_card_obj = [self.convert_to_card(x) for x in hole_card]
        community_card_obj = [self.convert_to_card(x) for x in round_state["community_card"]]
        result = evaluator.gen_hand_rank_info(hole_card_obj, community_card_obj)
        print(result)
        # Encode state
        hole_card_encoding = self.encode_cards(hole_card)
        # Encode community cards
        community_card_encoding = self.encode_cards(round_state['community_card'])
        biding = [0, 0, 0]
        if valid_actions != None:
            biding = [valid_actions[1]['amount'], valid_actions[2]['amount']['min'], valid_actions[2]['amount']['max']]
        state = (
            [round_state['pot']['main']['amount']]  # Number of community cards
            + hole_card_encoding
            + community_card_encoding
            + biding
            # + next_stake
        )
        return np.array(state, dtype=int)

    def convert_to_card(card_str):
        print("hey")
        rank_str = card_str[1]  # Get the rank part of the string
        suit_str = card_str[0]   # Get the suit part of the string
        
        # Mapping of rank strings to integers
        rank_map = {
            'A': 14, 'K': 13, 'Q': 12, 'J': 11,
            'T': 10, '9': 9, '8': 8, '7': 7,
            '6': 6, '5': 5, '4': 4, '3': 3, '2': 2
        }
        
        # Mapping of suit strings
        suit_map = {
            'C': 'clubs',
            'D': 'diamonds',
            'H': 'hearts',
            'S': 'spades'
        }
        
        rank = rank_map[rank_str]
        suit = suit_map[suit_str]
        return Card(rank=rank, suit=suit)

    def encode_cards(self, cards):
        card_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
        encoding = [0] * (13 * 4)
        if(cards == None):
            return encoding
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
        self.gamePlay += 1
        if self.last_state is not None and self.last_action is not None and self.training:
            next_state = self.get_state(None, round_state)
            self.agent.remember(self.last_state, self.last_action, self.calculate_reward(winners, hand_info), next_state, True)
            self.agent.replay(32)  # Batch size for replay
            self.last_state = None
            self.last_action = None

    def calculate_reward(self, winners, hand_info):
        for winner in winners:
            if winner["name"] == "RL":
                self.gameWins += 1
                return 1
            else:
                return -1
        # Define a reward function based on the game result
    
    def save_model(self, name = "Model01"):
        self.agent.save(name)

    def load_model(self, name = "Model01"):
        self.agent.load(name)

def setup_ai(training = True, Model_file = None):
    state_size = 108  # Adjusted state size to include the encoded state
    action_size = 6  # Assuming 3 possible actions: fold, call, raise
    return PokerDQNPlayer(state_size, action_size, training, Model_file)
