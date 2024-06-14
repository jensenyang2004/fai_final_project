from game.players import BasePokerPlayer
from agents.DQNAgent import DQNAgent
from game.engine.hand_evaluator import HandEvaluator as evaluator
# from agents.DQNAgent_torch import DQNAgent
import numpy as np
import random
from collections import namedtuple


Card = namedtuple('Card', ['rank', 'suit'])

# import os
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['TF_NUM_INTEROP_THREADS'] = '1'
# os.environ['TF_NUM_INTRAOP_THREADS'] = '1'



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
        self.name = None

    def declare_action(self, valid_actions, hole_card, round_state):
        if(self.name == None):
            self.name = round_state["seats"][round_state["next_player"]]["name"]
        if self.last_state is not None and self.last_action is not None and self.training:
            reward = 0
            for player in round_state["seats"]:
                if (player["name"] == "RL"):
                    if(player["stack"] > 1000):
                        reward = 1
                    elif(player["stack"] < 1000):
                        reward = -1
            next_state = self.get_state(None, round_state)
            self.agent.remember(self.last_state, self.last_action, reward, next_state, True)
            self.agent.replay(32)  # Batch size for replay
            self.last_state = None
            self.last_action = None

        state = self.get_state(hole_card, round_state, valid_actions)
        action_idx = self.agent.act(state)
        action, amount = self.get_action_amount(action_idx, round_state, valid_actions)
        self.last_action = action_idx
        self.last_state = state
        if(action == "raise"):
            amount = min(amount, valid_actions[2]["amount"]["max"])
            amount = max(amount, valid_actions[2]["amount"]["min"])
        if amount <= 0:
            self.last_action = 1
            return "call", valid_actions[1]["amount"]
        return action, amount

    def get_state(self, hole_card, round_state, valid_actions = None):
        hole_card_encoding = self.encode_cards(hole_card)
        community_card_encoding = self.encode_cards(round_state['community_card'])

        # Normalize numerical values and use one-hot encoding where appropriate
        pot_size = round_state['pot']['main']['amount'] / 1000.0  # Normalize pot size
        stage = self.encode_stage(round_state['street'])  # One-hot encode stage
        player_state = self.encode_player_state(round_state, valid_actions)  # Encode player-specific state
        state = (
            [pot_size]  # Community pot
            + stage  # Stage state
            + player_state  # Player state
            + hole_card_encoding  # Hole cards
            + community_card_encoding  # Community cards
        )
        return np.array(state, dtype=float)


    def get_action_amount(self, action, round_state, valid_actions):
        if action == 0:
            return "fold", 0
        elif action == 1:
            return "call", valid_actions[1]["amount"]
        elif action == 2:
            return "raise", valid_actions[2]["amount"]["min"]
        elif action == 3:
            return "raise", round_state['pot']['main']['amount'] / 2
        elif action == 4:
            return "raise", round_state['pot']['main']['amount']
        elif action == 5:
            return "raise", 2 * round_state['pot']['main']['amount']
        elif action == 6:
            return "raise", round_state['seats'][round_state['next_player']]['stack']
        else:
            raise ValueError("Invalid action")

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

    def encode_stage(self, street):
        stages = ['preflop', 'flop', 'turn', 'river']
        encoding = [0] * len(stages)
        if street in stages:
            encoding[stages.index(street)] = 1
        return encoding

    def encode_player_state(self, round_state, valid_actions):
        stack = 0
        for player in round_state["seats"]:
            if player["name"] == self.name:
                stack = player["stack"]
        stack_norm =  stack / 1000.0  # Normalize stack size
        contribution = self.encode_contribution(stack, round_state['pot']['main']['amount'])
        active_players = sum(seat['state'] == 'participating' for seat in round_state['seats']) / len(round_state['seats'])
        return [stack_norm, contribution, active_players]

    def encode_position(self, next_player):
        positions = ['small_blind', 'big_blind', 'other']
        encoding = [0] * len(positions)
        if next_player in positions:
            encoding[positions.index(next_player)] = 1
        return encoding

    def encode_contribution(self, stack, pot):
        return stack / (pot + stack)

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
            reward = 0
            for player in round_state["seats"]:
                if (player["name"] == "RL"):
                    reward = float(player["stack"] - 1000)/1000

            for player in winners:
                if (player["name"] == "RL"):
                    reward = 1  
            self.agent.remember(self.last_state, self.last_action, reward, next_state, True)
            self.agent.replay(32)  # Batch size for replay
            self.last_state = None
            self.last_action = None
    
    def save_model(self, name = "Model01"):
        self.agent.save(name)

    def load_model(self, name = "Model01"):
        self.agent.load(name)

def setup_ai(training = False, Model_file = "./Model_summer_BananaMilk_v2"):
    state_size = 112  # Adjusted state size to include the encoded state
    action_size = 7  # Assuming 3 possible actions: fold, call, raise
    return PokerDQNPlayer(state_size, action_size, training, Model_file)
