from game.players import BasePokerPlayer
import random
from collections import Counter

class MonteCarloPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()

    def declare_action(self, valid_actions, hole_card, round_state):
        # Monte Carlo simulation parameters
        num_simulations = 1000  # Number of simulations per decision

        # Extract community cards
        community_cards = round_state['community_card']
        deck = self.get_deck(hole_card + community_cards)
        my_score = self.hand_strength(hole_card + community_cards)

        call_action_info = valid_actions[1]
        raise_action_info = valid_actions[2]
        fold_action_info = valid_actions[0]


        call_amount = call_action_info['amount']
        min_raise_amount = raise_action_info['amount']['min']
        max_raise_amount = raise_action_info['amount']['max']

        win_rate = self.monte_carlo_simulation(deck, community_cards, hole_card, num_simulations)
        
        print(win_rate)
        # Decision making based on win rate
        if win_rate > 0.9:
            action = raise_action_info['action']
            amount = max_raise_amount  # Simplification: always raise minimum amount
        elif win_rate > 0.6:
            action = raise_action_info['action']
            amount = min_raise_amount  # Simplification: always raise minimum amount
        elif win_rate > 0.2:
            action = call_action_info['action']
            amount = call_amount
        else:
            action = fold_action_info['action']
            amount = fold_action_info['amount']

        return action, amount

    def monte_carlo_simulation(self, deck, community_cards, my_hole_cards, num_simulations):
        wins = 0
        losses = 0

        for _ in range(num_simulations):
            deck_copy = deck[:]
            random.shuffle(deck_copy)

            opponent_hole_cards = [deck_copy.pop(), deck_copy.pop()]
            remaining_community_cards = community_cards[:]

            while len(remaining_community_cards) < 5:
                remaining_community_cards.append(deck_copy.pop())

            my_score = self.hand_strength(my_hole_cards + remaining_community_cards)
            opponent_score = self.hand_strength(opponent_hole_cards + remaining_community_cards)

            if my_score > opponent_score:
                wins += 1
            else:
                losses += 1

        return wins / num_simulations

    def get_deck(self, excluded_cards):
        SUITS = 'SHDC'
        RANKS = '23456789TJQKA'
        deck = [suit + rank for suit in SUITS for rank in RANKS]
        for card in excluded_cards:
            deck.remove(card)
        return deck

    def hand_strength(self, cards):
        # Simplified hand strength evaluation
        # In a real scenario, you would use a library to evaluate poker hands
        card_ranks = [card[0] for card in cards]
        counter = Counter(card_ranks)
        most_common = counter.most_common(1)[0]
        return most_common[1]

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
    return MonteCarloPlayer()
