import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai
from agents.tensorPlayer import setup_ai as tensor_ai

from baseline0 import setup_ai as baseline0_ai

from agents.DQNplayer import setup_ai as DQNplayer

config = setup_config(max_round=10000, initial_stack=1000, small_blind_amount=5)
config.register_player(name="p1", algorithm=baseline0_ai())
config.register_player(name="p2", algorithm=random_ai())
config.register_player(name="p3", algorithm=call_ai())
config.register_player(name="p4", algorithm=DQNplayer())

## Play in interactive mode if uncomment
#config.register_player(name="me", algorithm=console_ai())

all_game_results = []
for i in range(1000):
    game_result = start_poker(config, verbose=1)
    all_game_results.append(game_result)
    print(f"Game {i+1} result added.")

with open("all_game_results.json", "w") as f:
    json.dump(all_game_results, f, indent=4)


# for i in range (1000):
#     game_result = start_poker(config, verbose=1)
#     print(json.dumps(game_result, indent=4))


