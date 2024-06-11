import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai

from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai 
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai

from agents.DQNplayer import setup_ai as DQNplayer
from agents.RLplayer import setup_ai as RLplayer
from agents.MonetCarlo import setup_ai as MonteCarlo


# player = DQNplayer(False, "./fai_final_project/torchModel/Model_108_6_100")
player = DQNplayer(False, "./fai_final_project/tfModel/Model_comprehensive")

config = setup_config(max_round=1, initial_stack=1000, small_blind_amount=5)
config.register_player(name="p1", algorithm=baseline0_ai())
# config.register_player(name="p1", algorithm=call_ai())
config.register_player(name="RL", algorithm=player)


all_game_results = []
RL_gameWins = 0
total_games = 1   

for i in range(total_games):
    print(f"\033[32mepisod {i+1} / {total_games} \033[0m")
    game_result = start_poker(config, verbose=1)
    all_game_results.append(game_result)
    if(game_result["players"][0]['stack'] < game_result["players"][1]['stack']):
        RL_gameWins += 1

if(RL_gameWins >= 3):
    print(f"\033[32mtotal wins {RL_gameWins} / {total_games} \033[0m")
else:
    print(f"\033[31mtotal wins {RL_gameWins} / {total_games} \033[0m")
    
