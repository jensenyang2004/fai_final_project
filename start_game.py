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
# from agents.RLplayer import setup_ai as RLplayer
# from agents.MonetCarlo import setup_ai as MonteCarlo


# player = DQNplayer(True, "./fai_final_project/torchModel/Model_108_6_100")
# player = DQNplayer(True, "./fai_final_project/tfModel/Model_Chocolate")
player = DQNplayer(True, "./fai_final_project/tfModel/Model_Simplified_v2")
# player = DQNplayer(True, )
# player_prac = DQNplayer(False, "./fai_final_project/tfModel/Model_Chocolate")



config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
# config.register_player(name="p1", algorithm=player)
config.register_player(name="p2", algorithm=baseline1_ai())
config.register_player(name="p3", algorithm=baseline2_ai())
# config.register_player(name="p4", algorithm=baseline3_ai())
# config.register_player(name="p5", algorithm=baseline4_ai())
# config.register_player(name="p6", algorithm=baseline5_ai())
config.register_player(name="p7", algorithm=baseline7_ai())

# config.register_player(name="p8", algorithm=player_prac)
# config.register_player(name="RL_practice", algorithm=player_practice)
config.register_player(name="RL", algorithm=player)

all_game_results = []
RL_gameWins = 0
total_games = 10  

for j in range(20):
    for i in range(total_games):
        print(f"\033[34mepisod {i + 1} / {total_games} \033[0m")
        game_result = start_poker(config, verbose=1)

    # print(f"\033[32mgame wins {player.gameWins} / {player.gamePlay} \033[0m")
        
    player.save_model("./fai_final_project/tfModel/Model_Simplified_v2")
