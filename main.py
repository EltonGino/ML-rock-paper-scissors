# main.py

from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import player_lstm, player_feedforward, player_markov, player_ngram, player_pattern
from unittest import main

# List of player functions and their names
players = [
    (player_markov, "Markov Chain"),
    (lambda x, y=[]: player_ngram(x, y, n=3), "N-gram (n=3)"),
    (player_pattern, "Pattern Matching"),
    (player_feedforward, "Feedforward Neural Network"),
    (player_lstm, "LSTM Neural Network")
]

opponents = [
    (quincy, "Quincy"),
    (abbey, "Abbey"),
    (kris, "Kris"),
    (mrugesh, "Mrugesh")
]

# Test each bot against the provided bots
for player_func, player_name in players:
    print(f"\nTesting {player_name} against opponents...")
    for opponent_func, opponent_name in opponents:
        print(f"\nPlaying against {opponent_name} with {player_name}")
        play(player_func, opponent_func, 1000, verbose=True)

# Uncomment the line below to play interactively against a bot:
# play(human, abbey, 20, verbose=True)

# Uncomment the line below to play against a bot that plays randomly:
# play(human, random_player, 1000)

# Uncomment the line below to run unit tests automatically
main(module='test_module', exit=False)