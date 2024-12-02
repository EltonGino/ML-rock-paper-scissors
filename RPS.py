# RPS.py

from RPS_game import play, quincy, abbey, mrugesh, kris
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Function to simulate games and generate training data with noise
def generate_data_with_noise(opponent, num_games=5000):
    opponent_history = []
    bot_history = []

    def random_player(prev_play, opponent_history=[]):
        return np.random.choice(['R', 'P', 'S'])

    for _ in range(num_games):
        bot_move = random_player(None)
        if np.random.rand() < 0.1:  # Add 10% noise
            opponent_move = random_player(bot_move)
        else:
            opponent_move = opponent(bot_move)

        bot_history.append(bot_move)
        opponent_history.append(opponent_move)

    return opponent_history

# Generate data from multiple opponents with noise
opponent_history_quincy = generate_data_with_noise(quincy, 5000)
opponent_history_abbey = generate_data_with_noise(abbey, 5000)
opponent_history_mrugesh = generate_data_with_noise(mrugesh, 5000)
opponent_history_kris = generate_data_with_noise(kris, 5000)

# Combine all opponent histories
opponent_history_all = (opponent_history_quincy +
                        opponent_history_abbey +
                        opponent_history_mrugesh +
                        opponent_history_kris)

# Process data for LSTM model training
def process_data_lstm(opponent_history, sequence_length=10):
    move_map = {'R': 0, 'P': 1, 'S': 2}
    encoded_moves = [move_map[move] for move in opponent_history]

    # Create sequences of moves and their corresponding next move
    X = [encoded_moves[i:i + sequence_length] for i in range(len(encoded_moves) - sequence_length)]
    y = encoded_moves[sequence_length:]  # The next move after each sequence

    return np.array(X), np.array(y)

# Prepare data for feedforward neural network
def process_data_feedforward(opponent_history, sequence_length=10):
    move_map = {'R': [1, 0, 0], 'P': [0, 1, 0], 'S': [0, 0, 1]}
    encoded_moves = [move_map[move] for move in opponent_history]

    X = [np.array(encoded_moves[i:i + sequence_length]).flatten() for i in range(len(encoded_moves) - sequence_length)]
    y = [encoded_moves[i + sequence_length] for i in range(len(encoded_moves) - sequence_length)]

    return np.array(X), np.array(y)

# Build and train the LSTM model
def build_lstm_model():
    model = Sequential([
        Embedding(input_dim=3, output_dim=5),
        LSTM(100, activation='tanh', return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='tanh'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train the feedforward neural network
def build_feedforward_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare data for LSTM model
X_lstm, y_lstm = process_data_lstm(opponent_history_all, sequence_length=10)
# Split into training and validation datasets
X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Build and train the LSTM model
model_lstm = build_lstm_model()
early_stopping_lstm = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_lstm.fit(X_train_lstm, y_train_lstm, validation_data=(X_val_lstm, y_val_lstm), epochs=50, batch_size=32, callbacks=[early_stopping_lstm])
# Save the LSTM model
model_lstm.save('rps_model_lstm.keras')

# Prepare data for feedforward neural network
X_ff, y_ff = process_data_feedforward(opponent_history_all, sequence_length=10)
X_train_ff, X_val_ff, y_train_ff, y_val_ff = train_test_split(X_ff, y_ff, test_size=0.2, random_state=42)

# Build and train the feedforward neural network
input_dim = X_train_ff.shape[1]
model_ff = build_feedforward_model(input_dim)
early_stopping_ff = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_ff.fit(X_train_ff, y_train_ff, validation_data=(X_val_ff, y_val_ff), epochs=50, batch_size=32, callbacks=[early_stopping_ff])
# Save the feedforward model
model_ff.save('rps_model_feedforward.keras')

# Load the trained models
model_lstm = load_model('rps_model_lstm.keras')
model_ff = load_model('rps_model_feedforward.keras')

# Player functions

# 1. LSTM Model Player
def player_lstm(prev_play, opponent_history=[]):
    if prev_play == '':
        prev_play = 'R'

    opponent_history.append(prev_play)

    if len(opponent_history) < 10:
        # Randomize the first few moves to avoid predictability
        return np.random.choice(['R', 'P', 'S'])

    # Convert opponent's last 10 moves into a numerical sequence
    move_map = {'R': 0, 'P': 1, 'S': 2}
    reverse_map = {0: 'R', 1: 'P', 2: 'S'}
    input_seq = [move_map[move] for move in opponent_history[-10:]]
    input_seq = np.array(input_seq).reshape(1, -1)

    # Predict the next move
    prediction = model_lstm.predict(input_seq, verbose=0)
    predicted_move = reverse_map[np.argmax(prediction)]

    # Return the counter to the predicted move
    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
    return ideal_response[predicted_move]

# 2. Feedforward Neural Network Player
def player_feedforward(prev_play, opponent_history=[]):
    if prev_play == '':
        prev_play = 'R'
    opponent_history.append(prev_play)

    if len(opponent_history) < 10:
        return np.random.choice(['R', 'P', 'S'])

    move_map = {'R': [1, 0, 0], 'P': [0, 1, 0], 'S': [0, 0, 1]}
    reverse_map = {0: 'R', 1: 'P', 2: 'S'}
    input_seq = [move_map[move] for move in opponent_history[-10:]]
    input_seq = np.array(input_seq).flatten().reshape(1, -1)

    prediction = model_ff.predict(input_seq, verbose=0)
    predicted_move = reverse_map[np.argmax(prediction)]

    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
    return ideal_response[predicted_move]

# 3. Markov Chain Player
def player_markov(prev_play, opponent_history=[]):
    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}

    if prev_play == '':
        prev_play = 'R'
    opponent_history.append(prev_play)

    n = 1  # Use the last move to predict the next move

    if len(opponent_history) <= n:
        return np.random.choice(['R', 'P', 'S'])

    # Build transition matrix
    history = ''.join(opponent_history)
    transitions = defaultdict(lambda: defaultdict(int))

    for i in range(len(history) - n):
        seq = history[i:i+n]
        next_move = history[i+n]
        transitions[seq][next_move] += 1

    last_seq = ''.join(opponent_history[-n:])
    if last_seq in transitions:
        next_moves = transitions[last_seq]
        predicted_move = max(next_moves, key=next_moves.get)
    else:
        predicted_move = np.random.choice(['R', 'P', 'S'])

    return ideal_response[predicted_move]

# 4. N-gram Model Player
def player_ngram(prev_play, opponent_history=[], n=3):
    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}

    if prev_play == '':
        prev_play = 'R'
    opponent_history.append(prev_play)

    if len(opponent_history) <= n:
        return np.random.choice(['R', 'P', 'S'])

    # Build n-gram frequency dictionary
    history = ''.join(opponent_history)
    ngrams = defaultdict(lambda: defaultdict(int))

    for i in range(len(history) - n):
        seq = history[i:i+n]
        next_move = history[i+n]
        ngrams[seq][next_move] += 1

    last_seq = ''.join(opponent_history[-n:])
    if last_seq in ngrams:
        next_moves = ngrams[last_seq]
        predicted_move = max(next_moves, key=next_moves.get)
    else:
        predicted_move = np.random.choice(['R', 'P', 'S'])

    return ideal_response[predicted_move]

# 5. Pattern Matching Player
# def player_pattern(prev_play, opponent_history=[]):
#     ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
#
#     if prev_play == '':
#         prev_play = 'R'
#     opponent_history.append(prev_play)
#
#     if len(opponent_history) < 3:
#         return np.random.choice(['R', 'P', 'S'])
#
#     # Convert history to a string
#     history_str = ''.join(opponent_history)
#     # Use a sliding window to find the longest matching pattern
#     match_found = False
#     for l in range(len(history_str)-1, 0, -1):
#         pattern = history_str[-l:]
#         occurrences = history_str[:-1].count(pattern)
#         if occurrences > 0:
#             match_found = True
#             break
#
#     if match_found:
#         # Find all occurrences of the pattern and the subsequent move
#         indices = [i for i in range(len(history_str) - l) if history_str[i:i+l] == pattern]
#         next_moves = [history_str[i+l] for i in indices if i+l < len(history_str)-1]
#         if next_moves:
#             predicted_move = max(set(next_moves), key=next_moves.count)
#             return ideal_response[predicted_move]
#
#     # If no pattern found, return random
#     return np.random.choice(['R', 'P', 'S'])

def player_pattern(prev_play, opponent_history=[], my_history=[]):
    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}

    if prev_play == '':
        prev_play = 'R'
    opponent_history.append(prev_play)

    if len(opponent_history) < 3:
        guess = np.random.choice(['R', 'P', 'S'])
    else:
        # Build a string of opponent's moves
        history_str = ''.join(opponent_history)
        # Find the longest pattern in opponent's history
        pattern = history_str[-3:]
        # Find all occurrences of the pattern
        occurrences = [i for i in range(len(history_str) - 3) if history_str[i:i + 3] == pattern]
        # Collect the moves that followed the pattern
        next_moves = [history_str[i + 3] for i in occurrences if i + 3 < len(history_str)]
        if next_moves:
            predicted_move = max(set(next_moves), key=next_moves.count)
            guess = ideal_response[predicted_move]
        else:
            guess = np.random.choice(['R', 'P', 'S'])

    my_history.append(guess)
    return guess

# Choose the best performing player as 'player' for unit tests
player = player_pattern  # Replace with the best-performing player function