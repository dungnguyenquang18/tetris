from model.base_model.dqnagent import DQNAgent
from TetrisBattle.envs.tetris_env import TetrisDoubleEnv, TetrisSingleEnv
from Personal_tetris_env import Tetris
from datetime import datetime
from statistics import mean
from logs import CustomTensorBoard
from tqdm import tqdm
        
from keras.models import Sequential, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random
from copy import deepcopy
from Personal_tetris_env import Tetris
from TetrisBattle.envs.tetris_env import TetrisDoubleEnv, TetrisSingleEnv
START_IN_ROW = 4
START_IN_COL = 2

DEPTH_BOARD = 20
WIDTH_BOARD = 10

DEFAULT_GRID = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

ipieces = [[[0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]],
           [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0]],
           [[0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0]],
           [[0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]]
opieces = [[[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]],
           [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]],
           [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]],
           [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]]]

jpieces = [[[0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]],
           [[0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 0]],
           [[0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0]],
           [[0, 0, 0, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]]

lpieces = [[[0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]],
           [[0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]],
           [[0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]],
           [[0, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]]
zpieces = [[[0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]],
           [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0]],
           [[0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]],
           [[0, 0, 1, 1],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]]
spieces = [[[0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]],
           [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0]],
           [[0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]],
           [[1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]]

tpieces = [[[0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]],
           [[0, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]],
           [[0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 0]],
           [[0, 0, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]]

NUM_PIECES = [1, 2, 3, 4, 5, 6, 7]
MAP_NUM_PIECE = {1: ipieces, 2: opieces, 3: jpieces, 4: lpieces, 5: zpieces, 6: spieces, 7: tpieces}
def initialize(state):
    # initialize
    board = []
    holding = 0
    pieces = []
    for i in range(20):
        row = []
        for j in range(0, 10):
            row.append(state[i][j][0])
        board.append(row[:])

    for row in range(20):
        for i in range(10):
            if board[row][i] == np.float32(0.7) or board[row][i] == np.float32(0.3):
                board[row][i] = int(0)
            elif board[row][i] == 0:
                board[row][i] = int(0)
            else:
                board[row][i] = int(1)

    new_board = []
    for col in range(10):
        new_row = []
        for row in range(20):
            new_row.append(board[row][col])
        new_board.append(new_row[:])

    # get the holding piece
    for i in range(10, 17):
        if state[0][i][0] == 1:
            holding = i - 9

    # get next 5 pieces
    for j in range(1, 6):
        for i in range(10, 17):
            if state[j][i][0] == 1:
                pieces.append(i - 9)
                break
    return new_board, holding, pieces


def get_a_possible_move_list(right=0, left=0, rot_right=0, rot_left=0):
    a_possible_move_list = []
    for _ in range(rot_left):
        a_possible_move_list.append(4)
    for _ in range(rot_right):
        a_possible_move_list.append(3)
    for _ in range(right):
        a_possible_move_list.append(5)
    for _ in range(left):
        a_possible_move_list.append(6)
    a_possible_move_list.append(2)
    return a_possible_move_list


def get_possible_move_lists(possible_move_lists, nowblock):
    max_left = 4
    max_right = 3
    """extra"""
    newpossible_movelists = []
    for item in possible_move_lists:
        new_item = deepcopy(item)
        if nowblock == 1:
            """ no rotate"""
            max_left = 4
            max_right = 2
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right))
            """rotate right: 1"""
            max_left = 6
            max_right = 3
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left, rot_right=1))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right, rot_right=1))
        elif nowblock == 2:
            max_left = 5
            max_right = 3
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right))
        elif nowblock == 3 or nowblock == 4 or nowblock == 7:
            """no rotate"""
            max_left = 4
            max_right = 3
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right))
            """rotate left = 2"""
            max_left = 4
            max_right = 3
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left, rot_left=2))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right, rot_left=2))
            """ rotate left 1"""
            max_left = 4
            max_right = 4
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left, rot_left=1))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right, rot_left=1))
            """rotate right"""
            max_left = 5
            max_right = 3
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left, rot_right=1))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right, rot_right=1))
        elif nowblock == 5 or nowblock == 6:
            """no rotate"""
            max_left = 4
            max_right = 3
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right))
            """rotate_right"""
            max_left = 5
            max_right = 3
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left, rot_right=1))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right, rot_right=1))
            """rotate left"""
            max_left = 4
            max_right = 4
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left, rot_left=1))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right, rot_left=1))
    return newpossible_movelists


def get_env_from_move(board, list_block, list_move):
    game_check = Tetris()
    game_check.board = board
    game_check.clear()
    extra_height = game_check.cleared
    game_check.cleared = 0
    game_check.index_block = list_block[0]
    game_check.next_blocks = list_block[1:]
    game_check.current_block = MAP_NUM_PIECE[game_check.index_block][0]
    done = False
    state_board = game_check.board
    for one_move in list_move:
        state_board, done = game_check.move(one_move)
        if done == True:
            break
    return game_check


def get_list_poss_env(board, list_block):
    possible_move_lists = [[]]
    for block in list_block:
        new_list = get_possible_move_lists(possible_move_lists, block)
        possible_move_lists = deepcopy(new_list)
    list_poss_env = {}
    # key: move list
    # value: env

    for cur_list in possible_move_lists:
        # env_copy = env.copy()
        # env will be copied in get_rating_from_move() function, so env local will be not changed
        env = get_env_from_move(board, list_block, cur_list)
        # Convert the list to a tuple before using it as a key
        list_poss_env[tuple(env.get_info_from_state())] = cur_list

    return list_poss_env
# Run dqn with Tetris
def dqn():
    env = TetrisSingleEnv()
    episodes = 600 # total number of episodes
    max_steps = None # max number of steps per game (None for infinite)
    epsilon_stop_episode = 2000 # at what episode the random exploration stops
    mem_size = 1000 # maximum number of steps stored by the agent
    discount = 0.95 # discount in the Q-learning formula (see DQNAgent)
    batch_size = 128 # number of actions to consider in each training
    epochs = 1 # number of epochs per training
    render_every = 50 # renders the gameplay every x episodes
    render_delay = None # delay added to render each frame (None for no delay)
    log_every = 50 # logs the current stats every x episodes
    replay_start_size = 1000 # minimum steps stored in the agent required to start training
    train_every = 1 # train every x episodes
    n_neurons = [32, 32, 32] # number of neurons for each activation layer
    activations = ['relu', 'relu', 'relu', 'linear'] # activation layers
    save_best_model = True # saves the best model so far at "best.keras"

    agent = DQNAgent( n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []
    best_score = 0

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # Game
        diem = 0
        list_act = deque()
        while not done and (not max_steps or steps < max_steps):
            # state -> action
            # next_states = {tuple(v):k for k, v in env.get_next_states().items()}
            # best_state = agent.best_state(next_states.keys())
            # best_action = next_states[best_state]
            #
            # reward, done = env.play(best_action[0], best_action[1], render=render,
            #                         render_delay=render_delay)
            #
            # agent.add_to_memory(current_state, best_state, reward, done)
            cur_board, cur_holding, cur_pieces  = initialize(current_state)
            cur_env = Tetris()
            cur_env.board = cur_board
            cur_in4 = cur_env.get_info_from_state()
            if len(list_act) == 0:
                list_poss_in4 = get_list_poss_env(cur_board, cur_pieces[:1])
                best_in4 = agent.best_state(list_poss_in4.keys())
                list_act = deque(best_in4)
            
            nxt_state, reward, done,_ = env.step(list_act.popleft())
            nxt_board, nxt_holding, nxt_pieces  = initialize(nxt_state)
            nxt_env = Tetris()
            nxt_env.board = nxt_board
            nxt_in4 = nxt_env.get_info_from_state()
            agent.add_to_memory(cur_in4, nxt_in4, reward, done)
            current_state = nxt_state
            
            diem += reward
            steps += 1

        scores.append(steps)

        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score)

        # Save model
        if save_best_model and steps > best_score:
            print(f'Saving a new best model (score={steps}, episode={episode})')
            best_score = steps
            agent.save_model("best.keras")


if __name__ == "__main__":
    dqn()
