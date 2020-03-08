import numpy as np
import gym
from gym import spaces


class TicTacToe(gym.Env):

    reward_range = (-np.inf, np.inf)
    observation_space = spaces.MultiDiscrete([2 for _ in range(0, 9 * 3)])
    action_space = spaces.Discrete(9)

    """
    Board looks like:
    [0, 1, 2,
     3, 4, 5,
     6, 7, 8]
    """
    winning_streaks = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]

    def __init__(self, summary: dict = None):
        super().__init__()
        if summary is None:
            summary = {
                "total games": 0,
                "ties": 0,
                "illegal moves": 0,
                "player 0 wins": 0,
                "player 1 wins": 0,
            }
        self.summary = summary

    def seed(self, seed=None):
        pass

    def _one_hot_board(self):
        if self.current_player == 0:
            return np.eye(3)[self.board].reshape(-1)
        if self.current_player == 1:
            # permute for symmetry
            return np.eye(3)[self.board][:, [0, 2, 1]].reshape(-1)

    def reset(self):
        self.current_player = 0
        self.board = np.zeros(9, dtype="int")
        return self._one_hot_board()

    def step(self, actions):
        exp = {"state": "in progress"}

        # get the current player's action
        action = actions

        reward = 0
        done = False
        # illegal move
        if self.board[action] != 0:
            reward = -10  # illegal moves are really bad
            exp = {"state": "done", "reason": "Illegal move"}
            done = True
            self.summary["total games"] += 1
            self.summary["illegal moves"] += 1
            return self._one_hot_board(), reward, done, exp

        self.board[action] = self.current_player + 1

        # check if the other player can win on the next turn:
        for streak in self.winning_streaks:
            if ((self.board[streak] == 2 - self.current_player).sum() >= 2) and (
                self.board[streak] == 0
            ).any():
                reward = -2
                exp = {
                    "state": "in progress",
                    "reason": "Player {} can lose on the next turn".format(
                        self.current_player
                    ),
                }

        # check if we won
        for streak in self.winning_streaks:
            if (self.board[streak] == self.current_player + 1).all():
                reward = 1  # player wins!
                exp = {
                    "state": "in progress",
                    "reason": "Player {} has won".format(self.current_player),
                }
                self.summary["total games"] += 1
                self.summary["player {} wins".format(self.current_player)] += 1
                done = True
        # check if we tied, which ends the game
        if (self.board != 0).all():
            reward = 0
            exp = {
                "state": "in progress",
                "reason": "Player {} has tied".format(self.current_player),
            }
            done = True
            self.summary["total games"] += 1
            self.summary["ties"] += 1

        # move to the next player
        self.current_player = 1 - self.current_player

        return self._one_hot_board(), reward, done, exp

    def render(self, mode: str = "human"):
        print("{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}".format(*self.board.tolist()))
