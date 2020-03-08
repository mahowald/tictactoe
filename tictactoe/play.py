import os
import time
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import TicTacToe
from .model import Policy


def play():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TicTacToe()
    model = load_model("pytorch_dqn.pt", device)
    
    done = False
    obs = env.reset()
    exp = {}
    
    player = 0
    while not done:
        os.system("clear")
        print("Commands:\n{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}\n\nBoard:".format(*[x for x in range(0, 9)]))
        
        env.render()
        
        action = None
        
        if player == 1:
            action = int(input())
        else:
            time.sleep(1)
            action = act(model, torch.tensor([obs], dtype=torch.float).to(device)).item()
            
        obs, _, done, exp = env.step(action)
        player = 1 - player
    
    os.system("clear")
    print("Commands:\n{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}\n\nBoard:".format(*[x for x in range(0, 9)]))
    env.render()
    print(exp)
    if "tied" in exp["reason"]:
        print("A strange game. The only winning move is not to play.")
    exit(0)


def load_model(path: str, device: torch.device):
    model = Policy(n_inputs=3*9, n_outputs=9).to(device)
    model_state_dict = torch.load(path, map_location=device)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

def act(model: Policy, state: torch.tensor):
    with torch.no_grad():
        p = F.softmax(model.forward(state)).cpu().numpy()
        valid_moves = (state.cpu().numpy().reshape(3,3,3).argmax(axis=2).reshape(-1) == 0)
        p = valid_moves*p
        return p.argmax()