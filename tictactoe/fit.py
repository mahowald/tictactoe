from .env import TicTacToe
from .model import Policy, Transition, ReplayMemory

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import random
import logging
import io

def fit(
    n_steps: int = 500_000,
    batch_size: int = 128,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.1,
    eps_steps: int = 200_000,
) -> bytes:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Beginning training on: {}".format(device))

    target_update = int((1e-2) * n_steps)
    policy = Policy(n_inputs=3 * 9, n_outputs=9).to(device)
    target = Policy(n_inputs=3 * 9, n_outputs=9).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    memory = ReplayMemory(50_000)

    env = TicTacToe()
    state = torch.tensor([env.reset()], dtype=torch.float).to(device)
    old_summary = {
        "total games": 0,
        "ties": 0,
        "illegal moves": 0,
        "player 0 wins": 0,
        "player 1 wins": 0,
    }
    _randoms = 0
    summaries = []

    for step in range(n_steps):
        t = np.clip(step / eps_steps, 0, 1)
        eps = (1 - t) * eps_start + t * eps_end

        action, was_random = select_model_action(device, policy, state, eps)
        if was_random:
            _randoms += 1
        next_state, reward, done, _ = env.step(action.item())

        # player 2 goes
        if not done:
            next_state, _, done, _ = env.step(select_dummy_action(next_state))
            next_state = torch.tensor([next_state], dtype=torch.float).to(device)
        if done:
            next_state = None

        memory.push(state, action, next_state, torch.tensor([reward], device=device))

        state = next_state
        optimize_model(
            device=device,
            optimizer=optimizer,
            policy=policy,
            target=target,
            memory=memory,
            batch_size=batch_size,
            gamma=gamma,
        )
        if done:
            state = torch.tensor([env.reset()], dtype=torch.float).to(device)
        if step % target_update == 0:
            target.load_state_dict(policy.state_dict())
        if step % 5000 == 0:
            delta_summary = {k: env.summary[k] - old_summary[k] for k in env.summary}
            delta_summary["random actions"] = _randoms
            old_summary = {k: env.summary[k] for k in env.summary}
            logging.info("{} : {}".format(step, delta_summary))
            summaries.append(delta_summary)
            _randoms = 0

    logging.info("Complete")

    res = io.BytesIO()
    torch.save(policy.state_dict(), res)

    return res.getbuffer()


def optimize_model(
    device: torch.device,
    optimizer: optim.Optimizer,
    policy: Policy,
    target: Policy,
    memory: ReplayMemory,
    batch_size: int,
    gamma: float,
):
    """Model optimization step, copied verbatim from the Torch DQN tutorial.
    
    Arguments:
        device {torch.device} -- Device
        optimizer {torch.optim.Optimizer} -- Optimizer
        policy {Policy} -- Policy module
        target {Policy} -- Target module
        memory {ReplayMemory} -- Replay memory
        batch_size {int} -- Number of observations to use per batch step
        gamma {float} -- Reward discount factor
    """
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def select_dummy_action(state: np.array) -> int:
    """Select a random (valid) move, given a board state.
    
    Arguments:
        state {np.array} -- Board state observation
    
    Returns:
        int -- Move to make.
    """
    state = state.reshape(3, 3, 3)
    open_spots = state[:, :, 0].reshape(-1)
    p = open_spots / open_spots.sum()
    return np.random.choice(np.arange(9), p=p)


def select_model_action(
    device: torch.device, model: Policy, state: torch.tensor, eps: float
) -> Tuple[torch.tensor, bool]:
    """Selects an action for the model: either using the policy, or
    by choosing a random valid action (as controlled by `eps`)
    
    Arguments:
        device {torch.device} -- Device
        model {Policy} -- Policy module
        state {torch.tensor} -- Current board state, as a torch tensor
        eps {float} -- Probability of choosing a random state.
    
    Returns:
        Tuple[torch.tensor, bool] -- The action, and a bool indicating whether
                                     the action is random or not.
    """

    sample = random.random()
    if sample > eps:
        return model.act(state), False
    else:
        return (
            torch.tensor(
                [[random.randrange(0, 9)]],
                device=device,
                dtype=torch.long,
            ),
            True,
        )

