# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import io
import os

from typing import List, Optional

import torch

from pearl.api.environment import Environment
from pearl.pearl_agent import PearlAgent
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.utils.functional_utils.requests_get import requests_get
from pearl.utils.functional_utils.train_and_eval.online_learning import run_episode
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


def is_file_readable(file_path: str) -> bool:
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)


def get_offline_data_in_buffer(
    is_action_continuous: bool,
    url: Optional[str] = None,
    data_path: Optional[str] = None,
    max_number_actions_if_discrete: Optional[int] = None,
    size: int = 1000000,
    device: str = "cpu",
) -> ReplayBuffer:
    """
    Fetches offline data from a url and returns a replay buffer which can be sampled
    to train the offline agent. For this implementation we use FIFOOffPolicyReplayBuffer.

    - Assumes the offline data is an iterable consisting of transition tuples
        (observation, action, reward, next_observation, curr_available_actions,
        next_available_actions, terminated) as dictionaries.

    - Also assumes offline data is in a .pt file; reading from a
        csv file can also be added later.

    Args:
        is_action_continuous (bool): whether the action space is continuous or discrete.
            for continuous actions spaces, we need to set this flag; see 'push' method
            in FIFOOffPolicyReplayBuffer class
        url (str, optional): from where offline data needs to be fetched from
        data_path (str, optional): local path to the offline data
        max_number_actions_if_discrete (int, optional): To work with a discrete action space in
            Pearl, each transition tuple requires specifying the maximum number of actions in the
            action space. For offline learning, we expect users to know the maximum number of
            actions in the discrete action space.
            For continuous action spaces, we do not need to specify this parameter. Hence it is
            optional and defaults to None.
        size (int): Size of the replay buffer. Defaults to 1000000.
        device (cpu): Device to load the data onto. If no device is specified, defaults to cpu.

    Returns:
        ReplayBuffer: a FIFOOffPolicyReplayBuffer containing offline data of transition tuples.
            The transition tuples are in the format as expected by a Pearl agent.
    """

    if is_action_continuous:
        if max_number_actions_if_discrete is not None:
            raise ValueError(
                "get_offline_data: is_action_continuous = True requires \
                max_number_actions to be None"
            )
    elif max_number_actions_if_discrete is None:
        raise ValueError(
            "get_offline_data: is_action_continuous = False requires \
            max_number_actions to be an integer value"
        )

    if url is not None:
        offline_transitions_data = requests_get(url)
        stream = io.BytesIO(offline_transitions_data.content)  # implements seek()
        raw_transitions_buffer = torch.load(stream, map_location=torch.device(device))
    else:
        if data_path is None:
            raise ValueError(
                "provide either a data_path or a url to fetch offline data"
            )

        # loads data on the specified device
        raw_transitions_buffer = torch.load(
            data_path, map_location=torch.device(device)
        )

    offline_data_replay_buffer = FIFOOffPolicyReplayBuffer(size)
    if is_action_continuous:
        offline_data_replay_buffer._is_action_continuous = True

    for transition in raw_transitions_buffer:
        if transition["curr_available_actions"].__class__.__name__ == "Discrete":
            transition["curr_available_actions"] = DiscreteActionSpace(
                actions=list(
                    torch.arange(transition["curr_available_actions"].n).view(-1, 1)
                )
            )
        if transition["next_available_actions"].__class__.__name__ == "Discrete":
            transition["next_available_actions"] = DiscreteActionSpace(
                actions=list(
                    torch.arange(transition["next_available_actions"].n).view(-1, 1)
                )
            )

        offline_data_replay_buffer.push(
            state=transition["observation"],
            action=transition["action"],
            reward=transition["reward"],
            next_state=transition["next_observation"],
            curr_available_actions=transition["curr_available_actions"],
            next_available_actions=transition["next_available_actions"],
            terminated=transition["done"],
            max_number_actions=max_number_actions_if_discrete,
        )

    return offline_data_replay_buffer


def offline_learning(
    offline_agent: PearlAgent,
    data_buffer: ReplayBuffer,
    training_epochs: int = 1000,
    seed: int = 100,
) -> None:
    """
    Trains the offline agent using transition tuples from offline data (provided in
    the data_buffer). Must provide a replay buffer with transition tuples - please
    use the method get_offline_data_in_buffer to create an offline data buffer.

    Args:
        offline agent: a conservative learning agent (CQL or IQL).
        data_buffer: a replay buffer to sample a batch of transition data.
        training_epochs: number of training epochs for offline learning.
    """
    set_seed(seed=seed)

    # move replay buffer to device of the offline agent
    data_buffer.device_for_batches = offline_agent.device

    # training loop
    for i in range(training_epochs):
        batch = data_buffer.sample(offline_agent.policy_learner.batch_size)
        assert isinstance(batch, TransitionBatch)
        loss = offline_agent.learn_batch(batch=batch)
        if i % 500 == 0:
            print("training epoch", i, "training loss", loss)


def offline_evaluation(
    offline_agent: PearlAgent,
    env: Environment,
    number_of_episodes: int = 1000,
    seed: Optional[int] = None,
) -> List[float]:
    """
    Evaluates the performance of an offline trained agent.

    Args:
        agent: the offline trained agent.
        env: the environment to evaluate the agent in
        number_of_episodes: the number of episodes to evaluate for.
    Returns:
        returns_offline_agent: a list of returns for each evaluation episode.
    """

    # sanity check: during offline evaluation, the agent should not learn or explore.
    learn = False
    exploit = True
    learn_after_episode = False

    returns_offline_agent = []
    for i in range(number_of_episodes):
        evaluation_seed = seed + i if seed is not None else None
        episode_info, total_steps = run_episode(
            agent=offline_agent,
            env=env,
            learn=learn,
            exploit=exploit,
            learn_after_episode=learn_after_episode,
            seed=evaluation_seed,
        )
        g = episode_info["return"]
        if i % 1 == 0:
            print(f"\repisode {i}, return={g}", end="")
        returns_offline_agent.append(g)

    return returns_offline_agent
