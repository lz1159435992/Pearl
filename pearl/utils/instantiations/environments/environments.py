# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from pearl.api.observation import Observation
from pearl.api.space import Space
from pearl.utils.instantiations.spaces.discrete import DiscreteSpace

from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym  # noqa

import torch
import torch.nn.functional as F
from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.environment import Environment


class FixedNumberOfStepsEnvironment(Environment):
    def __init__(self, number_of_steps: int = 100) -> None:
        self.number_of_steps_so_far = 0
        self.number_of_steps: int = number_of_steps
        self._action_space = DiscreteActionSpace(
            [torch.tensor(True), torch.tensor(False)]
        )

    def step(self, action: Action) -> ActionResult:
        self.number_of_steps_so_far += 1
        return ActionResult(
            observation=self.number_of_steps_so_far,
            reward=self.number_of_steps_so_far,
            terminated=True,
            truncated=True,
            info={},
        )

    def render(self) -> None:
        print(self.number_of_steps_so_far)

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    def reset(self, seed: Optional[int] = None) -> Tuple[Observation, ActionSpace]:
        return self.number_of_steps_so_far, self.action_space

    def __str__(self) -> str:
        return type(self).__name__


class ObservationTransformationEnvironmentAdapterBase(Environment, ABC):
    """
    A base for environment adapters tranforming observations.
    """

    def __init__(
        self,
        base_environment: Environment,
    ) -> None:
        self.base_environment = base_environment
        self.observation_space: Space = self.make_observation_space(base_environment)

    @staticmethod
    @abstractmethod
    def make_observation_space(base_environment: Environment) -> Space:
        pass

    @abstractmethod
    def compute_tensor_observation(self, observation: Observation) -> torch.Tensor:
        pass

    @property
    def action_space(self) -> ActionSpace:
        return self.base_environment.action_space

    def step(self, action: Action) -> ActionResult:
        action_result = self.base_environment.step(action)
        action_result.observation = self.compute_tensor_observation(
            action_result.observation
        )
        return action_result

    def reset(self, seed: Optional[int] = None) -> Tuple[Observation, ActionSpace]:
        observation, action_space = self.base_environment.reset(seed=seed)
        return self.compute_tensor_observation(observation), action_space

    def __str__(self) -> str:
        return f"{self.short_description} from {self.base_environment}"

    @property
    def short_description(self) -> str:
        return self.__class__.__name__


class OneHotObservationsFromDiscrete(ObservationTransformationEnvironmentAdapterBase):
    """
    An environment adapter mapping a Discrete observation space into
    a Box observation space with dimension 1
    where the observation is a one-hot vector.

    This is useful to use with agents expecting tensor observations.
    """

    def __init__(self, base_environment: Environment) -> None:
        super(OneHotObservationsFromDiscrete, self).__init__(base_environment)

    @staticmethod
    def make_observation_space(base_environment: Environment) -> Space:
        # pyre-fixme: need to add `observation_space` property in Environment
        # and implement it in all concrete subclasses
        assert isinstance(base_environment.observation_space, DiscreteSpace)
        n = base_environment.observation_space.n
        elements = [F.one_hot(torch.tensor(i), n).float() for i in range(n)]
        return DiscreteSpace(elements)

    def compute_tensor_observation(self, observation: Observation) -> torch.Tensor:
        if isinstance(observation, torch.Tensor):
            observation_tensor = observation
        else:
            observation_tensor = torch.tensor(observation)
        # pyre-fixme: need to add `observation_space` property in Environment
        # and implement it in all concrete subclasses
        assert isinstance(self.base_environment.observation_space, DiscreteSpace)
        return F.one_hot(
            observation_tensor,
            self.base_environment.observation_space.n,
        ).float()

    @property
    def short_description(self) -> str:
        return f"One-hot observations on {self.base_environment}"
