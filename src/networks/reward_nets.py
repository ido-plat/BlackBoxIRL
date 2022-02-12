from imitation.rewards import reward_nets
import abc
from typing import Callable, Iterable, Sequence, Tuple
from torchvision import models
import gym
import torch as th
from torch import nn


class ClassificationRewardNet(reward_nets.RewardNet):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            n_classes,
            return_func=lambda x: x,
            use_state: bool = True,
            use_action: bool = False,
            use_next_state: bool = True,
            **kwargs,
    ):
        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        super().__init__(observation_space, action_space)
        self.return_func = return_func

        self.net = UnifiedResnet(use_state=use_state, use_action=use_action, n_classes=n_classes,
                                 use_next_state=use_next_state, **kwargs)

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:

        state = state if self.use_state else None
        action = action if self.use_action else None
        next_state = next_state if self.use_next_state else None
        out = self.net(state, action, next_state)
        return self.return_func(th.argmax(out, dim=1))


class ClassificationShapedRewardNet(reward_nets.ShapedRewardNet):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        n_classes: int,
        *,
        return_func: Callable = lambda x: x,
        use_state: bool = True,
        use_action: bool = False,
        use_next_state: bool = True,
        discount_factor: float = 0.99,
        **kwargs,
    ):
        base_reward_net = ClassificationRewardNet(
            observation_space=observation_space,
            action_space=action_space,
            n_classes=n_classes,
            return_func=return_func,
            use_state=use_state,
            use_action=use_action,
            use_next_state=use_next_state,
            **kwargs,
        )

        potential_net = PotentialResnet18(**kwargs)

        super().__init__(
            observation_space,
            action_space,
            base_reward_net,
            potential_net,
            discount_factor=discount_factor,
        )


class PotentialResnet18(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = UnifiedResnet(True, False, False, 1, **kwargs)

    def forward(self, state: th.Tensor) -> th.Tensor:
        return self.net(state)


class UnifiedResnet(nn.Module):
    def __init__(self, use_state, use_action, use_next_state, n_classes, num_actions=None,
                 input_channels=4, hid_dim=32, **kwargs):

        super().__init__()
        self.state_net = None
        self.n_classes = n_classes
        input_mlp = 0

        def make_resnet():
            net = models.resnet18(False)
            net.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            net.fc = nn.Linear(512, hid_dim)
            return net

        if use_state:
            input_mlp += hid_dim
            self.state_net = make_resnet()
        self.action_net = None
        if use_action:
            assert num_actions
            input_mlp += num_actions
            raise NotImplementedError

        self.next_state_net = None
        if use_next_state:
            input_mlp += hid_dim
            self.next_state_net = make_resnet()
        self.final_net = nn.Linear(input_mlp, n_classes)

    def forward(self, state=None, action=None, next_state=None):
        inputs = []
        if self.state_net:
            assert state is not None
            inputs.append(self.state_net(state))
        if self.action_net:
            assert action is not None
            inputs.append(self.action_net(action))
        if self.next_state_net:
            assert next_state is not None
            inputs.append(self.next_state_net(next_state))
        return self.final_net(th.cat(inputs, dim=1))

