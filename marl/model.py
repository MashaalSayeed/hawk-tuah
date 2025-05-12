import torch
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from gymnasium.spaces import Box
from collections import OrderedDict


class CentralizedCriticModel(TorchModelV2, nn.Module):
    """
    Custom model for CTDE (Centralized Training with Decentralized Execution).
    This model provides:
    1. A regular policy network (actor) that uses only the agent's observations
    2. A centralized value function (critic) that uses global state information
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.use_centralized_critic = model_config.get("custom_model_config", {}).get(
            "use_centralized_critic", True
        )
        self.use_obs_before_centralizing = model_config.get("custom_model_config", {}).get(
            "use_obs_before_centralizing", True
        )

        # Create the policy network (actor)
        self.actor = FullyConnectedNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_actor"
        )

        # If using centralized critic
        if self.use_centralized_critic:
            # For the critic, we'll use the global state passed in the info dict
            # Determine input size for critic based on state dimension
            if self.use_obs_before_centralizing:
                # If including observation before centralizing
                critic_input_size = obs_space.shape[0]
            else:
                # Get the state dimension from the environment
                critic_input_size = 444  # Default size if we can't determine from config
                if hasattr(model_config, "get") and model_config.get("custom_model_config"):
                    if "state_dim" in model_config["custom_model_config"]:
                        critic_input_size = model_config["custom_model_config"]["state_dim"]

            # Create a separate state space for the critic
            critic_obs_space = Box(
                low=-np.inf,
                high=np.inf, 
                shape=(critic_input_size,),
                dtype=np.float32
            )

            # Create the centralized value network (critic)
            self.critic = FullyConnectedNetwork(
                critic_obs_space,
                action_space,  # Dummy, not used
                1,  # Output one value
                model_config,
                name + "_critic"
            )

        # Hold the last value output for the value function
        self._cur_value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        The forward pass for the actor (policy) network
        """
        # Standard policy network forward pass
        policy_features, _ = self.actor(input_dict, state, seq_lens)
        return policy_features, state

    @override(TorchModelV2)
    def value_function(self):
        """
        Return the current value function estimate
        """
        return self._cur_value

    def central_value_function(self, obs, state):
        """
        Forward pass for the centralized value function using global state
        """
        if not self.use_centralized_critic:
            # If centralized critic is disabled, use the regular value function
            return self.value_function()

        # Process the global state through the critic network
        input_dict = {"obs": state}
        print("stare", state)
        central_value, _ = self.critic(input_dict, [], None)
        return central_value

    def custom_loss(self, policy_loss, loss_inputs):
        """
        Custom loss function that adds centralized value function loss
        """
        if not self.use_centralized_critic:
            # If not using centralized critic, just return the original loss
            return policy_loss

        # Extract the required fields from the input
        actions = loss_inputs["actions"]
        rewards = loss_inputs["rewards"]
        state = loss_inputs["state"]  # Global state
        prev_actions = loss_inputs.get("prev_actions", None)
        prev_rewards = loss_inputs.get("prev_rewards", None)

        # Forward pass for the centralized value function
        vf_preds = self.central_value_function(None, state)

        # Compute value function loss
        if "value_targets" in loss_inputs:
            value_targets = loss_inputs["value_targets"]
            vf_loss = torch.mean(torch.pow(value_targets - vf_preds, 2.0))
        else:
            # If value targets aren't provided, we can't compute the loss
            vf_loss = torch.tensor(0.0)

        # Combine policy and value function losses
        total_loss = policy_loss + vf_loss * self.model_config.get("vf_loss_coeff", 1.0)
        
        return total_loss