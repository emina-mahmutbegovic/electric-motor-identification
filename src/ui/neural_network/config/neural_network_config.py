# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited

from src.ui.neural_network.config.neural_network_config_base import NeuralNetworkConfigBase

class NeuralNetworkConfig(NeuralNetworkConfigBase):
    def __init__(self):
        super().__init__()

        self.output_layer_activation_combo = None
        self.hidden_layer_activation_combo = None
        self.hidden_layer_arch_edit = None
