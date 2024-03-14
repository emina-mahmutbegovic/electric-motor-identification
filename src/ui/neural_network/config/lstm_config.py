# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited

from src.ui.neural_network.config.neural_network_config_base import NeuralNetworkConfigBase

class LSTMConfig(NeuralNetworkConfigBase):
    def __init__(self):
        super().__init__()

        self.num_of_units = None
        self.num_of_units_edit = None
        self.activation_function = None
        self.look_back_edit = None