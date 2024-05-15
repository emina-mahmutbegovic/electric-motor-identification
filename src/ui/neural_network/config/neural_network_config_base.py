# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited

class NeuralNetworkConfigBase:
    def __init__(self):
        self.loss_combo = None
        self.optimizer_combo = None
        self.metrics_combo = None
        self.regressor_combo = None

        self.epochs_edit = None
        self.batch_size_edit = None

        self.epochs = None
        self.batch_size = None
