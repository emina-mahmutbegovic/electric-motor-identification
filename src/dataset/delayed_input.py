# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited

from enum import Enum
import numpy as np

class Regressor(Enum):
    NONE = 0
    SINGLE_INPUT = 1
    DOUBLE_INPUT = 2
    THREE_FIVE = 3
    SEVEN_ELEVEN = 4
    SIX_EIGHT_FIFTEEN = 5
    TWO_TEN_TWELVE = 6
    THREE_FOUR_FIVE = 7
    TWO_THREE_FOUR_FIVE = 8


class DelayedInput:
    def __init__(self, regressor, inputs, targets):
        id_k = [sublist[0] for sublist in inputs]
        iq_k = [sublist[1] for sublist in inputs]

        if regressor == Regressor.SINGLE_INPUT:
            # Create lagged variables
            id_k_lag = id_k[:-1]
            iq_k_lag = iq_k[:-1]

            self.inputs = np.array([[a1, a2, a3, a4] for a1, a2, a3, a4 in zip(id_k[1:], iq_k[1:], id_k_lag, iq_k_lag)])
            self.targets = targets[1:]
        elif regressor == Regressor.DOUBLE_INPUT:
            # Create lagged variables
            id_k_lag = id_k[1:-1]
            iq_k_lag = iq_k[1:-1]
            id_k_lag2 = id_k[:-2]
            iq_k_lag2 = iq_k[:-2]

            self.inputs = np.array([[a1, a2, a3, a4, a5, a6] for a1, a2, a3, a4, a5, a6 in zip(id_k[2:], iq_k[2:],
                                             id_k_lag, iq_k_lag, id_k_lag2, iq_k_lag2)])
            self.targets = targets[2:]
        elif regressor == Regressor.THREE_FIVE:
            id_k_lag = id_k[:-3]
            iq_k_lag = iq_k[:-3]
            id_k5_lag = id_k[:-5]
            iq_k5_lag = iq_k[:-5]

            self.inputs = np.array([[a1, a2, a3, a4, a5, a6] for a1, a2, a3, a4, a5, a6 in zip(id_k[5:], iq_k[5:], 
                                            id_k_lag, iq_k_lag, id_k5_lag, iq_k5_lag)])
            self.targets = targets[5:]
        elif regressor == Regressor.SEVEN_ELEVEN:
            id_k7_lag = id_k[:-7]
            iq_k7_lag = iq_k[:-7]
            id_k11_lag = id_k[:-11]
            iq_k11_lag = iq_k[:-11]

            self.inputs = np.array([[a1, a2, a3, a4, a5, a6] for a1, a2, a3, a4, a5, a6 in zip(id_k[11:], iq_k[11:], 
                                            id_k7_lag, iq_k7_lag, id_k11_lag, iq_k11_lag)])
            self.targets = targets[11:]
        elif regressor == Regressor.SIX_EIGHT_FIFTEEN:
            id_k6_lag = id_k[:-6]
            iq_k6_lag = iq_k[:-6]
            id_k8_lag = id_k[:-8]
            iq_k8_lag = iq_k[:-8]
            id_k15_lag = id_k[:-15]
            iq_k15_lag = iq_k[:-15]

            self.inputs = np.array([[a1, a2, a3, a4, a5, a6, a7, a8] for a1, a2, a3, a4, a5, a6, a7, a8 in zip(id_k[15:], iq_k[15:], 
                                            id_k6_lag, iq_k6_lag, id_k8_lag, iq_k8_lag, id_k15_lag, iq_k15_lag)])
            self.targets = targets[15:]
        elif regressor == Regressor.TWO_TEN_TWELVE:
            id_k2_lag = id_k[:-2]
            iq_k2_lag = iq_k[:-2]
            id_k10_lag = id_k[:-10]
            iq_k10_lag = iq_k[:-10]
            id_k20_lag = id_k[:-20]
            iq_k20_lag = iq_k[:-20]

            self.inputs = np.array([[a1, a2, a3, a4, a5, a6, a7, a8] for a1, a2, a3, a4, a5, a6, a7, a8 in zip(id_k[20:], iq_k[20:], 
                                            id_k2_lag, iq_k2_lag, id_k10_lag, iq_k10_lag, id_k20_lag, iq_k20_lag)])
            self.targets = targets[20:]
        elif regressor == Regressor.THREE_FOUR_FIVE:
            id_k3_lag = id_k[:-3]
            iq_k3_lag = iq_k[:-3]
            id_k4_lag = id_k[:-4]
            iq_k4_lag = iq_k[:-4]
            id_k5_lag = id_k[:-5]
            iq_k5_lag = iq_k[:-5]

            self.inputs = np.array([[a1, a2, a3, a4, a5, a6, a7, a8] for a1, a2, a3, a4, a5, a6, a7, a8 in zip(id_k[5:], iq_k[5:], 
                                            id_k3_lag, iq_k3_lag, id_k4_lag, iq_k4_lag, id_k5_lag, iq_k5_lag)])
            self.targets = targets[5:]
        elif regressor == Regressor.TWO_THREE_FOUR_FIVE:
            id_k2_lag = id_k[:-2]
            iq_k2_lag = iq_k[:-2]
            id_k3_lag = id_k[:-3]
            iq_k3_lag = iq_k[:-3]
            id_k4_lag = id_k[:-4]
            iq_k4_lag = iq_k[:-4]
            id_k5_lag = id_k[:-5]
            iq_k5_lag = iq_k[:-5]

            self.inputs = np.array([[a1, a2, a3, a4, a5, a6, a7, a8, a9, a10] for a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 in zip(id_k[5:], iq_k[5:], 
                                            id_k2_lag, iq_k2_lag, id_k3_lag, iq_k3_lag, id_k4_lag, iq_k4_lag, id_k5_lag, iq_k5_lag)])
            self.targets = targets[5:]
        elif regressor == Regressor.NONE:
            self.inputs = inputs
            self.targets = targets