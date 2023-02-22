#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import numpy as np
import math

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   sampled a random list from the provided list.                             #
#                                                                             #
#*****************************************************************************#
def random_sample(input_list, sample_fraction):
    """A generic function to randly sample a fraction of elements from a list"""
    # find number of items to sample
    n_samples = int(math.ceil(len(input_list) * sample_fraction))
    # return random picks
    return np.random.permutation(input_list)[:n_samples]