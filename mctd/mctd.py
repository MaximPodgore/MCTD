import sys
import os
import numbers
import copy
from copy import deepcopy
import warnings
import random
import math
from abc import abstractmethod
import json
import time
import pickle
import numpy as np

from .utils import SearchNodeABS, RealVector, TreeNode

class MCTD:
    def __init__(self,
                 dims = None,
                 lower_bound = None,
                 upper_bound = None,
                 objective_function = None,
    ):
        self.dims = dims
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.objective_function = objective_function
        return
    
    def descent(self):
        """
        This is the main function of the MCTD algorithm
        """
        sampleNotScaled = np.random.rand(self.dims)
        fnLb = fn(self.lower_bound);
        fnUb = fn(self.upper_bound);
        sampleScaled = self.lower_bound + sampleNotScaled * (fnUb - fnLb)
        firstVector = RealVector(
            y = sampleScaled,
            lb = self.lower_bound,
            ub = self.upper_bound,
        )
