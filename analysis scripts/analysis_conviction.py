# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:38:22 2021

look at communicating player conviction results

demonstrate learning of the conviction box

@author: Jasper
"""

from axelrod.action import Action
C, D = Action.C, Action.D

import numpy as np
import torch
import scipy
import pandas as pd
import matplotlib.pyplot as plt