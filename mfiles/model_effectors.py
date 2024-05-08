#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from math import sqrt
import numpy as np


def twoD_reach_velocity( model, r1, currstate, perturbstate, dt=1 ):
    ''' effector module - can be replaced? 
    - readout maps onto 2d velocity, that updates cursor/limb x-y (2D) position linearly
    '''
    velocity = model.readout( r1 )
    newstate = velocity*(dt/1000) + currstate  + perturbstate
    return newstate
