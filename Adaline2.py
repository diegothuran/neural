import numpy as np
import matplotlib.pyplot as plt
import math

LEARNING_RATE = 0.01

def activation(x):
    if x > 0:
        return 1
    else:
        return 0