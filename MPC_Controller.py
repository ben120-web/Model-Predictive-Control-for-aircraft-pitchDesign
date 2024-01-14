# This is a model predictive controller for the pitch of an aircraft
import numpy as np
import qpsolvers as qp
import polytope as pl
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg as spla

# Set system requirements.
max_angle_of_attack = 11.5 # Degrees
deflection_angle_range = [-24, 27]
pitch_angle_max = 35
pitch_angle_rate_max = 14
slope_max = 23

# Define the dynamical system
# -----------------------------
n = 2       # 2 states
m = 1       # 1 input
A = np.array([[0.9835, 2.782, 0], [-0.0006821, 0.978, 0], [-0.0009730, 2.804, 1]])
B = np.array([[0.01293], [0.001], [0.001425]])
N = 10


## Formulate the MPC problem.

# Define the objective function reflecting the goals of the controller.

## Use QP solvers to solve optimisation problem.

## Compute the maximal invariant set with polytope.
# This set defines the safe operating region for the system.

## Impliment and test controller.

## Performance analysis.


