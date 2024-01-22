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
def next_polytope(poly_j, poly_kappa_f, a_closed_loop):
    """
    Function:
        calculate the next polytope
    Inputs:
        poly_j        : the previous polytope
        poly_kappa_f  : the initial poly kappa f
        a_closed_loop : \dot{x}=Ax+B@Ku
    Returns:
        pc.Polytope(Hnext, bnext) : the next polytope
    """
    (Hj, bj) = (poly_j.A, poly_j.b)
    (Hkf, bkf) = (poly_kappa_f.A, poly_kappa_f.b)
    Hnext = np.vstack((Hkf, Hj @ a_closed_loop))
    bnext = np.concatenate((bkf, bj))
    return pc.Polytope(Hnext, bnext)


def determine_maximal_invariant_set(poly_kappa_f, a_closed_loop):
    """
    Function:
        determine the maximal invariant set
    Inputs:
        poly_kappa_f  : the initial poly kappa f
        a_closed_loop : \dot{x}=Ax+B@Ku
    Returns:
        inv_next      : the maximal invariant set
    """
    
    inv_prev = poly_kappa_f  # use the initial poly kappa f as the previous one before the loop
    keep_running = True
    while keep_running:  # loop to calculate the maximal set
        inv_next = next_polytope(inv_prev, poly_kappa_f, a_closed_loop)  # calculate the next one
        inv_next = pc.reduce(inv_next)
        keep_running = inv_next >= inv_prev  # if next one >= previous one, continue
        inv_prev = inv_next
    return inv_next



## Impliment and test controller.

def mpc_control(x_current, P, Q, R, A, B, G, h, x_min, x_max, N):
    """
    MPC control algorithm.

    :param x_current: Current state of the system.
    :param P, Q, R: Weight matrices for the cost function.
    :param A, B: System dynamics matrices.
    :param G, h: Inequality constraint matrices.
    :param x_min, x_max: State constraints.
    :param N: Prediction horizon.
    :return: Optimal control input.
    """
# Number of states and controls
    nx = A.shape[1]
    nu = B.shape[1]

    # Extended weight matrices
    P_ext = np.kron(np.eye(N), P)
    Q_ext = np.kron(np.eye(N), Q)
    R_ext = np.kron(np.eye(N-1), R)

    # Extended dynamics matrices
    A_ext = np.zeros((N*nx, nx))
    B_ext = np.zeros((N*nx, N*nu))
    for i in range(N):
        A_ext[i*nx:(i+1)*nx, :] = np.linalg.matrix_power(A, i+1)
        for j in range(i+1):
            B_ext[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = np.linalg.matrix_power(A, i-j) @ B

    # Cost function parameters
    H = 2 * (B_ext.T @ P_ext @ B_ext + R_ext)
    f = 2 * B_ext.T @ P_ext @ A_ext @ x_current

    # Constraints
    G_ext = np.kron(np.eye(N), G)
    h_ext = np.tile(h, N)

    # Solve QP problem
    u = solve_qp(H, f, G_ext, h_ext)
    return u[:nu]  # Return the first control input

# Define system matrices
A = np.array([[0.9835, 2.782, 0], [-0.0006821, 0.978, 0], [-0.0009730, 2.804, 1]])
B = np.array([[0.01293], [0.00100], [0.001425]])

# Define constraints
G = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
h = np.array([11.5, 11.5, 27, 24, 35, 35])  # degrees

# Define state and control limits
x_min, x_max = -np.array([11.5, 14, 35]), np.array([11.5, 14, 35])

# Weight matrices
P = np.eye(3)  # State weight
Q = np.eye(3)  # State weight over horizon
R = np.eye(1)  # Control weight

# Prediction horizon
N = 10

# Current state
x_current = np.array([0, 0, 0])  # Example state

# Compute control input
u_optimal = mpc_control(x_current, P, Q, R, A, B, G, h, x_min, x_max, N)


## Performance analysis.


