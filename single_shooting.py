import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from casadi import *
from casadi.tools import *

def single_shooting():
    T = 10. # Time horizon
    N = 20 # number of control intervals Based on this number you get the amount of u's

    # System Parameters
    E0 = 5  # MX.sym("E0")
    vm = 10  # MX.sym("vm")
    vA = 0.5  # 1 # MX.sym("vA")
    vf = 0.1  # MX.sym("vf")
    voff = np.pi  # MX.sym("voff")
    c = 0.028  # MX.sym("c")
    beta = 0  # MX.sym("beta")
    rho = 1  # MX.sym("rho")
    L = 300  # MX.sym("l")
    A = 160  # MX.sym("A")

    # States and control variables
    nx = 3
    nu = 1

    x = SX.sym("x", nx, 1)
    u = SX.sym("u", nu, 1)
    u_old = SX.sym("u_old", nu, 1)
    t = SX.sym("t")

    # Equations
    v0 = vm + vA * sin(2 * np.pi * vf * t + voff)
    v0_fcn = Function('v0_fcn', [t], [v0])
    E = E0 - c * (u ** 2)
    E_fcn = Function('E_fcn', [u], [E])
    va = v0_fcn(t) * E * cos(x[0])
    va_fcn = Function('va_fcn', [x, t], [va])
    PD = rho * (v0_fcn(t) ** 2) / 2
    PD_fcn = Function('PD_fcn', [t], [PD])
    TF = PD_fcn(t) * A * (cos(x[0]) ** 2) * (E_fcn(u) + 1) * np.sqrt(E_fcn(u) ** 2 + 1) * (
                cos(x[0]) * cos(beta) + sin(x[0]) * sin(beta) * sin(x[1]))
    tension = Function('tension', [x, u, t], [TF])

    height = L * cos(x[1]) * sin(x[0])
    height_fcn = Function('height_fcn', [x], [height])

    xdot = vertcat((va_fcn(x, t) / L) * (cos(x[2]) - tan(x[0]) / E_fcn(u)),
                   -va_fcn(x, t) * sin(x[2]) / (L * sin(x[0])),
                   va_fcn(x, t) * u / L - cos(x[0]) * (va_fcn(x, t) * sin(x[2]) / (L * sin(x[0]))))

    # Stage cost (objective term to be minimized)
    wF = 1e-4
    wu = 0.5

    ### FIX THIS #####
    stage_cost = -wF * tension(x, u, t) + wu * ((u - u_old) ** 2)
    stage_cost_fcn = Function('stage_cost_fcn', [x, u, t, u_old], [stage_cost])

    # CVODES from the SUNDIALS suite
    dae = {'x':x, 'p':u, 'ode':xdot, 'quad':stage_cost}
    opts = {'tf':T/N}
    F = integrator('F', 'cvodes', dae, opts)

    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    # Formulate the NLP
    #Xk = MX([0, 1]) initializes the states
    # initialize
    #x_0 = np.array([np.pi / 4, np.pi / 4, 0]).reshape(nx, 1)
    x_0 = MX([np.pi / 4, np.pi / 4, 0])
    t_k = np.array(np.linspace(0, (N_sim + N) * dt, N_sim + N + 1))
    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [-1]
        ubw += [1]
        w0 += [0]

        # Integrate till the end of the interval
        Fk = F(x0=x_0, p=Uk)
        Xk = Fk['xf']
        J=J+Fk['qf']

        # Add inequality constraint
        g += [Xk[0]]
        lbg += [-.25]
        ubg += [inf]

    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', prob);

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x']

    # Plot the solution
    u_opt = w_opt
    x_opt = [[0, 1]]
    for k in range(N):
        Fk = F(x0=x_opt[-1], p=u_opt[k])
        x_opt += [Fk['xf'].full()]
    x1_opt = [r[0] for r in x_opt]
    x2_opt = [r[1] for r in x_opt]

    tgrid = [T/N*k for k in range(N+1)]

    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, x1_opt, '--')
    plt.plot(tgrid, x2_opt, '-')
    plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-.')
    plt.xlabel('t')
    plt.legend(['x1','x2','u'])
    plt.grid()
    plt.show()