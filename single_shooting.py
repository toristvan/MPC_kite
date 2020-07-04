import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

from casadi.tools import *

#T = 10.  # Time horizon
#N = 5  # number of control intervals Based on this number you get the amount of u's
#dt = T/N
#N_sim = 80
#K=3, N=50, N_sim=200, dt=0.2, nx=3, nu=1, E0=5, vm=10, vA=0.5, vf=0.1, voff=np.pi, c=0.028, beta=0, rho=1, L=300, A=160, hmin=100, collocation_tech='legendre'
def single_shooting(N=5,N_sim=80, T=10):

    dt = T/N

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
    hmin = 100

    # States and control variables
    nx = 3
    nu = 1

    x = MX.sym("x", nx, 1)
    u = MX.sym("u", nu, 1)
    u_old = MX.sym("u_old", nu, 1)
    t = MX.sym("t")

    # Equations
    v0 = vm + vA * sin(2 * np.pi * vf * t + voff)
    v0_fcn = Function('v0_fcn', [t], [v0])
    E = E0 - c * (u ** 2)
    E_fcn = Function('E_fcn', [u], [E])
    va = v0_fcn(t) * E * cos(x[0])
    va_fcn = Function('va_fcn', [x, t], [va])
    PD = rho * (v0_fcn(t) ** 2) / 2
    PD_fcn = Function('PD_fcn', [t], [PD])
    TF = PD_fcn(t) * A * (cos(x[0]) ** 2) * (E_fcn(u) + 1) * sqrt(E_fcn(u) ** 2 + 1) * (
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

    stage_cost = -wF * tension(x, u, t) + wu * ((u - u_old) ** 2)
    stage_cost_fcn = Function('stage_cost_fcn', [x, u, t, u_old], [stage_cost])

    # CVODES from the SUNDIALS suite to construct the NLP
    dae = {'x': x, 'p': vertcat(u, t, u_old), 'ode': xdot, 'quad': stage_cost}
    opts = {'tf': dt}
    F = integrator('F', 'cvodes', dae, opts)

    # system integrator ODE
    ode = {'x': x, 'ode': xdot, 'p': vertcat(u, t)}
    opts = {'tf': dt}
    ode_solver = integrator('F', 'idas', ode, opts)

    # state constraints
    lb_x = -4 * np.ones((nx, 1))
    ub_x = 4 * np.ones((nx, 1))
    # input constraints
    lb_u = -1.5 * np.ones((nu, 1))
    ub_u = 1.5 * np.ones((nu, 1))

    #Xk = MX([np.pi / 4, np.pi / 4, 0])
    #Xk = MX.sym(nx, N+1)
    x_init = MX.sym('x_init', nx)
    #Xk = x_init

    #x_0 = opt_x['x', 0, 0] ####ENDRET SIST HER -> fiks start constraints
    #g.append(x_0 - x_init)

    # optimization variables and NLP
    U = MX.sym("U", N * nu, 1)
    tk = MX.sym('tk', N)

    J = 0
    u0 = []
    lb_X = []  # lower bound for X.
    ub_X = []  # upper bound for X
    lb_U = []  # lower bound for U
    ub_U = []  # upper bound for U
    g = []  # constraint expression g
    lb_g = []  # lower bound for constraint expression g
    ub_g = []  # upper bound for constraint expression g

    for k in range(N):

        #x_k = X[k * nx:(k + 1) * nx, :]
        #x_k_next = X[(k + 1) * nx:(k + 2) * nx, :]
        u_k = U[k * nu:(k + 1) * nu, :]
        u0 += [0]

        if k == 0:
            u_prev = U[k * nu:(k + 1) * nu, :]
            Xk = x_init
        else:
            u_prev = U[(k-1) * nu:k * nu, :]

        # # inequality constraints
        ineq = height_fcn(Xk)
        g.append(ineq)
        lb_g.append(hmin)
        ub_g.append(L)

        Fk = F(x0=Xk, p=vertcat(u_k, tk[k], u_prev))
        Xk = Fk['xf']
        J = J + Fk['qf']

        # state constraints
        #lb_g += [Xk[0]]
        #g += [Xk[0]]
        g += [Xk]
        ub_g += [ub_x]
        lb_g += [lb_x]

        #lb_X.append(lb_x)
        #ub_X.append(ub_x)
        lb_U.append(lb_u)
        ub_U.append(ub_u)

    # Create an NLP solver
    #Xk = reshape(Xk, nx*(N+1), 1)
    #print(vertcat(tk, Xk).shape)
    lbx = vertcat(*lb_U)
    ubx = vertcat(*ub_U)
    x = vertcat(U)
    g = vertcat(*g)
    lbg = vertcat(*lb_g)
    ubg = vertcat(*ub_g)

    pr = vertcat(tk, x_init)
    print(pr.shape)
    prob = {'f': J, 'x': x, 'g': g, 'p': vertcat(tk, x_init)}
    solver = nlpsol('solver', 'ipopt', prob)

    # Set number of iterations
    # N_sim = 10

    # Initialize result lists for states and inputs
    x_0 = np.array([np.pi / 4, np.pi / 4, 0]).reshape(nx, 1)
    t_k = np.array(np.linspace(0, (N_sim + N) * dt, N_sim + N + 1))
    res_x_mpc = [x_0]
    res_u_mpc = []
    costs = []
    u_curr = []
    u_plot = []
    solve_times = []

    for i in range(N_sim):
        start_time = datetime.now().timestamp()
        if i == 0:
            sol = solver(p=vertcat(t_k[i:N+i], x_0), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        else:
            sol = solver(p=vertcat(t_k[i:N+i], x_0), x0=u_curr, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        solve_times.append([datetime.now().timestamp() - start_time])
        u_curr = sol['x']
        u_k = u_curr[0]
        u_plot.append(u_k)

        # append cost
        cost_k = sol['f']
        costs.append(cost_k)

        # simulate the system
        res_integrator = ode_solver(x0=x_0, p=vertcat(u_k, t_k[i]))
        x_next = res_integrator['xf']

        # Update the initial state
        x_0 = x_next

        # Store the results
        res_x_mpc.append(x_next)
        res_u_mpc.append(u_k)

    res_x_mpc = np.concatenate(res_x_mpc, axis=1)
    res_u_mpc = np.concatenate(res_u_mpc, axis=1)

    costs = np.concatenate(costs, axis=1)
    solve_times = np.reshape(solve_times, (-1, 1))

    # fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    #
    # # plot the states
    # ax[0].plot(res_x_mpc.T)
    # ax[1].plot(res_u_mpc.T)
    #
    # # Set labels
    # ax[0].set_ylabel('states')
    # ax[1].set_ylabel('inputs')
    # ax[1].set_xlabel('time')
    # plt.show()
    #np.vstack(u_plot), F,

    return res_x_mpc, res_u_mpc, costs, solve_times

def plotHorizon(u_opt, T, N, F, dt):
    # Plot the solution
    x_opt = [[np.pi / 4, np.pi / 4, 0]]
    t_k = np.array(np.linspace(0, T, N))
    u_0 = u_opt[0]
    for k in range(N):
        Fk = F(x0=x_opt[-1], p=vertcat(u_opt[k], t_k[k], u_0))
        x_opt += [Fk['xf'].full()]
        u_0 = u_opt[k]
    x1_opt = [r[0] for r in x_opt]
    x2_opt = [r[1] for r in x_opt]
    x3_opt = [r[2] for r in x_opt]

    tgrid = [dt * k for k in range(N + 1)]
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, x1_opt, '--')
    plt.plot(tgrid, x2_opt, ':')
    plt.plot(tgrid, x3_opt, '-')
    plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-.')
    plt.xlabel('t')
    plt.legend(['x1', 'x2', 'x3', 'u'])
    plt.grid()
    plt.show()

#u, F, _, _, _, _ = single_shooting()
#T = 10
#N_sim = 80
#dt = 10/5

#plotHorizon(u, T, N_sim, F, dt)