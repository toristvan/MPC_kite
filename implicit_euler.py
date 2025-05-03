
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl 

from datetime import datetime
from casadi import *
from casadi.tools import *

def implicit_euler(N_sim=200,
                   N=100,
                   dt=0.2,
                   x_init=np.array([np.pi/4, np.pi/4, 0 ]),
                   t_start=0,
                   euler="explicit",
                   uncertainty=0.0):

    res_x, res_u, costs, solve_times, res_x_preds = simulation(N_sim=N_sim, N=N, dt=dt, x_init = x_init, t_start=t_start, euler=euler, uncertainty=uncertainty)
    return res_x, res_u, costs, solve_times, res_x_preds

def euler_solver(N=10, dt=0.2, x_init=np.array([np.pi/4, np.pi/4, 0 ]), t_start=0, euler="implicit"):
    # For the euler method, there is no need for an integrator
    # Simulation parameters

    #dt = T/N
    
    #System Parameters
    E0 = 5 # MX.sym("E0")
    vm = 10 # MX.sym("vm")
    vA = 0.5 #1 # MX.sym("vA")
    vf = 0.1 # MX.sym("vf")
    voff = np.pi # MX.sym("voff")
    c = 0.028 # MX.sym("c")
    beta = 0 # MX.sym("beta")
    rho = 1 # MX.sym("rho")
    L = 300 # MX.sym("l")
    A = 160 # MX.sym("A")
    hmin = 100
    
    #States and control variables
    nx = 3
    nu = 1
    
    x = SX.sym("x", nx, 1)
    u = SX.sym("u", nu, 1)
    u_old = SX.sym("u_old", nu, 1)
    t = SX.sym("t")
    
    #Equations
    v0 = vm + vA*sin(2*np.pi*vf*t + voff)
    v0_fcn = Function('v0_fcn', [t], [v0])
    E = E0 - c*(u**2)
    E_fcn = Function('E_fcn', [u], [E])
    va = v0_fcn(t)*E*cos(x[0])
    va_fcn = Function('va_fcn', [x, t], [va])
    PD = rho*(v0_fcn(t)**2)/2
    PD_fcn = Function('PD_fcn', [t], [PD])
    TF = PD_fcn(t)*A*(cos(x[0])**2)*(E_fcn(u) + 1)*sqrt(E_fcn(u)**2 + 1)*(cos(x[0])*cos(beta) + sin(x[0])*sin(beta)*sin(x[1]))
    tension = Function('tension', [x,u,t], [TF])

    height = L * cos(x[1]) * sin(x[0])
    height_fcn = Function('height_fcn', [x], [height])

    xdot = vertcat((va_fcn(x, t)/L)*(cos(x[2]) - tan(x[0])/E_fcn(u)), 
                    -va_fcn(x, t)*sin(x[2])/(L*sin(x[0])),
                    va_fcn(x,t)*u/L - cos(x[0])*(-va_fcn(x, t)*sin(x[2])/(L*sin(x[0]))))

    # state equation formulated as a nonlinear function
    f = Function('f', [x,u,t],[xdot])

    # Stage cost (objective term to be minimized)
    wF = 1e-4
    wu = 0.5

    stage_cost = -wF * tension(x, u, t) + wu * ((u - u_old) ** 2)
    stage_cost_fcn = Function('stage_cost_fcn', [x, u, t, u_old], [stage_cost])

    # state constraints
    lb_x = np.array([0,-np.pi/2, -np.pi])
    ub_x = np.array([np.pi/2,np.pi/2, np.pi])

    # input constraints
    lb_u = -1.5 * np.ones((nu, 1))
    ub_u = 1.5 * np.ones((nu, 1))

    J = 0
    lb_X = []  # lower bound for X.
    ub_X = []  # upper bound for X
    lb_U = []  # lower bound for U
    ub_U = []  # upper bound for U
    g = []  # constraint expression g
    lb_g = []  # lower bound for constraint expression g
    ub_g = []  # upper bound for constraint expression g

    # Define optimization variables
    X = SX.sym("X",(N+1)*nx,1)
    U = SX.sym("U",N*nu,1)

    # Define the initial condition
    #x_init = np.array([np.pi/4, np.pi/4, 0 ])

    # Define arrays to store the results
    mpc_x = np.zeros((N+1,nx))
    mpc_x[0,:] = x_init.T
    mpc_u = np.zeros((N,nu))
    mpc_sim = np.zeros((N+1,nx)) # Simulation Results
    mpc_sim[0,:] = x_init.T

    # lift initial conditions
    lb_X.append(x_init)
    ub_X.append(x_init)

    # initialize
    tk = np.array(np.linspace(t_start, t_start + N*dt, N+1))
    
    for k in range(N):
        x_k = X[k*nx:(k+1)*nx]
        x_next = X[(k+1)*nx:(k+2)*nx]
        u_k = U[k*nu:(k+1)*nu]

        if k == 0:
            #u_prev = U[ k*nu : (k+1)*nu, :]
            u_prev = 0
            u_k = 0
            x_k = x_init
            #print("u_prev: {}".format(u_prev))
        else:
            u_prev = U[(k-1) * nu:k * nu, :]
        print("-----\n{}. iteration\nx_k: {}\nx_next: {}\nu_k: {}".format(k, x_k, x_next, u_k))
        print("u_prev: {}\n".format(u_prev))

        # objective
        J += stage_cost_fcn(x_k, u_k, tk[k], u_prev)

        # implicit euler
        if euler == "implicit":
            f_next = f(x_next, u_k, tk[k+1])
            gk = x_next - x_k - dt * f_next
            g.append(gk)

        # explicit euler
        elif (euler == "explicit"):
            g.append(x_next - x_k - dt*f(x_k, u_k, tk[k]))
        else :
            raise ValueError("Wrong euler method, choose implicit or explicit!")

        lb_g.append(np.zeros((nx,1)))
        ub_g.append(np.zeros((nx,1)))

        # general inequality constraints
        ineq = height_fcn(x_k)
        g.append(ineq)
        lb_g.append(hmin)
        ub_g.append(L)

        # state and input inequality constraints
        lb_X.append(lb_x)
        ub_X.append(ub_x)
        lb_U.append(lb_u)
        ub_U.append(ub_u)

    lb_g = vertcat(*lb_g)
    ub_g = vertcat(*ub_g)
    prob = {'f':J,'x':vertcat(X,U),'g':vertcat(*g)}
    solver = nlpsol('solver','ipopt',prob)

    start_time = datetime.now().timestamp()
    res = solver(lbx=vertcat(*lb_X, *lb_U),ubx=vertcat(*ub_X, *ub_U),lbg=lb_g,ubg=ub_g)
    solve_time = datetime.now().timestamp() - start_time
    cost = res['f']
    print("\ncost: {}\nsolve time: {}\n".format(cost, solve_time))

    # extract the solution
    for k in range(N):
        x_k_opt = np.asarray(res['x'][(k+1)*nx:(k+2)*nx,:])
        u_k_opt = np.asarray(res['x'][(N+1)*nx+k*nu:(N+1)*nx+(k+1)*nu,:])
        mpc_x[k+1,:] = x_k_opt.T
        mpc_u[k,:] = u_k_opt.T
        #x_init += dt*f(x_init.T, mpc_u[k,:].T, tk[k]) # simulation
        #mpc_sim[k+1,:] = x_init.T

    return mpc_x.T, mpc_u, cost, solve_time



def simulation(N_sim=10, N=10, dt=0.2, x_init=np.array([np.pi/4, np.pi/4, 0 ]), t_start=0, euler="implicit", uncertainty=0.05):

    #dt = T/N

    #System Parameters
    E0 = 5 # MX.sym("E0")
    vm = 10 # MX.sym("vm")
    vA = 0.5 #1 # MX.sym("vA")
    vf = 0.1 # MX.sym("vf")
    voff = np.pi # MX.sym("voff")
    c = 0.028 # MX.sym("c")
    beta = 0 # MX.sym("beta")
    rho = 1 # MX.sym("rho")
    L = 300 # MX.sym("l")
    A = 160 # MX.sym("A")
    hmin = 100

    #States and control variables
    nx = 3
    nu = 1

    x = SX.sym("x", nx, 1)
    u = SX.sym("u", nu, 1)
    u_old = SX.sym("u_old", nu, 1)
    t = SX.sym("t")

    #Equations
    v0 = vm + vA*sin(2*np.pi*vf*t + voff)
    v0_fcn = Function('v0_fcn', [t], [v0])
    E = E0 - c*(u**2)
    E_fcn = Function('E_fcn', [u], [E])
    va = v0_fcn(t)*E*cos(x[0])
    va_fcn = Function('va_fcn', [x, t], [va])
    PD = rho*(v0_fcn(t)**2)/2
    PD_fcn = Function('PD_fcn', [t], [PD])
    TF = PD_fcn(t)*A*(cos(x[0])**2)*(E_fcn(u) + 1)*sqrt(E_fcn(u)**2 + 1)*(cos(x[0])*cos(beta) + sin(x[0])*sin(beta)*sin(x[1]))
    tension = Function('tension', [x,u,t], [TF])

    height = L * cos(x[1]) * sin(x[0])
    height_fcn = Function('height_fcn', [x], [height])


    # "real world" model
    omega = np.random.uniform(1-uncertainty, 1+uncertainty, 1)
    # multiplicative uncertainty is added to E, to simulate th real world
    E_real= E0*omega - c*(u**2)
    E_real_fcn = Function('E_real_fcn', [u], [E_real])
    xdot_real = vertcat((va_fcn(x, t)/L)*(cos(x[2]) - tan(x[0])/E_real_fcn(u)),
                    -va_fcn(x, t)*sin(x[2])/(L*sin(x[0])),
                    va_fcn(x,t)*u/L - cos(x[0])*(-va_fcn(x, t)*sin(x[2])/(L*sin(x[0]))))

    # use the real world model for simulation
    f_sim = Function('f', [x,u,t], [xdot_real] )

    t = np.array(np.linspace(0, (N_sim)*dt, N_sim))

    # run simulation loop
    res_x = []
    na = np.newaxis
    print("shape of x_init: {}".format(x_init.shape))
    res_x.append(x_init[:,na])
    res_u = []
    costs = []
    solve_times = []

    # store all the predictions as well for presentation purposes
    res_x_preds = []

    for i in range(N_sim):

        x_preds, u_preds, cost, solve_time = euler_solver(N=N, dt=dt, x_init=x_init, t_start=t[i], euler=euler)
        u_pred = u_preds[0]
        # use first predicted input u and execute simulation model
        x_next = x_init + dt * f_sim(x_init, u_pred, t[i])
        x_init = x_next

        # store
        res_x.append(np.array(x_init))
        res_u.append(np.array(u_pred))
        costs.append(cost)
        solve_times.append(solve_time)

        res_x_preds.append(x_preds)

    costs = np.concatenate(costs, axis=1)
    solve_times = np.reshape(solve_times, (-1, 1))
    res_x = np.asarray(res_x).squeeze().T
    res_u = np.asarray(res_u)
    res_x_preds = np.asarray(res_x_preds).squeeze()

    print("res_x: {}".format(res_x))
    print("res_x.shape: {}".format(res_x.shape))
    print("res_u.shape: {}".format(res_u.shape))
    print("costs.shape: {}".format(costs.shape))
    print("solve_times.shape: {}".format(solve_times.shape))

    print("res_x_preds.shape: {}".format(res_x_preds.shape))
    return res_x, res_u, costs, solve_times, res_x_preds
