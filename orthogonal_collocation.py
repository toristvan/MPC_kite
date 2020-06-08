import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from casadi import *
from casadi.tools import *

#Simulation parameters
N_sim = 200
N = 50
dt = 0.2
nx = 3
nu = 1

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


## Lagrange polynomials

# Computes L_j(tau) for a given list of collocation points
def Lgr(tau_col, tau, j):
    L_j = 1
    for k in range(len(tau_col)):
        if k!=j:
            L_j *= (tau-tau_col[k])/(tau_col[j]-tau_col[k])
    return L_j

# Evaluates x_i^K(t)
def LgrInter(tau_col, tau, xk):
    xk_i = 0
    for j in range(len(xk)):
        xk_i += Lgr(tau_col, tau, j)*xk[j, :]
    return xk_i


def Orthogonal_collocation_MPC(K=3, N=50, N_sim=200, dt=0.2, nx=3, nu=1, E0=5, vm=10, vA=0.5, vf=0.1, voff=np.pi, c=0.028, beta=0, rho=1, L=300, A=160, hmin=100, collocation_tech='legendre'):
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
    #va = vm*E*cos(x[0])
    va_fcn = Function('va_fcn', [x, t], [va])
    PD = rho*(v0_fcn(t)**2)/2
    PD_fcn = Function('PD_fcn', [t], [PD])
    TF = PD_fcn(t)*A*(cos(x[0])**2)*(E_fcn(u) + 1)*np.sqrt(E_fcn(u)**2 + 1)*(cos(x[0])*cos(beta) + sin(x[0])*sin(beta)*sin(x[1]))
    tension = Function('tension', [x,u,t], [TF])

    height = L*cos(x[1])*sin(x[0])
    height_fcn = Function('height_fcn', [x], [height])

    xdot = vertcat((va_fcn(x, t)/L)*(cos(x[2]) - tan(x[0])/E_fcn(u)), 
    -va_fcn(x, t)*sin(x[2])/(L*sin(x[0])), 
    va_fcn(x,t)*u/L - cos(x[0])*(va_fcn(x, t)*sin(x[2])/(L*sin(x[0]))))

    # System and numerical integration
    system = Function('sys', [x,u,t], [xdot])
    ode = {'x': x, 'ode': xdot, 'p': vertcat(u,t)}
    opts = {'tf': dt}
    ode_solver = integrator('F', 'idas', ode, opts)




    # collocation degree
    #K = 3
    # collocation points
    #tau_col = collocation_points(K, 'radau')
    #tau_col = collocation_points(K, 'legendre')

    tau_col = collocation_points(K, collocation_tech)
    tau_col = [0]+tau_col

    # Orthogonal collocation coefficients
    tau = SX.sym("tau")
    A = np.zeros((K+1,K+1))
    for j in range(K+1):
        dLj = gradient(Lgr(tau_col, tau, j), tau)
        dLj_fcn = Function('dLj_fcn', [tau], [dLj])
        for k in range(K+1):
            A[j][k] = dLj_fcn(tau_col[k])

    # Continuity coefficients
    #to determine final state in each finite element

    D = np.zeros((K+1, 1))

    for j in range(K+1):
        L_j = Lgr(tau_col, tau, j)
        L_j_fcn = Function('L_j_fcn', [tau], [L_j])
        D[j] = L_j_fcn(1)


    ##  MPC loop

    #cost
    wF = 1e-4
    wu = 0.5

    stage_cost = -wF*tension(x,u,t) + wu*((u - u_old)**2)
    stage_cost_fcn = Function('stage_cost_fcn', [x, u, t, u_old], [stage_cost])

    #prediction horizon
    #N = 50

    #state_constraints
    #TODO: do something with constraint for psi
    lb_x = np.array([0,-np.pi/2, -np.pi])
    ub_x = np.array([np.pi/2,np.pi/2, np.pi])

    lb_u = np.array([-10])
    ub_u = np.array([10])

    #lb_t
    #ub_t

    #optimization variables

    opt_x = struct_symSX([
        entry('x', shape=nx, repeat=[N+1, K+1]),
        entry('u', shape=nu, repeat=[N])
    ])

    #time = np.array(np.linspace(dt, N_sim*dt, N_sim))


    lb_opt_x = opt_x(0)
    ub_opt_x = opt_x(0)

    lb_opt_x['x'] = lb_x
    ub_opt_x['x'] = ub_x

    lb_opt_x['u'] = lb_u
    ub_opt_x['u'] = ub_u

    #formulate optimization problem

    J = 0
    g = []
    lb_g = []
    ub_g = []

    x_init = SX.sym('x_init', nx)
    time = SX.sym('time', N) #or use t?

    x0 = opt_x['x', 0, 0]
    u_prev = opt_x['u', 0]

    g.append(x0-x_init)
    lb_g.append(np.zeros((nx,1)))
    ub_g.append(np.zeros((nx,1)))

    # objective
    for i in range(N):
        J += stage_cost_fcn(opt_x['x',i,0], opt_x['u',i], time[i], u_prev)
        
        # equality constraints (system equation)
        for k in range(1,K+1):
            gk = -dt*system(opt_x['x',i,k], opt_x['u',i], time[i])
            for j in range(K+1):
                gk += A[j,k]*opt_x['x',i,j]
                
            
            g.append(gk)
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))   

        #inequality constraints
        ineq = height_fcn(opt_x['x', i, 0])
        #ineq = L*cos(opt_x['x', i, 0, 1])*sin(opt_x['x', i, 0, 0])
        g.append(ineq)
        lb_g.append(hmin)
        ub_g.append(L)
        
        x_next = horzcat(*opt_x['x',i])@D
        g.append(x_next - opt_x['x', i+1, 0])
        lb_g.append(np.zeros((nx,1)))
        ub_g.append(np.zeros((nx,1)))
        u_prev = opt_x['u', i]
        

    #J += terminal_cost_fcn(opt_x['x', N, 0])

    g = vertcat(*g)
    lb_g = vertcat(*lb_g)
    ub_g = vertcat(*ub_g)

    prob = {'f':J,'x':vertcat(opt_x),'g':g, 'p':vertcat(x_init, time)}
    mpc_solver = nlpsol('solver','ipopt',prob)

    # MPC Main loop

    #initialize
    x_0 = np.array([np.pi/4, np.pi/4, 0]).reshape(nx, 1)
    t_k = np.array(np.linspace(0, (N_sim+N)*dt, N_sim+N+1))
    res_x_mpc = [x_0]
    res_u_mpc = []
    costs =[]


    for i in range(N_sim):
        # solve optimization problem
        mpc_res = mpc_solver(p=vertcat(x_0, t_k[i:N+i]), lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)
        
        # optionally: Warmstart the optimizer by passing the previous solution as an initial guess!
        if i>0:
            #mpc_res = mpc_solver(p=x_0, x0=opt_x_k, lbg=0, ubg=0, lbx = lb_opt_x, ubx = ub_opt_x)
            mpc_res = mpc_solver(p=vertcat(x_0, t_k[i:N+i]),x0=opt_x_k, lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)

            
        #extract cost
        cost_k = mpc_res['f']
        costs.append(cost_k)

        # Extract the control input
        opt_x_k = opt_x(mpc_res['x'])
        u_k = opt_x_k['u',0]

        # simulate the system
        res_integrator = ode_solver(x0=x_0, p=vertcat(u_k, t_k[i]))
        x_next = res_integrator['xf']
        
        # Update the initial state
        x_0 = x_next
        
        # Store the results
        res_x_mpc.append(x_next)
        res_u_mpc.append(u_k)
        

    # Make an array from the list of arrays:
    res_x_mpc = np.concatenate(res_x_mpc,axis=1)
    res_u_mpc = np.concatenate(res_u_mpc, axis=1)

    costs = np.concatenate(costs, axis=1)
    #cost_mean = np.mean(costs)

    return res_x_mpc, res_u_mpc, costs

'''
fig, ax = plt.subplots(3,3, figsize=(15,12))

#plot position
ax[0][0].plot(L*sin(res_x_mpc[0].T)*sin(res_x_mpc[1].T), L*sin(res_x_mpc[0].T)*cos(res_x_mpc[1].T))
#plot angles towards each other
ax[1][0].plot(res_x_mpc[1].T, res_x_mpc[0].T)
# plot the input
ax[2][0].plot(res_u_mpc.T)

#plot angles over time
ax[0][1].plot(res_x_mpc[0].T)
ax[1][1].plot(res_x_mpc[1].T)
ax[2][1].plot(res_x_mpc[2].T)

#plot wind
#ax[2][1].plot(t_k, v0_fcn(t_k))
#plot cost
ax[0][2].plot(costs.T)
ax[0][2].set_xlabel('time')
ax[0][2].set_ylabel('cost')

#ax[0].plot(res_x_mpc[0].T, res_x_mpc[1].T)

# Set labels
ax[1][0].set_ylabel('theta')
ax[1][0].set_xlabel('phi')

ax[0][0].set_ylabel('height')
ax[0][0].set_xlabel('horizontal position')

ax[2][0].set_ylabel('inputs')
ax[2][0].set_xlabel('time')

ax[0][1].set_ylabel('theta')
ax[0][1].set_xlabel('time')

ax[1][1].set_ylabel('phi')
ax[1][1].set_xlabel('time')

ax[2][1].set_ylabel('psi')
ax[2][1].set_xlabel('time')


plt.show()
'''