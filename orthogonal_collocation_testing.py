import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from casadi import *
from casadi.tools import *



#Parameters
E0 = 5 # MX.sym("E0")
vm = 10 # MX.sym("vm")
vA = 0.5 #1 # MX.sym("vA")
vf = 0.1 # MX.sym("vf")
voff = np.pi # MX.sym("voff")
c = 0.028 # MX.sym("c")
beta = 0 # MX.sym("beta")
rho = 1 # MX.sym("rho")
L = 500 # MX.sym("l")
A = 30 # MX.sym("A")

#States and control variables
dt = 0.2
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
TF = PD_fcn(t)*A*(cos(x[0])**2)*(E_fcn(u) + 1)*np.sqrt(E_fcn(u)**2 + 1)*(cos(x[0])*cos(beta) + sin(x[0])*sin(beta)*sin(x[1]))
tension = Function('tension', [x,u,t], [TF])

xdot = vertcat((va_fcn(x, t)/L)*(cos(x[2]) - tan(x[0])/E_fcn(u)), -va_fcn(x, t)*sin(x[2])/(L*sin(x[0])), va_fcn(x,t)*u/L - cos(x[0])*(-va_fcn(x, t)*sin(x[2])/(L*sin(x[0]))))
#xdot = vertcat((va_fcn(x)/L)*(cos(x[2])-tan(x[0])/E_fcn(u)), -va_fcn(x)*sin(x[2]/(L*sin(x[0]))), va_fcn(x)*u/L - cos(x[0])*xdot[1])

# System and numerical integration
system = Function('sys', [x,u,t], [xdot])
ode = {'x': x, 'ode': xdot, 'p': vertcat(u,t)}
opts = {'tf': dt}
ode_solver = integrator('F', 'idas', ode, opts)

#print(ode_solver)

## Simulation of system with and without time input
# preparation
N_sim = 500
x_0 = np.array([np.pi/4, 0, 0]).reshape(nx, 1)
x_0_t = x_0
u_k = np.array([[0]])
t_k = np.array(np.linspace(dt, N_sim*dt, N_sim))

res_x_sundials = [x_0]
res_x_sundials_t = [x_0]

# simulation
for k in range(N_sim):
    sol = ode_solver(x0 = x_0, p = vertcat(u_k, u_k))
    x_f = sol['xf']
    res_x_sundials.append(x_f)
    x_0 = x_f

    #with time for wind oscillations
    sol_t = ode_solver(x0 = x_0_t, p = vertcat(u_k, t_k[k]))
    x_f_t = sol_t['xf']
    res_x_sundials_t.append(x_f_t)
    x_0_t = x_f_t

res_x_sundials = np.concatenate(res_x_sundials, axis = 1)
res_x_sundials_t = np.concatenate(res_x_sundials_t, axis = 1)

#print(res_x_sundials)

# Plot results
fig, ax = plt.subplots(figsize=(10,6))
lines = ax.plot(res_x_sundials.T)
ax.legend(lines, ['theta', 'phi', 'psi'])
lines_t = ax.plot(res_x_sundials_t.T)
ax.legend(lines + lines_t, ['theta', 'phi', 'psi', 'theta_t', 'phi_t', 'psi_t'])
ax.set_ylabel('states')
ax.set_xlabel('time')
#plt.show()

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


# collocation degree
K = 2
# collocation points
tau_col = collocation_points(K, 'radau')
#tau_col = collocation_points(K-1, 'legendre')
tau_col = [0]+tau_col

# Orthogonal collocation coefficients
tau = SX.sym("tau")
A = np.zeros((K+1,K+1))
for j in range(K+1):
    dLj = gradient(Lgr(tau_col, tau, j), tau)
    dLj_fcn = Function('dLj_fcn', [tau], [dLj])
    for k in range(K+1):
        A[j][k] = dLj_fcn(tau_col[k])

#print(A)

# Continuity coefficients
#to determine final state in each finite element

D = np.zeros((K+1, 1))

for j in range(K+1):
    L_j = Lgr(tau_col, tau, j)
    L_j_fcn = Function('L_j_fcn', [tau], [L_j])
    D[j] = L_j_fcn(1)

#print(D)

# Test integration with orthgonal collocation
### NOT WORKING
#Create optimization problem
#Initialize empty list of constraints
'''

g = []

#states
X = SX.sym("X", nx, K+1)
x_init = SX.sym("x0", nx)
u = SX.sym("u", nu)
t = SX.sym("t")

g.append(X[:,0] - x_init)

for k in range(1, K+1):
    gk = -dt*system(X[:,k], u, t)
    for j in range(K+1):
        gk += A[j][k]*X[:,j]
        
    g.append(gk)

g = vertcat(*g)

#print(g.shape)
#print(X.reshape((-1,1)).shape)
nlp = {'x': X.reshape((-1,1)), 'g': g, 'p': vertcat(x_init, u, t)}
#opts = {'halt_on_ampl_error':'yes'}
#option ipopt_options "halt_on_ampl_error=yes"
solver = nlpsol('solver', 'ipopt', nlp)
print(solver)

#Solve ODE with orthogonal collocation

x_0 = np.array([np.pi/4, np.pi/4, np.pi/4]).reshape(nx, 1)

res_x_oc = [x_0]

for i in range(N_sim):
    res_oc = solver(lbg=0, ubg=0, p=vertcat(x_0, u_k, t_k[i]))
    X_k = res_oc['x'].full().reshape(K+1,nx)
    x_next = X_k.T@D
    res_x_oc.append(x_next)
    x_0 = x_next

# Make an array from the list of arrays:
res_x_oc = np.concatenate(res_x_oc,axis=1)
'''

#  MPC loop

#cost
wF = 1e-4
wu = 0.5

stage_cost = -wF*tension(x,u,t) + wu*(u_old - u)**2
stage_cost_fcn = Function('stage_cost_fcn', [x, u, t, u_old], [stage_cost])

#prediction horizon
N = 20

#state_constraints
lb_x = np.array([0,-np.pi/2, 0])
ub_x = np.array([np.pi/2,np.pi/2, 2*np.pi])

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

    #TODO: add inequality constraints
    
    x_next = horzcat(*opt_x['x',i])@D
    g.append(x_next - opt_x['x', i+1, 0])
    lb_g.append(np.zeros((nx,1)))
    ub_g.append(np.zeros((nx,1)))
    u_prev = opt_x['u', i]
    

#J += terminal_cost_fcn(opt_x['x', N, 0])

g = vertcat(*g)

prob = {'f':J,'x':vertcat(opt_x),'g':g, 'p':vertcat(x_init, time)}
mpc_solver = nlpsol('solver','ipopt',prob)

# test solver
'''
x_0 = np.array([np.pi/4, 0, 0]).reshape(nx, 1)
mpc_res = mpc_solver(p=vertcat(x_0, t_k[:N]), lbg=0, ubg=0, lbx = lb_opt_x, ubx = ub_opt_x)

#get results
opt_x_k = opt_x(mpc_res['x'])

X_k = horzcat(*opt_x_k['x',:,0,:])
U_k = horzcat(*opt_x_k['u',:])

fig, ax = plt.subplots(2,1, figsize=(10,6))

# plot the states
ax[0].plot(X_k.T)
ax[1].plot(U_k.T)

# Set labels
ax[0].set_ylabel('states')
ax[1].set_ylabel('inputs')
ax[1].set_xlabel('time')
plt.show()

'''

# MPC Main loop

#initialize
x_0 = np.array([np.pi/4, 0, 0]).reshape(nx, 1)
res_x_mpc = [x_0]
res_u_mpc = []


for i in range(N_sim):
    # solve optimization problem
    mpc_res = mpc_solver(p=vertcat(x_0, t_k[:N]), lbg=0, ubg=0, lbx = lb_opt_x, ubx = ub_opt_x)
    
    # optionally: Warmstart the optimizer by passing the previous solution as an initial guess!
    #if i>0:
        #mpc_res = mpc_solver(p=x_0, x0=opt_x_k, lbg=0, ubg=0, lbx = lb_opt_x, ubx = ub_opt_x)
        
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
    # 05
    
# Make an array from the list of arrays:
res_x_mpc = np.concatenate(res_x_mpc,axis=1)
res_u_mpc = np.concatenate(res_u_mpc, axis=1)

fig, ax = plt.subplots(2,1, figsize=(10,6))

# plot the states
ax[0].plot(res_x_mpc.T)
ax[1].plot(res_u_mpc.T)

# Set labels
ax[0].set_ylabel('states')
ax[1].set_ylabel('inputs')
ax[1].set_xlabel('time')
plt.show()