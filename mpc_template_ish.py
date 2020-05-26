import numpy as np
import matplotlib.pyplot as plt
from casadi import *

## Framework from exercise - to be altered
## system in (discrete-time) state-space form

### System (A an B) is arbitrary
### Cost function is not working correctly - T_F and tension is not used.

nx = 3
nu = 1

x = SX.sym("x", nx, 1)
u = SX.sym("u", nu, 1)
u_old = SX.sym("u_old", nu, 1)

#Parameters
E0 = 5 # MX.sym("E0")
vm = 10 # MX.sym("vm")
vA = 0.5 # MX.sym("vA")
vf = 0.1 # MX.sym("vf")
voff = np.pi # MX.sym("voff")
c = 0.028 # MX.sym("c")
beta = 0 # MX.sym("beta")
rho = 1 # MX.sym("rho")
l = 500 # MX.sym("l") #length of wire
A = 30 # MX.sym("A")

#Equations
v0 = vm + vA*sin(2*np.pi)

E = E0 - c*(u**2)
E_fnc = Function('E', [u], [E])

va = v0*E_fnc(u)*cos(x[1])
va_fnc = Function('va', [x, u], [va])

PD = rho*(v0**2)/2
#TF = PD*A*(cos(theta)**2)*(E + 1)*np.sqrt(E**2 + 1)*(cos(theta)*cos(beta) + sin(theta)*sin(beta)*sin(phi))

# random values
A = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1]).reshape(3,3)
B = np.array([0.5, 0, 0.5]).reshape(3,1)

#eigenvalues, stable?
lam, v = np.linalg.eig(A)
'''
fig, ax = plt.subplots(figsize=(6,6))
ax.axhline(0, color='k')
ax.axvline(0, color='k')
ax.plot(np.real(lam), np.imag(lam), '+', color='r', markersize=12)
ax.add_artist(plt.Circle((0,0),1, edgecolor='b', fill=False))
ax.set_xlabel("Real")
ax.set_ylabel("Img")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
plt.show()
'''
#system
x_next = A@x + B@u
system = Function('system', [x,u], [x_next])

'''
N_sim = 50

x_0 = np.array([-1, 1, 0]).reshape(3,1)
u_k = np.array([[0]]) #constant

res_x = [x_0]
x_i = x_0
#res_x = np.array([res_x])
for i in range(N_sim):
    #u_i = u_k[i]
    #print(x_i)
    x_knext = system(x_i, u_k)
    res_x.append(x_knext)
    x_i = x_knext
    


res_x = np.concatenate(res_x, axis=1)
#print(res_x)
'''

'''
fig, ax = plt.subplots(figsize=(10,6))

# plot the states
lines = ax.plot(res_x.T)
ax.legend(lines, ['1st state', '2nd state', '3rd state'])
ax.set_ylabel('states')
ax.set_xlabel('time')

plt.show()
'''

### MPC

N = 50

#Cost function/Objective term

T_F = PD*A*(cos(x[0])**2)*(E_fnc(u) + 1)*np.sqrt(E_fnc(u)**2 + 1)*(cos(x[0])*cos(beta) + sin(x[0])*sin(beta)*sin(x[1]))
#T_F = x.T@x
tension = Function('tension', [x,u], [T_F])
#print(T_F)
# weights
wF = 1e-4
wu = 0.5

#test matrices
Q = 0.2*np.diag(np.ones(nx))
R = 0.5*np.diag(np.ones(nu))

#stage_cost = -wF*tension(x,u) + wu*(u_old - u)**2
stage_cost = x.T@Q@x + u.T@R@u
stage_cost_fnc = Function('stage_cost', [x, u, u_old], [stage_cost])

#terminal_cost = -wF*tension(x, u_old)
terminal_cost = x.T@Q@x
terminal_cost_fnc = Function('terminal_cost', [x, u_old], [terminal_cost])

#height constraints
hmin = 100
h = l*sin(x[0])*cos(x[1])
h_func = Function('h', [x], [h])

#state and control input bounds
lb_x = -4*np.ones((nx, 1))
ub_x = 4*np.ones((nx, 1))

lb_u = -1*np.ones((nu, 1))
ub_u = 1*np.ones((nu, 1))

#Stacked vector of states and inputs
X = SX.sym("X", nx*(N+1), 1)
U = SX.sym("U", nu*N, 1)

#Optimization problem variables

J = 0 #cost
lb_X = []
ub_X = []
lb_U = []
ub_U = []
g = []
lb_g = []
ub_g = []

#Formulation of problem


#first value can start wherever or at zero?
u_old = U[0:nu, :]
#u_old = 0
for k in range(N):
    #fetch states
    x_k = X[k*nx:(k+1)*nx, :]
    x_k_next = X[(k+1)*nx:(k+2)*nx, :]
    u_k = U[k*nu:(k+1)*nu, :]


    #cost
    J += stage_cost_fnc(x_k, u_k, u_old)
    u_old = u_k

    #calculate next x according to system dynamics
    x_k_next_calc = system(x_k, u_k)
    
    #height constraint (to kick in after certain time?)
    height = h_func(x_k) - hmin

    #constraint to make sure x_next coincides with system
    
    #constraints = vertcat(x_k_next - x_k_next_calc, height)
    #g.append(constraints)
    #lb_g.append(np.zeros((nx + 1, 1)))
    #ub_g.append(vertcat(np.zeros((nx, 1)), l))

    g.append(x_k_next - x_k_next_calc)
    lb_g.append(np.zeros((nx, 1)))
    ub_g.append(np.zeros((nx, 1)))

    #constant for all times
    lb_X.append(lb_x)
    ub_X.append(ub_x)
    lb_U.append(lb_u)
    ub_U.append(ub_u)

x_terminal = X[N*nx:(N+1)*nx, :]

#u_old is most relevant u, but is in reality not correct
J += terminal_cost_fnc(x_terminal, u_old)
lb_X.append(lb_x)
ub_X.append(ub_x)


#Create CasADI solver

x_sol = vertcat(X,U)
lbx_sol = vertcat(*lb_X, *lb_U)
ubx_sol = vertcat(*ub_X, *ub_U)
g_sol = vertcat(*g)
lbg_sol = vertcat(*lb_g)
ubg_sol = vertcat(*ub_g)

prob = {'f':J,'x':x_sol,'g':g_sol}
solver = nlpsol('solver','ipopt',prob)

#print(solver)

# Run loop once
'''
x_0 = np.array([-1, 1, 0]).reshape(3,1)

lbx_sol[:nx] = x_0
ubx_sol[:nx] = x_0

res = solver(lbx=lbx_sol, ubx=ubx_sol, lbg=lbg_sol, ubg=ubg_sol)

#f_opt = res['f']
x_opt = res['x']
X_opt = x_opt[:(N+1)*nx].full().reshape(N+1, nx)
U_opt = x_opt[(N+1)*nx:].full().reshape(N, nu)

fig, ax = plt.subplots(2,1, figsize=(10,8), sharex=True)
ax[0].plot(X_opt)
ax[1].plot(U_opt)
ax[0].set_ylabel('states')
ax[1].set_ylabel('control input')
ax[1].set_xlabel('time')

# Highlight the selected initial state (the lines should start here!)
ax[0].plot(0,x_0.T, 'o', color='black')

fig.align_ylabels()
fig.tight_layout()

#plt.show()
'''
# Run Full MPC

x_0 = np.array([-1, 1, 0]).reshape(3,1)

res_x = [x_0]
res_u = []
x_k = x_0
for k in range(N):
    lbx_sol[:nx] = x_k
    ubx_sol[:nx] = x_k

    res = solver(lbx = lbx_sol, ubx = ubx_sol, lbg = lbg_sol, ubg = ubg_sol)
    U_opt = res['x'][(N+1)*nx:]
    u_k = U_opt[0]

    x_k_next = system(x_k, u_k)
    res_u.append(u_k)
    x_k = x_k_next
    res_x.append(x_k)

res_x = np.concatenate(res_x, axis = 1)
res_u = np.concatenate(res_u, axis = 1)


fig, ax = plt.subplots(2,1, figsize=(10,8), sharex=True)
ax[0].plot(res_x.T)
ax[1].plot(res_u.T)
ax[0].set_ylabel('states')
ax[1].set_ylabel('control input')
ax[1].set_xlabel('time')

fig.align_ylabels()
fig.tight_layout()

plt.show()