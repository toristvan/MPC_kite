import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from casadi import *



#Parameters
E0 = 5 # MX.sym("E0")
vm = 10 # MX.sym("vm")
vA = 5 #1 # MX.sym("vA")
vf = 0.1 # MX.sym("vf")
voff = np.pi # MX.sym("voff")
c = 0.028 # MX.sym("c")
beta = 0 # MX.sym("beta")
rho = 1 # MX.sym("rho")
L = 500 # MX.sym("l")
A = 30 # MX.sym("A")

#States and control variables
dt = 0.1
nx = 3
nu = 1

x = SX.sym("x", nx, 1)
u = SX.sym("u", nu, 1)
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

xdot = vertcat((va_fcn(x, t)/L)*(cos(x[2])-tan(x[0]/E_fcn(u))), -va_fcn(x, t)*sin(x[2]/(L*sin(x[0]))), va_fcn(x,t)*u/L - cos(x[0])*(-va_fcn(x, t)*sin(x[2]/(L*sin(x[0])))))
#xdot = vertcat((va_fcn(x)/L)*(cos(x[2])-tan(x[0]/E_fcn(u))), -va_fcn(x)*sin(x[2]/(L*sin(x[0]))), va_fcn(x)*u/L - cos(x[0])*xdot[1])

# System and numerical integration
system = Function('sys', [x,u,t], [xdot])
ode = {'x': x, 'ode': xdot, 'p': vertcat(u,t)}
opts = {'tf': dt}
ode_solver = integrator('F', 'idas', ode, opts)

#print(ode_solver)

## Simulation of system with and without time input
# preparation
N_sim = 1000
x_init = np.array([np.pi/4, 0, 0]).reshape(nx, 1)
x_0 = x_init
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
plt.show()

## Lagrange polynomials

# Computes L_j(tau) for a given list of collocation points
def Lj(tau_col, tau, j):
    L_j = 1
    for k in range(len(tau_col)):
        if k!=j:
            L_j *= (tau-tau_col[k])/(tau_col[j]-tau_col[k])
    return L_j

# Evaluates x_i^K(t)
def LgrInter(tau_col, tau, xk):
    xk_i = 0
    for j in range(len(xk)):
        xk_i += Lj(tau_col, tau, j)*xk[j, :]
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
    dLj = gradient(L(tau_col, tau, j), tau)
    dLj_fcn = Function('dLj_fcn', [tau], [dLj])
    for k in range(K+1):
        A[j][k] = dLj_fcn(tau_col[k])
