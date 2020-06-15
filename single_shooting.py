import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from casadi import *
from casadi.tools import *

T = 10. # Time horizon
N = 20 # number of control intervals

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

# Model equations
xdot = vertcat((1-x2**2)*x1 - x2 + u, x1)

# Objective term
L = x1**2 + x2**2 + u**2

# Formulate discrete time dynamics

# CVODES from the SUNDIALS suite
dae = {'x':x, 'p':u, 'ode':xdot, 'quad':L}
opts = {'tf':T/N}
F = integrator('F', 'cvodes', dae, opts)

# Evaluate at a test point
Fk = F(x0=[0.2,0.3],p=0.4)
print(Fk['xf'])
print(Fk['qf'])

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
Xk = MX([0, 1])
for k in range(N):
    # New NLP variable for the control
    Uk = MX.sym('U_' + str(k))
    w += [Uk]
    lbw += [-1]
    ubw += [1]
    w0 += [0]

    # Integrate till the end of the interval
    Fk = F(x0=Xk, p=Uk)
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