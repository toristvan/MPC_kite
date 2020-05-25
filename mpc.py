import numpy as np
import matplotlib.pyplot as plt
from casadi import *

## Framework from exercise - to be altered
## system in (discrete-time) state-space form

nx = 3
nu = 1

x = SX.sym("x", nx, 1)
u = SX.sym("u", nu, 1)
u_old = SX.sym("u_old", nu, 1)

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
print(res_x)

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

## Cost function

#Parameters
E0 = 5 # MX.sym("E0")
vm = 10 # MX.sym("vm")
vA = 0.5 # MX.sym("vA")
vf = 0.1 # MX.sym("vf")
voff = np.pi # MX.sym("voff")
c = 0.028 # MX.sym("c")
beta = 0 # MX.sym("beta")
rho = 1 # MX.sym("rho")
l = 500 # MX.sym("l")
A = 30 # MX.sym("A")

#Equations
v0 = vm + vA*sin(2*np.pi)
E = E0 - c*(u**2)
va = v0*E*cos(x[1])
va_fnc = Function('va', [x], [va])

PD = rho*(v0**2)/2
#TF = PD*A*(cos(theta)**2)*(E + 1)*np.sqrt(E**2 + 1)*(cos(theta)*cos(beta) + sin(theta)*sin(beta)*sin(phi))

T_F = PD*A*(cos(x[0])**2)*(E + 1)*np.sqrt(E**2 + 1)*(cos(x[0])*cos(beta) + sin(x[0])*sin(beta)*sin(x[1]))

tension = Function('tension', [x], [T_F])

#Cost function/Objective term
wF = 1e-4
wu = 0.5

stage_cost = -wF*tension(x) + wu*(u_old - u)**2

stage_cost_fnc = Function('stage_cost', [x, u, u_old], [stage_cost])


#state and control input bounds
lb_x = -4*np.ones(nx, 1)
ub_x = 4*np.ones(nx, 1)

lb_u = -1*np.ones(nu, 1)
ub_u = -1*np.ones(nu, 1)

#Stacked vector of states and inputs

X = SX.sym("X", )

