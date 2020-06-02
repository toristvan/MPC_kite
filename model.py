
from casadi import *
import numpy as np


#States
#theta = MX.sym("theta")
#phi = MX.sym("phi")
#psi = MX.sym("psi")
#x = vertcat(theta, phi, psi)

#Control input
#u = MX.sym("u")
#u_old = MX.sym("u_old")

dt = 0.1
nx = 3
nu = 1

#Parameters
E0 = 5 # MX.sym("E0")
vm = 10 # MX.sym("vm")
vA = 1 #1 # MX.sym("vA")
vf = 0.1 # MX.sym("vf")
voff = np.pi # MX.sym("voff")
c = 0.028 # MX.sym("c")
beta = 0 # MX.sym("beta")
rho = 1 # MX.sym("rho")
L = 500 # MX.sym("l")
A = 30 # MX.sym("A")

#Equations of motion

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
#Equations of motion
#thetadot = (va/l)*(cos(psi) - tan(theta)/E)
#phidot = -sin(psi)*va/(l*sin(theta))
#psidot = va*u/l + cos(theta)*phidot
#xdot = vertcat(thetadot, phidot, psidot)

#Constrains
hmin = 100
h = l*sin(x[0])*cos(x[1])
h_func = Function('h', [x], [h])

#Cost function/Objective term
wF = 1e-4
wu = 0.5

#cost_func = Function('cost', [TF, u_new, u],[-wF*TF + wu*(u_new - u)**2])
stage_cost = -wF*tension(x,u) + wu*(u_old - u)**2
stage_cost_fnc = Function('stage_cost', [x, u, u_old], [stage_cost])



F = Function('F', [x,u, u_old], [xdot, L])
#Implicit Euler
neu = 0.1 #step size
k1 = F(x + neu*k1, u)
x = x + neu*k1

