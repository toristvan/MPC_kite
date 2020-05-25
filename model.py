
from casadi import *
import numpy as np

T = 100 #Time horizon
N = 200 #Number of control intervals

#States
theta = MX.sym("theta")
phi = MX.sym("phi")
psi = MX.sym("psi")

x = vertcat(theta, phi, psi)

#Control input
u = MX.sym("u")
u_old = MX.sym("u_old")

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
va = v0*E*cos(theta)
PD = rho*(v0**2)/2
TF = PD*A*(cos(theta)**2)*(E + 1)*np.sqrt(E**2 + 1)*(cos(theta)*cos(beta) + sin(theta)*sin(beta)*sin(phi))

#Equations of motion
thetadot = (va/l)*(cos(psi) - tan(theta)/E)
phidot = -sin(psi)*va/(l*sin(theta))
psidot = va*u/l + cos(theta)*phidot


#xdot= vertcat((va/L)*(cos(psi) - tan(theta)/E), -sin(psi)*va/(L*sin(theta)), va*u/L + cos(theta)*phidot)
xdot = vertcat(thetadot, phidot, psidot)

#Constrains
hmin = 100
h = l*sin(x[0])*cos(x[1])
h_func = Function('h', [x], [h])

#Cost function/Objective term
wF = 1e-4
wu = 0.5

#cost_func = Function('cost', [TF, u_new, u],[-wF*TF + wu*(u_new - u)**2])
L = -wF*TF + wu*(u_old - u)**2


F = Function('F', [x,u, u_old], [xdot, L])
#Implicit Euler
neu = 0.1 #step size
k1 = F(x + neu*k1, u)
x = x + neu*k1

