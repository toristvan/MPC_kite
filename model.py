
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
u_new = MX.sym("u_new")

#Parameters
E0 = MX.sym("E0")
vm = MX.sym("vm")
vA = MX.sym("vA")
vf = MX.sym("vf")
voff = MX.sym("voff")
c = MX.sym("c")
beta = MX.sym("beta")
rho = MX.sym("rho")
L = MX.sym("L")
A = MX.sym("A")

#Equations
v0 = vm + vA*sin(2*np.pi)
E = E0 - c*(u**2)
va = v0*E*cos(theta)
PD = rho*(v0**2)/2
TF = PD*A*(cos(theta)**2)*(E + 1)*np.sqrt(E**2 + 1)*(cos(theta)*cos(beta) + sin(theta)*sin(beta)*sin(phi))

#Equations of motion
thetadot = (va/L)*(cos(psi) - tan(theta)/E)
phidot = -sin(psi)*va/(L*sin(theta))
psidot = va*u/L + cos(theta)*phidot


#xdot= vertcat((va/L)*(cos(psi) - tan(theta)/E), -sin(psi)*va/(L*sin(theta)), va*u/L + cos(theta)*phidot)
xdot = vertcat(thetadot, phidot, psidot)

#Constrains
hmin = 100
h = L*sin(x[0])*cos(x[1])
h_func = Function('h', [x], [h])

#Cost function
wF = 1e-4
wu = 0.5

#cost_func = Function('cost', [TF, u_new, u],[-wF*TF + wu*(u_new - u)**2])
