import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from casadi import *
from casadi.tools import *
#from orthogonal_collocation import Orthogonal_collocation_MPC
from orthogonal_collocation_testing import test_orth_col
#from datetime import datetime

def main():
    for t in range(2,7,2):

        test_orth_col(dt=t/10, Kmin=3, Kmax=6, Nmin=50, Nmax=70)



if __name__ == '__main__':
    main()