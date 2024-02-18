import numpy as np
import matplotlib.pyplot as plt

dwx = (0.2,0.5,1,2)
dwy = (0.8,0.25,0.12,0.035)
 
mp = np.linspace(0.2,2,40)
f_still = 0.09*mp**(-1.4)
 
plt.plot(dwx,dwy,'o')
plt.plot(mp,f_still)

plt.xlabel(r'Planet mass [$M_{jup}$]')
plt.ylabel(r'$f_{still} = \Gamma/\Gamma_0$')

plt.xscale('log')
plt.yscale('log')
plt.show()

dwad  = (0.4, 0.6, 1., 1.8, 3.3, 4.3, 6.)
dwsig = (0.02, 0.05, 0.1, 0.2, 0.5, 1., 2.)

sig  = np.linspace(0.02,2.,40)
adot = 4 * sig**(0.6)

plt.plot(dwsig,dwad,'o')
plt.plot(sig,adot)

plt.xlabel(r'$\Sigma/\Sigma_0$')
plt.ylabel(r'$\dot{a}/v_{visc}$')

plt.xscale('log')
plt.yscale('log')
plt.show()

dw_fmig = (0.43,0.3,0.2,0.15,0.08,0.03,0.02) 

plt.plot(dwsig, dw_fmig,'o')
plt.plot(sig,1/adot)

plt.ylabel(r'$f_{mig} = \Gamma/(f_{still}\Gamma_0)$')
plt.xlabel(r'$\Sigma/\Sigma_0$')

plt.xscale('log')
plt.yscale('log')
plt.show()
