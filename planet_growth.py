import numpy as np

class Planet:
    def __init__(self, mass, dist=10.0, N_comp=3, f_comp):
        self.mass = mass
        self.dist = dist 
        self.N_comp = N_comp
        self.f_comp = np.zeros(N_comp)

def seed_mass(hr, flaring, gas_slope, dist):
    dist = dist*Rau
    vk = np.sqrt(mass_star*G_Newton/dist)
    pres_grad = 2*(flaring-1)+gas_slope
    eta = - 0.5* hr**2 * pres_grad
    m_min = (eta*vk)**3/G_Newton/vk*dist/np.sqrt(3)
    return m_min

def h_over_r (dist, mass_star, temperature, mu=2.35):
    vk = np.sqrt(mass_star*G_Newton/dist)
    cs = np.sqrt(k_boltzmann/mu/m_hydrogen*temperature)
    hr = cs/vk
    return hr

def stokes():
    pass

def pebble_accretion(mass_p, mass_star, dist, pebble_density, stokes, hr, alpha):
    r_hill = dist*(mass_p/mass_star/3.)**(1./3.)
    v_hill = r_hill*np.sqrt(mass_star*G_Newton/dist**3)
    h_peb = hr*dist*np.sqrt(alpha/stokes)
    dm_2d = 2.0 * (stokes/0.1)**(2./3.)*r_hill*v_hill*pebble_density
    dm_3d = dm_2d * (r_hill * np.pi**0.5 / 2**1.5 / h_peb *(stokes/0.1)**(1./3.))
    crit_h = np.pi* (stokes/0.1)**(1./3.)*r_hill/2/np.sqrt(2*np.pi)
    if h_peb > crit_h: dm_peb = dm_2d
    else: dm_peb = dm_3d
    return dm_peb

def gas_accretion(mass_p, hr, dist, gas_density):
    r_hill = dist*(mass_p/mass_star/3.)**(1./3.)
    omg_k  = np.sqrt(mass_star*G_Newton/dist**3)
    dm_low = 0.83 * omg_k * gas_density* (hr*dist)**2 * (r_hill/hr/dist)**(4.5)
    dm_high = 0.14 * np.sqrt(mass_star*G_Newton/dist**3) * gas_density * (hr*dist)**2 
    dm_gas = np.min(()dm_low,dm_high))
    return dm_gas

def planet_growth(planet, hh, acc_peb, acc_gas, dt, peb_comp, gas_comp):
    M_Earth = 5.9722e27
    if planet.mass < 20 * (hr/0.05)**3. * M_Earth:
        dm = acc_peb
        comp = peb_comp
    else:
        dm = acc_gas
        comp = gas_comp
    mass = planet.mass+dm*dt

    pass
