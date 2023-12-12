import numpy as np
from drift_composition.constants import k_boltzmann, m_hydrogen, G_Msun, Rau, Msun, yr, Mearth
from drift_composition.planetesimal_flux import Plansi, plansi_flux

def seed_mass(hr, flaring, gas_slope, dist):
    dist = dist*Rau
    vk = np.sqrt(mass_star*G_Msun/dist)
    pres_grad = 2*(flaring-1)+gas_slope
    eta = - 0.5* hr**2 * pres_grad
    m_min = (eta*vk)**3/G_Msun/vk*dist/np.sqrt(3)
    return m_min

def loc_disc (g_val, Rg, dist):
    cid = np.argmin(np.abs(Rg-dist))
    #loc_val = g_val[cid]
    d_val = (g_val[cid+1]-g_val[cid-1])/(Rg[cid+1]-Rg[cid-1])
    loc_val = g_val[cid] + d_val*(dist-Rg[cid])
    return loc_val

class Planet:
    def __init__(self, mass, mc, mg, f_comp, dist=10.0):
        self.mass = mass
        self.mc = mc
        self.mg = mg
        self.dist = dist 
        self.f_comp = f_comp
    def rhill (self, ms):
        return self.dist*(self.mass/3./ms)**(1./3.)

class PlanetEnv:
    def __init__(self, grid, alpha, mu, mass_star):
        self.alpha = alpha
        self.mu    = mu
        self.mass_star = mass_star
        self.grid  = grid

    def temp(self, T, dist):
        return loc_disc(T, self.grid.Rc, dist) 

    def sig_gas(self, disc, dist):    
        return loc_disc(disc.Sigma_gas, self.grid.Rc, dist) 

    def sig_dust(self, disc, dist):    
        return loc_disc(disc.Sigma_dust, self.grid.Rc, dist) 

    def mols(self, disc): 
        molc = disc.Molecules
        return molc

    def sig_mol(self, disc, dist):
        molc = disc.Molecules
        sigma_mol  = disc.Sigma_mol

        sig_mol_d = {}
        sig_mol_g = {}
        for mol in molc:
            s_mol_d = loc_disc(sigma_mol[mol.name][:,1], self.grid.Rc, dist)
            sig_mol_d[mol.name] = s_mol_d
            
            s_mol_g = loc_disc(sigma_mol[mol.name][:,0], self.grid.Rc, dist)
            sig_mol_g[mol.name] = s_mol_g
        return sig_mol_g, sig_mol_d

    def Stokes(self, disc, dist):
        st = disc.Stokes
        if np.isscalar(st):
            stokes = st
        else:
            stokes = loc_disc(st, self.grid.Re, dist)
        return stokes

    def vk(self, dist):
        return np.sqrt(self.mass_star*G_Msun/dist)

    def hr(self, T, dist):
        temp = loc_disc(T, self.grid.Rc, dist) 
        cs = np.sqrt(k_boltzmann/self.mu/m_hydrogen*temp)  
        return cs/(np.sqrt(self.mass_star*G_Msun/dist))

def gas_accretion(planet, p_env, disc, T, f_dm=0.5, f=0.2, kap=0.05, rho_c=5.):
    mass_p = planet.mass
    mc     = planet.mc
    mg     = planet.mg
    dist   = planet.dist
    gas_density = p_env.sig_gas(disc,dist)
    temperature = p_env.temp(T,dist)
    hr          = p_env.hr(T,dist)
    vk          = p_env.vk(dist)

    r_hill = dist*(mass_p/p_env.mass_star/3.)**(1./3.)
    omg_k  = np.sqrt(p_env.mass_star*G_Msun/dist**3)
    if mc > mg:
        dm_gas = (0.00175/f/f/ kap * (rho_c/5.5)**(-1./6.) * np.sqrt(81/temperature)
            *(mc/(Mearth/Msun))**(11./3.) * (0.1*Mearth/Msun / mg) * Mearth/1e6)/Msun
    else:
        dm_disc = -3.0*np.pi*gas_density*p_env.alpha*hr**2*vk*dist/Msun*yr
        dm_low  = 0.83 * omg_k * gas_density * (hr*dist)**2 * (r_hill/hr/dist)**(4.5) /Msun*yr
        dm_high = 0.14 * omg_k * gas_density * (hr*dist)**2 /Msun*yr
        dm_gas = np.min((dm_low,dm_high,-dm_disc*f_dm))
    return dm_gas 

def visc_mig(planet, p_env, disc, T):
    dist   = planet.dist
    gas_density = p_env.sig_gas(disc,dist)
    temperature = p_env.temp(T,dist)    
    hr          = p_env.hr(T,dist)
    alpha       = p_env.alpha
    dr          = dist*1e-2
    X0          = (dist-dr)**(3./2.)*p_env.hr(T,dist-dr)**2*p_env.vk(dist-dr)*p_env.sig_gas(disc,dist-dr)
    X1          = (dist+dr)**(3./2.)*p_env.hr(T,dist+dr)**2*p_env.vk(dist+dr)*p_env.sig_gas(disc,dist+dr)
    dr_X        = alpha * (X1-X0)/(2*dr)   
    vr = - 3/np.sqrt(dist)/gas_density *dr_X*yr
    return vr

def dk_mig(planet, p_env, disc, T):
    dist = planet.dist
    hr   = p_env.hr(T,dist)
    vk   = p_env.vk(dist)
    nu   = p_env.alpha*hr**2*vk*planet.dist
    gas_density = p_env.sig_gas(disc,dist) /Msun
    sig_std = (1e-7 / yr) /3./np.pi/nu
    #print(gas_density/sig_std)
    f_still = np.min((0.1*(planet.mass*1e3)**(-1.5), 2.0)) 
    f_mig   = np.min((3.5 * (gas_density/sig_std)**(0.55), 5.))
    tau_0   = gas_density*vk**2*dist**2*(planet.mass/p_env.mass_star/hr)**2 / Msun
    ang_mom = planet.mass*dist*vk
    v_visc  = visc_mig(planet, p_env, disc, T)
    a_dot   = np.min(((f_still*f_mig)*v_visc,0.1*v_visc))
    print(a_dot/v_visc,  f_mig, f_still)
    #if a_dot > 3*v_visc:
        #print(a_dot, v_visc)
    return a_dot 
    

def plansi_flux (plansi_frac, planet, p_env, disc, T):
    massdensity = plansi_frac*p_env.sig_dust(disc, planet.dist)/Msun
    rh = planet.rhill(p_env.mass_star)
    prob = np.min((3e-2*(Rau/planet.dist)**(1.19),3e-2))#probability (plansi, planet, p_env, disc, T)
    period = 2*np.pi*planet.dist/p_env.vk(planet.dist)/yr
    return 2*np.pi*massdensity*rh**2/period*prob

def mass_growth_pl(planet, p_env, disc, T, dt, plansi_frac):
    dist = planet.dist
    mol_comp = planet.f_comp

    dm_pla = plansi_flux(plansi_frac, planet, p_env, disc, T)
    dm_gas = gas_accretion(planet, p_env, disc, T)
    mc = planet.mc + dm_pla*dt
    #print (plansi_flux(plansi, planet, p_env, disc, T), dm_pla)
    mg = planet.mg + dm_gas*dt
    
    #print(dm_pla,dm_gas)
    sg = p_env.sig_gas(disc,dist)
    sd = p_env.sig_dust(disc,dist)

    molg, mold = p_env.sig_mol(disc,dist)
    mol_names = list(molg.keys())

    for mol in mol_names:
        dm_mol_g = dm_gas*(molg[mol]/sg)
        dm_mol_d = dm_pla*(mold[mol]/sd)
        mol_comp[mol][0] = mol_comp[mol][0] + dm_mol_g*dt
        mol_comp[mol][1] = mol_comp[mol][1] + dm_mol_d*dt
        #mass = planet.mass+dm*dt

    new_planet = Planet(mc+mg, mc, mg, mol_comp, planet.dist)
    return new_planet

def mig_planet(planet, p_env, disc, T, dt):
    a_dot = dk_mig(planet,p_env, disc,T)
    #a_dot = visc_mig(planet, p_env, disc, T)
    return planet.dist + a_dot*dt
