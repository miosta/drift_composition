import numpy as np
from drift_composition.constants import k_boltzmann, m_hydrogen, G_Msun, Rau, Msun, yr, Mearth, G_Newton
from drift_composition.atoms import molecule_mass
from drift_composition.simple_planet import PlanetEnv, loc_disc, gas_accretion, visc_mig, dk_mig, gap_dens, mass_growth_pl 

class Planet:
    '''Planet data-object
    
    All masses in Solar masses, distance in cm
    mass, mc, mg: float mass of total, core, gas content
    f_comp: dict key - species, values array of [gas, solid] mass
    '''
    def __init__(self, mass, mc, mg, f_comp, dist=10.0*Rau, time =0.):
        self.mass = mass
        self.mc = mc
        self.mg = mg
        self.dist = dist 
        self.f_comp = f_comp
        self.time = time
    def rhill (self, ms):
        return self.dist*(self.mass/3./ms)**(1./3.)

    def Rcapture(self, pl, T, p_env, disc):
        """Capture radius"""
        if self.mg > self.mc:
            # Detached, Valletta & Helled (2021)
            # At 10^7 yr
            R = np.array([12.80662188, -50.86303789, 382.66267044,
                 -1388.57741163, 1902.60362959])
            x = self.mc/self.mg
            return 1e9 * np.sum(R*x**np.arange(0, 5))
        else:
            # Embedded phase, as in Danti & Bitsch (2023)
            R_B = G_Newton*(self.mc + self.mg) * Msun / p_env.sound_speed(T, self.dist)**2
            R_H = self.rhill(p_env.mass_star)*Rau
            R = 1/(1/R_B + 4/R_H)

            D = 0.44 # Drag coefficient
            rho_s = 0.66*pl.r*pl.rho/(D*R_H)

            return R*R / (R + R_B*np.log(np.maximum(rho_s/disc.rho_cgs_z(self.dist),1)))


class Planetesimals:
    def __init__(self, grid, r=10, e0=1e-4, i0=None, rho=5, fSigma=1e-3,  Ncells=1000, fdot=0):
        self.r = r*1e5
        self.ae = grid.Re
        self.a = 0.5*(self.ae[1:]+self.ae[:-1])
        self.e = np.full_like(self.a, e0)
        self.i = self.e/2 if i0 is None else np.full_like(self.a,i0)
        self.fSigma = np.full_like(self.a, fSigma)
        self.rho = rho 
        self.fdot = fdot

    def PQ_VS(self, hill_scale):
            e, i = self.e*hill_scale, self.i*hill_scale
            e2, i2 = e*e, i*i 
            B = i/e
            L2 = i2*(e2 + i2)**2 / 144

            B = np.maximum(np.minimum(B, 1), 0)

            IP = (B - 0.36251) / (0.061547 + B*(0.16112 + 0.054473*B))
            IQ = (0.71946 - B) / (0.21239  + B*(0.49764 + 0.14369 *B))

            l2 = 72*np.log1p(L2) / (np.pi*i*e)

            P = 7.3*e2                 *np.log1p(10*L2/e2) / L2 + IP*l2 
            Q = (0.4*i2/e + 0.02*i*e2) *np.log1p(10*L2*e ) / L2 + IQ*l2

            return P, Q

    def Pcoll(self, Rc, Rpl, hill_scale):
            #Radius normalized to RHill             
            e, i = self.e*hill_scale, self.i*hill_scale
            e2, i2 = e*e, i*i 
            B = i/e

            IF = (1/B +  0.95925 + 0.77251*B) / (0.13142 + B* 0.12295)
            IG = (1/B +  0.3996             ) / (0.0369 +  B*(0.048333 + 0.006874*B))

            PH = ((Rc + Rpl)**2/(2*np.pi))  *(IF   + 6*IG/((Rc+Rpl)*e2))
            PM = ((Rc + Rpl)**2/(4*np.pi*i))*(17.3 + 232 / (Rc+Rpl))
            PL = 11.3*(Rc + Rpl)**0.5

            return np.minimum(PM, (PH**-2 + PL**-2)**-0.5)

    @property
    def m(self):
        return 4*np.pi*self.rho/3 * (self.r)**3 / Mearth

def plansi_excite (plansi, planets, p_env, disc, T, feeding_zone=10.):
    d = disc
    pl = plansi

    Porb = yr * (p_env.mass_star/pl.a**3)**-0.5

    vK0 = 29.78e5
    vK = vK0*np.sqrt(p_env.mass_star/pl.a)
    tau = 8*pl.rho*pl.r/(3*0.44*d.rho_cgs_z(pl.a)*vK)
    eta = p_env.eta(pl.a, T)
    b = feeding_zone

    e2, i2 = pl.e**2, pl.i**2
    f = pl.fSigma
    f0 = pl.fSigma

    de2dt = 0
    di2dt = 0

    for p in planets:
        RH = p_env.R_Hill(p)

        P, Q = plansi.PQ_VS(p.dist/RH)

        t = (p.mc + p.mg)*Msun/(3*b*p_env.mass_star*Msun*Porb)

        fD = 1 / (1 + ((pl.a-p.dist)/(0.5*b*RH))**10)

        de2dt += t*P*fD
        di2dt += t*Q*fD

    h = (2*pl.m * Mearth / (3*p_env.mass_star*Msun))**(1/3.)
    P2, Q2 = plansi.PQ_VS(4/h**2)
                
    k = (np.pi/3) * (disc.Sigma_gas*f*(pl.a*Rau)**2 * h /(p_env.mass_star*Msun*Porb))

    de2dt += k*P2 
    di2dt += k*Q2 

    de2dt = np.maximum(de2dt, 0)
    di2dt = np.maximum(di2dt, 0)

        # Drag - Adachi+ (1976)
    tau_e2 = (2/tau)*np.sqrt(eta*eta + 0.625*e2 + 0.5**i2)
    tau_i2 = (1/tau)*np.sqrt(eta*eta + 0.625*e2 + 0.5**i2)
    return tau_e2, tau_i2, de2dt, di2dt


def plansi_fluxes(pl, p, p_env, disc, T, feeding_zone=10.):
    f = pl.fSigma
    b = feeding_zone
    Porb = yr * (p_env.mass_star/pl.a**3)**-0.5
    e2, i2 = pl.e**2, pl.i**2
                
    # Accretion/Ejection
    RH = p_env.R_Hill(p)
    Rc  = p.Rcapture(pl, T, p_env, disc)/(RH*Rau)
    Rpl = pl.r/(RH*Rau)

    Pc = pl.Pcoll(Rc, Rpl, p.dist/RH)
    fej = (pl.a*(p.mc + p.mg)*Msun/(p_env.mass_star*Msun*Rc*RH))**2 # Ejection prob.

    fD = 1 / (1 + ((pl.a-p.dist)/(0.5*b*RH))**10)
    tau_f = fD*(RH/(b*p.dist))*Pc/Porb

    dMcdt = disc.Sigma_gas*tau_f * np.pi*np.diff(pl.ae**2)*Rau**2

    return tau_f, fej, dMcdt/Msun*yr

    
def pebble_accretion(planet, p_env, disc, T):
    '''Following Bitsch 2015'''
    mass_p = planet.mass
    dist   = planet.dist
    hr        = p_env.hr(T,dist)
    stokes    = np.max((p_env.Stokes(disc,dist),1e-10))
    mass_star = p_env.mass_star
    alpha     = p_env.alpha
    _ , sig_dust = p_env.sigs_tot(disc, dist)
    pebble_density = sig_dust
    r_hill = planet.rhill(mass_star)
    
    v_hill = r_hill * np.sqrt(mass_star*G_Msun / dist**3)
    h_peb = hr * dist / np.sqrt(1 + stokes/alpha)
    dm_2d = min(2.0 * (stokes / 0.1)**(2. / 3.) * r_hill * v_hill * pebble_density,2.0 * r_hill * v_hill * pebble_density)
    dm_3d = dm_2d * (r_hill * np.pi**0.5 / 2**1.5 / h_peb *(stokes/0.1)**(1./3.))
    crit_h = np.pi* (stokes/0.1)**(1./3.) * r_hill /2/np.sqrt(2*np.pi)
    if h_peb > crit_h: dm_peb = dm_2d
    else: dm_peb = dm_3d
    if planet.mass > 20 * (p_env.hr(T, dist)/0.05)**3. * Mearth/Msun:
        dm_peb = 0
    return dm_peb/Msun*yr

def mass_growth_plansi(planets, p_env, disc, T, dt, plansi):
    f = plansi.fSigma

    e0, i0 = plansi.e, plansi.i
    ei = plansi.e**2, plansi.i**2
    for i in range(2):
        tau_e2, tau_i2, de2dt, di2dt = plansi_excite (plansi, planets, p_env, disc, T, feeding_zone=10.)
        e2 = 0.5*(e0*e0 + e0*np.sqrt((e0**2 + de2dt*dt*(1 + dt*tau_e2))))/(1 + dt*tau_e2)
        i2 = 0.5*(i0*i0 + i0*np.sqrt((i0**2 + di2dt*dt*(1 + dt*tau_i2))))/(1 + dt*tau_i2)
        e2 = np.sqrt(e2*ei[0]) 
        i2 = np.sqrt(i2*ei[1])
        ei = e2, i2

    plansi.e = np.sqrt(e2)
    plansi.i = np.sqrt(i2)
    
    for npi, planet in enumerate(planets):
        dist = planet.dist
        tau_f, fej, dm_f = plansi_fluxes(plansi, planet, p_env, disc, T, feeding_zone=10.)
        f /= (1 + dt*tau_f*(1+fej))
        dm_pla = np.sum(f*dm_f)
        dm_peb = pebble_accretion(planet, p_env, disc, T)
        dm_gas = gas_accretion(planet, p_env, disc, T)
        mc = planet.mc + dm_pla*dt + dm_peb*dt
    #print (plansi_flux(plansi, planet, p_env, disc, T), dm_pla)
        mg = planet.mg + dm_gas*dt
    
    #print(dm_pla,dm_gas)
        sg , sd = p_env.sigs_tot(disc, dist)
        molg, mold = p_env.sig_mol(disc,dist)
        mol_names = list(molg.keys())
    
        mol_comp = {
            k: np.array([
                v[0] + dm_gas*(molg[k]/sg)*dt,
                v[1] + (dm_pla+dm_peb)*(mold[k]/sd)*dt
            ])
            for k, v in planet.f_comp.items()
        }

        planets[npi] = Planet(mc+mg, mc, mg, mol_comp, planet.dist, planet.time+dt)

    plansi.fSigma = f
    return planets, plansi

def mig_planet(planet, p_env, disc, T, dt):
    #a_dot = dk_mig(planet,p_env, disc,T)
    a_dot = 0.25*visc_mig(planet, p_env, disc, T)
    return planet.dist + a_dot*dt

def multi_evo_comp(planets_ini, DM, p_env, T, plansi, dt_ini, nt, final_radius= 1e-3, final_mass= 2e-3, final_time= 1e7):
    multi_evo = np.array(planets_ini)
    print(np.shape(multi_evo))
    for npi,planet_in in enumerate(planets_ini):
        planet_evo = np.array([planet_in])
        multi_evo[npi] = planet_evo
    r_grid     = p_env.grid.Rc
    dt_adapt   = dt_ini
    t          = 0.
    nn = 0
    for nn in range(nt-1):
        #evolution

        planets, plansi = mass_growth_plansi(planets_ini, p_env, DM, T, dt_adapt, plansi) 
        for npi,(planet, planet_evo) in enumerate(zip(planets, multi_evo)):
            planet.dist = np.max((mig_planet(planet, p_env, DM, T, dt_adapt) ,1e-4*Rau))
            if nn%10==0:
                multi_evo[npi] = np.append(multi_evo[npi], planet)
            planets_ini[npi] = planet
        t += dt_adapt        
        
        #adaptive time step
        dt_ps = np.ones_like(planets_ini)
        for nps,planet in enumerate(planets_ini):
            ir = np.argmin(abs(r_grid-planet.dist))
            dr = r_grid[ir+1]-r_grid[ir]
            dt_ps[nps] = min((abs(dr / dk_mig(planet, p_env, DM, T))*0.5,
                        abs(planet_in.mass*5e-2 / gas_accretion(planet_in, p_env, DM, T))*0.5,
                        10000))
            dt_ps[nps] = max((dt_adapt, 100))
        dt_adapt = np.min(dt_ps)

        #end sim
        end = False
        for planet in planets_ini:
            if planet.dist < final_radius*Rau:
                print('accreted at t = {}; n= {}'.format(planet.time,nn))
                end = True
            elif planet.mass > final_mass:
                print('2Mjup at t = {}; n= {}'.format(t,nn))
                end = True
        if t > final_time:
            print('1e7 yr evolution reached ; n= {}'.format(t,nn))
            end = True
        if end:
            break

    for k,m in enumerate(multi_evo):
        multi_evo[k] = m[:nn//10]
    return multi_evo, nn//10

def multi_evo_naiv(planets_ini, DM, p_env, T, f_plansi, dt_ini, nt, final_radius= 1e-3, final_mass= 2e-3, final_time= 1e7):
    multi_evo = np.array(planets_ini)
    print(np.shape(multi_evo))
    for npi,planet_in in enumerate(planets_ini):
        planet_evo = np.array([planet_in])
        multi_evo[npi] = planet_evo
    r_grid     = p_env.grid.Rc
    dt_adapt   = dt_ini
    t          = 0.
    nn = 0
    for nn in range(nt-1):
        #evolution
        for npi,(planet, planet_evo) in enumerate(zip(planets_ini, multi_evo)):
            planet = mass_growth_pl(planet, p_env, DM, T, dt_adapt, f_plansi) 
            planet.dist = np.max((mig_planet(planet, p_env, DM, T, dt_adapt) ,1e-4*Rau))
            if nn%10==0:
                multi_evo[npi] = np.append(multi_evo[npi], planet)
            planets_ini[npi] = planet
        t += dt_adapt        
        
        #adaptive time step
        dt_ps = np.ones_like(planets_ini)
        for nps,planet in enumerate(planets_ini):
            ir = np.argmin(abs(r_grid-planet.dist))
            dr = r_grid[ir+1]-r_grid[ir]
            dt_ps[nps] = min((abs(dr / dk_mig(planet, p_env, DM, T))*0.5,
                        abs(planet_in.mass*5e-2 / gas_accretion(planet_in, p_env, DM, T))*0.5,
                        10000))
            dt_ps[nps] = max((dt_adapt, 100))
        dt_adapt = np.min(dt_ps)

        #end sim
        end = False
        for planet in planets_ini:
            if planet.dist < final_radius*Rau:
                print('accreted at t = {}; n= {}'.format(planet.time,nn))
                end = True
            elif planet.mass > final_mass:
                print('2Mjup at t = {}; n= {}'.format(t,nn))
                end = True
        if t > final_time:
            print('1e7 yr evolution reached ; n= {}'.format(t,nn))
            end = True
        if end:
            break

    for k,m in enumerate(multi_evo):
        multi_evo[k] = m[:nn//10]
    return multi_evo, nn//10
