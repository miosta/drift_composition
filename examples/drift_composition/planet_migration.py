import numpy as np
from drift_composition.constants import k_boltzmann, m_hydrogen, G_Msun, Rau, Msun, yr, Mearth
from drift_composition.planet_growth import Planet, PlanetEnv

def torq_0 (planet, p_env, disc, T):
    ms = p_env.mass_star
    mp = planet.mass
    q  = mp/ms
    dist = planet.dist
    h    = p_env.hr(T, dist)
    sig  = p_env.sig_gas(disc, dist)/Msun
    vk   = p_env.vk(dist)
    return (q/h)**2*sig*dist**2*vk**2

def a_dot_typeI(planet, p_env, disc, T, flaring, gas_slope, iso=True):
    beta = 1- 2*flaring
    if iso==True:
        gamma = 1
    else:
        gamma = 1.4 #fix to t_eff
    tq0 = torq_0(planet,p_env,disc,T)
    b = 2./3.
    lindblad = -tq0*gamma*(2.5+1.7*beta-0.1*gas_slope)*b**(0.71)
    corotate = tq0*(1.1*(1.5-gas_slope)*b + 2.2*beta*b**0.71 - 1.4*beta*b**1.26) #fix
    dist = planet.dist
    vk   = p_env.vk(dist)
    mp = planet.mass
    ang_mom  = mp*vk*dist 
    norm_torq = (lindblad + corotate)/ang_mom
    norm_adot = 2 *norm_torq
    print((lindblad+corotate)/tq0, tq0/ang_mom*yr)
    return norm_adot*dist*yr

def crit_k(planet,p_env,disc,T):
    K = (planet.mass/p_env.mass_star)**2 *pow(p_env.hr(T, planet.dist),-5) / p_env.alpha
    return K

def a_dot_typeII(planet, p_env, disc, T):
    "Kanagawa 2018, cross flow K description"
    K = crit_k(planet, p_env, disc, T)
    sig_min = p_env.sig_gas(disc,planet.dist)/(1+0.4*K)
    net_torq = - 0.4 * (planet.mass/p_env.mass_star)**2*  p_env.hr(T, planet.dist)**(-3) * planet.dist**2 * p_env.vk(planet.dist)**2 * sig_min / Msun
    dist = planet.dist
    vk   = p_env.vk(planet.dist)
    mp = planet.mass
    net_ang  = mp*vk*dist 
    print(net_torq/net_ang*yr)
    a_dot = planet.dist* 2 * (net_torq/net_ang) #in cm/s
    return a_dot*yr

def mig_planet(planet, p_env, disc, T, flaring, gas_slope, dt, iso=True):
    k = crit_k(planet, p_env, disc, T)
    if k < 20:
        a_dot = a_dot_typeI(planet, p_env, disc, T, flaring, gas_slope, iso)
    else:
        a_dot = a_dot_typeII(planet, p_env, disc, T)
    return planet.dist + a_dot*dt