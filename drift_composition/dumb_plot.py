from drift_composition.constants import Mearth, Msun, Rau, yr
from drift_composition.grid import Grid
from drift_composition.disc import DiscModel
from drift_composition.molecule import get_molecular_properties
from drift_composition.dumb_planet import Planet, PlanetEnv
import drift_composition.dumb_planet as dumb
import matplotlib.pyplot as plt
import numpy as np

def std_mig(planet, DM, p_env, T, f_plansi, dt, nt, comp='CO'):
    masses = [planet.mass]
    mcs = [planet.mc]
    mgs = [planet.mg]
    mco_g = [planet.f_comp[comp][0]]
    mco_d = [planet.f_comp[comp][1]]
    rr = [planet.dist]
    for _ in range(nt-1):
        planet = dumb.mass_growth_pl(planet, p_env, DM, T, dt, f_plansi)
        planet.dist = np.max((dumb.mig_planet(planet, p_env, DM, T, dt),1e-6))
        masses.append(planet.mass)
        mcs.append(planet.mc)
        mgs.append(planet.mg)
        mco_g.append(planet.f_comp[comp][0])
        mco_d.append(planet.f_comp[comp][1])
        rr.append(planet.dist)
    return np.array(masses),np.array(mcs),np.array(mgs),np.array(mco_g),np.array(mco_d),np.array(rr)

def multi_plot(ms,mcs,mgs,mco_g,mco_d,rrs,Nt,dt,titles):
    fig, ax = plt.subplots()
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    ax.set_ylabel("distance [AU]")
    ax.plot(0,0,'o')
    ax2 = ax.twinx()
    
    for (rr,c,t) in zip(rrs,colors,titles):
        ax.plot(np.arange(Nt)*dt,  np.array(rr)/Rau, c=c,alpha=0.7, label='radius {}'.format(t))
    for (m,mc,mg,rr,c,t) in zip(ms,mcs,mgs,rrs,colors,titles):
        #print(rr)
        if rr[0] == rrs[0,0]:
            ax2.plot(np.arange(Nt)*dt, np.array(m)*Msun/Mearth, '-', c=c, label='mass {}'.format(t))
            ax2.plot(np.arange(Nt)*dt, np.array(mc)*Msun/Mearth, ':', c=c, label='core {}'.format(t))
            ax2.plot(np.arange(Nt)*dt, np.array(mg)*Msun/Mearth, '--', c=c,label='gas {}'.format(t))
        else:
            ax2.plot(np.arange(Nt)*dt, np.array(m)*Msun/Mearth, '-', c=c)
            ax2.plot(np.arange(Nt)*dt, np.array(mc)*Msun/Mearth, ':', c=c,)
            ax2.plot(np.arange(Nt)*dt, np.array(mg)*Msun/Mearth, '--', c=c)
    
    ax2.set_yscale('log')
    #ax2.set_xscale('log')
    ax.set_xlabel("time [yr]")
    ax2.set_ylabel("mass [M_Earth]")

    ax.legend(loc=2)
    ax2.legend(loc=1)
    #ax.set_title('M_dot =1e-9')
    #plt.savefig('mdot1e-9.png')
    plt.show()
    pass

def plot_planet_comp(masses,mcs,mgs,mco_g,mco_d,rr,Nt,dt,title=''): 
    fig, ax = plt.subplots()
    ax.plot(np.arange(Nt)*dt, np.array(masses)*Msun/Mearth, 'k-', label='total')
    ax.plot(np.arange(Nt)*dt, np.array(mcs)*Msun/Mearth, 'g:', label='core')
    ax.plot(np.arange(Nt)*dt, np.array(mgs)*Msun/Mearth, 'b-.', label='gas')
    ax.plot(np.arange(Nt)*dt, np.array(mco_g)*Msun/Mearth, 'c--', label='CO gas')
    ax.plot(np.arange(Nt)*dt, np.array(mco_d)*Msun/Mearth, 'm--', label='CO dust')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("time [yr]")
    ax.set_ylabel("mass [M_Earth]")
    ax.legend()
    #ax.savefig('frag_masses.png')
    ax.text(0, 1, title)
    plt.show()
    pass

Mdot_gas = 1e-8
Mdot_dust = 1e-9
Stokes = 0.01

T = lambda R: 150*(R/Rau)**-0.5
alpha = lambda R: 1e-3

species, abundances = get_molecular_properties()
f_comp = dict(zip([spec.name for spec in species],np.zeros((len(species),2))))
grid = Grid(0.1*Rau, 300*Rau, 512)

planet_ini = Planet(0.1*Mearth/Msun, 0.1*Mearth/Msun, 1e-10, f_comp, 100.0*Rau)
p_env = PlanetEnv(grid, alpha(grid.Rc), 2.35, 1.0)

dt = 1000
Nt = 1000
f_plansi = 5e-1

#m,mcs,mgs,mco_g,mco_d,rr = std_mig(planet, DM, p_env, T(grid.Rc), dt, Nt, f_plansi)


ms2=np.zeros(Nt)
mcs2=np.zeros(Nt)
mgs2=np.zeros(Nt)
mco2_gs=np.zeros(Nt)
mco2_ds=np.zeros(Nt)
rrs2=np.zeros(Nt)

frac_gc = 0.25
init_m  = 5.0

for r in (1,5,10,15,25):
    alpha = lambda R: 1e-3
    DM = DiscModel(grid, Mdot_gas, alpha, T)
    DM.compute_dust_surface_density(Mdot_dust, Stokes)
    p_env = PlanetEnv(grid, 1e-3, 2.35, 1.0)
    planet = Planet(init_m*Mearth/Msun, init_m*(1-frac_gc)*Mearth/Msun,  init_m*(frac_gc)*Mearth/Msun, f_comp,r*Rau)
    m,mc,mg,mco_g,mco_d,rr = std_mig(planet, DM, p_env, T(grid.Rc),f_plansi, dt, Nt)
    ms2=np.vstack((ms2,m))
    mcs2=np.vstack((mcs2,mc))
    mgs2=np.vstack((mgs2,mg))
    mco2_gs=np.vstack((mco2_gs,mco_g))
    mco2_ds=np.vstack((mco2_ds,mco_d))
    rrs2=np.vstack((rrs2,rr))
    
multi_plot(ms2[1:],mcs2[1:],mgs2[1:],mco2_gs[1:],mco2_ds[1:],rrs2[1:],Nt,dt,(1,5,10,15,25))

