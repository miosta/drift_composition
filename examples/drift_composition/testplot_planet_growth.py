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
    for nn in range(nt-1):
        #if nn > nt-100:
        #    TT = lambda R: 150*(R/Rau)**-0.5
        #    alp = lambda R: p_env.alpha
        #    DM = DiscModel(grid, Mdot_gas*(1-(nn-nt+100)/100), alp, TT)
        #    DM.compute_dust_surface_density(Mdot_dust, Stokes)
        #    print( Mdot_gas*(1-(nn-nt+100)/100))
        planet = dumb.mass_growth_pl(planet, p_env, DM, T, dt, f_plansi) 
        planet.dist = np.max((dumb.mig_planet(planet, p_env, DM, T, dt) ,1e-6*Rau))
        if planet.dist < 1e-3*Rau:
            print('accreted')
        masses.append(planet.mass)
        mcs.append(planet.mc)
        mgs.append(planet.mg)
        mco_g.append(planet.f_comp[comp][0])
        mco_d.append(planet.f_comp[comp][1])
        rr.append(planet.dist)
    return np.array(masses),np.array(mcs),np.array(mgs),np.array(mco_g),np.array(mco_d),np.array(rr)

def std_mig_comp(planet, DM, p_env, T, f_plansi, dt, nt):
    planet_evo = np.array([planet])
    
    masses = [planet.mass]
    mcs = [planet.mc]
    mgs = [planet.mg]
    ms_comp = [planet.f_comp]
    mco_g = {}
    mco_d = {}
    species, abundances = get_molecular_properties()
    specs = [spec.name for spec in species]
    planet.f_comp = dict(zip(specs,np.zeros((len(specs),2))))
    for comp in specs:
        print(comp, planet.f_comp[comp][0])
        mco_g[comp] = [planet.f_comp[comp][0]]
        mco_d[comp] = [planet.f_comp[comp][1]]

    rr = [planet.dist]
    for nn in range(nt-1):
        #if nn > nt-100:
        #    TT = lambda R: 150*(R/Rau)**-0.5
        #    alp = lambda R: p_env.alpha
        #    DM = DiscModel(grid, Mdot_gas*(1-(nn-nt+100)/100), alp, TT)
        #    DM.compute_dust_surface_density(Mdot_dust, Stokes)
        #    print( Mdot_gas*(1-(nn-nt+100)/100))
        planet = dumb.mass_growth_pl(planet, p_env, DM, T, dt, f_plansi) 
        planet.dist = np.max((dumb.mig_planet(planet, p_env, DM, T, dt) ,1e-6*Rau))
        planet_evo = np.append(planet_evo, planet)
        if planet.dist < 1e-3*Rau:
            print('accreted')
        masses.append(planet.mass)
        mcs.append(planet.mc)
        mgs.append(planet.mg)
        for comp in specs:
            mco_g[comp].append(planet.f_comp[comp][0])
            mco_d[comp].append(planet.f_comp[comp][1])
        rr.append(planet.dist)
    #print(np.array(mco_d['CO2'])*Msun/Mearth)
    #return np.array(masses),np.array(mcs),np.array(mgs),np.array(rr),mco_g, mco_d ,specs
    return planet_evo

def multi_plot(ms,mcs,mgs,mco_g,mco_d,rrs,Nt,dt,titles):
    fig, (ax,ax2) = plt.subplots(2,1,sharex=True)
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    ax.set_ylabel("migrated distance [AU]")
    ax.plot(0,0,'o')
    #ax2 = ax.twinx()
    
    for (rr,c,t) in zip(rrs,colors,titles):
        ax.plot(np.arange(Nt)*dt,  (rr[0]-np.array(rr))/Rau, c=c,alpha=0.7, label='radius {}'.format(t))
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
    
    #ax2.set_yscale('log')
    #ax.set_yscale('log')
    #ax2.set_xscale('log')
    ax2.set_xlabel("time [yr]")
    ax2.set_ylabel("mass [M_Earth]")

    ax.legend()
    ax2.legend()
    #ax.set_title('M_dot =1e-9')
    #plt.savefig('mdot1e-9.png')
    plt.show()
    pass

def multi_plan(ms,mcs,mgs,rrs,Nt,dt,titles):
    fig, (ax) = plt.subplots()
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    #ax.set_ylabel("distance [AU]")
    #ax.plot(0,0,'o')

    for (m,mc,mg,rr,c,t) in zip(ms,mcs,mgs,rrs,colors,titles):
        print(rr.shape, m.shape)
        if rr[0] == rrs[0,0]:            
            ax.plot(np.array(rr)/Rau, np.array(m)*Msun/Mearth, c=c, label='total mass'.format())
            ax.plot(np.array(rr)/Rau, np.array(mg)*Msun/Mearth, '--', c=c, label='gas'.format())
            ax.plot(np.array(rr)/Rau, np.array(mc)*Msun/Mearth, ':', c=c, label='core'.format())
        else:
            ax.plot(np.array(rr)/Rau, np.array(m)*Msun/Mearth, '-', c=c)
            ax.plot(np.array(rr)/Rau, np.array(mg)*Msun/Mearth, '--', c=c)
            ax.plot(np.array(rr)/Rau, np.array(mc)*Msun/Mearth, ':', c=c)    

    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_xlim(1,4.5e1)
    ax.set_xlabel("Distance [au]")
    ax.set_ylabel(r"Planet Mass [$\mathrm{M}_{\oplus}$]")

    ax.legend()
    #ax2.legend(loc=1)
    #ax.set_title('M_dot =1e-9')
    #plt.savefig('mdot1e-9.png')
    plt.show()
    pass

def plot_planet_comp(masses,mcs,mgs,mco_g,mco_d,rr,Nt,dt,title=''): 
    fig, ax = plt.subplots()
    ax.plot(np.arange(Nt)*dt, np.array(masses)*Msun/Mearth, 'k-', label='total')
    ax.plot(np.arange(Nt)*dt, np.array(mcs)*Msun/Mearth, 'g:', label='core')
    ax.plot(np.arange(Nt)*dt, np.array(mgs)*Msun/Mearth, 'b-.', label='gas')
    ax.plot(np.arange(Nt)*dt, np.array(mco_g)*Msun/Mearth, 'c:', label='CO gas')
    ax.plot(np.arange(Nt)*dt, np.array(mco_d)*Msun/Mearth, 'c--', label='CO dust')
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_xlabel("time [yr]")
    ax.set_ylabel("mass [M_Earth]")
    ax.legend()
    #ax.savefig('frag_masses.png')
    ax.text(0, 1, title)
    plt.show()
    pass

def lecture_plot():
    Mdot_gas = 1e-8
    Mdot_dust = 1e-9
    Stokes = 0.01

    T = lambda R: 150*(R/Rau)**-0.5
    alpha = lambda R: 1e-3

    species, abundances = get_molecular_properties()
    f_comp = dict(zip([spec.name for spec in species],np.zeros((len(species),2))))
    grid = Grid(0.1*Rau, 300*Rau, 512)

    p_env = PlanetEnv(grid, alpha(grid.Rc), 2.35, 1.0)

    dt = 1000
    Nt = 450
    f_plansi = 1e-1
    frac_gc = 0.1
    init_m  = 0.50

    ms2=np.zeros(Nt)
    mcs2=np.zeros(Nt)
    mgs2=np.zeros(Nt)
    rrs2=np.zeros(Nt)

    for r,mm in zip((15,30,40),(0.1,0.1,0.1,0.1,0.1,0.1)):
        DM = DiscModel(grid, Mdot_gas, alpha, T)
        DM.compute_dust_surface_density(Mdot_dust, Stokes)
        p_env = PlanetEnv(grid, 1e-3, 2.35, 1.0)
        planet = Planet(mm*init_m*Mearth/Msun, mm*init_m*(1-frac_gc)*Mearth/Msun, mm*init_m*(frac_gc)*Mearth/Msun, f_comp,r*Rau)
        m,mc,mg,mco_g,mco_d,rr = std_mig(planet, DM, p_env, T(grid.Rc),f_plansi, dt, Nt)
        ms2=np.vstack((ms2,m))
        mcs2=np.vstack((mcs2,mc))
        mgs2=np.vstack((mgs2,mg))
        rrs2=np.vstack((rrs2,rr))
    
        multi_plan(ms2[1:],mcs2[1:],mgs2[1:],rrs2[1:],Nt,dt,(15,30,40))
    pass


def main():
    Mdot_gas = 1e-8
    Mdot_dust = 1e-9
    Stokes = 0.01

    T = lambda R: 150*(R/Rau)**-0.5
    alpha = lambda R: 1e-3

    species, abundances = get_molecular_properties()
    print([spec.name for spec in species])
    grid = Grid(0.1*Rau, 300*Rau, 512)

    p_env = PlanetEnv(grid, alpha(grid.Rc), 2.35, 1.0)

    dt = 5000
    Nt = 100
    f_plansi = 5e-1

    ms2=np.zeros(Nt)
    mcs2=np.zeros(Nt)
    mgs2=np.zeros(Nt)
    mco2_gs=np.zeros(Nt)
    mco2_ds=np.zeros(Nt)
    rrs2=np.zeros(Nt)

    frac_gc = 0.1
    init_m  = 5.0

    for r,mm in zip((8,16,32,8),(1,1,1,1,1,1)):
        f_comp = dict(zip([spec.name for spec in species],np.zeros((len(species),2))))
        alpha = lambda R: 1e-3
        DM = DiscModel(grid, Mdot_gas, alpha, T)
        DM.compute_dust_surface_density(Mdot_dust, Stokes)
        species, abundances = get_molecular_properties()
        DM.compute_chemistry(species, abundances )
        p_env = PlanetEnv(grid, 1e-3, 2.35, 1.0)
        planet = Planet(mm*init_m*Mearth/Msun, mm*init_m*(1-frac_gc)*Mearth/Msun, mm*init_m*(frac_gc)*Mearth/Msun, f_comp,r*Rau)
        m,mc,mg,mco_g,mco_d,rr = std_mig(planet, DM, p_env, T(grid.Rc),f_plansi, dt, Nt, comp='H2O')
        ms2=np.vstack((ms2,m))
        mcs2=np.vstack((mcs2,mc))
        mgs2=np.vstack((mgs2,mg))
        mco2_gs=np.vstack((mco2_gs,mco_g))
        mco2_ds=np.vstack((mco2_ds,mco_d))
        rrs2=np.vstack((rrs2,rr))

    #print(mco2_gs[4])
    #multi_plot(ms2[1:],mcs2[1:],mgs2[1:],mco2_gs[1:],mco2_ds[1:],rrs2[1:],Nt,dt,(8,16,32,64))
    plot_planet_comp(ms2[4],mcs2[4],mgs2[4],mco2_gs[4],mco2_ds[4],rrs2[4],Nt,dt,'')

    ms2=np.zeros(Nt)
    mcs2=np.zeros(Nt)
    mgs2=np.zeros(Nt)
    mco2_gs=np.zeros(Nt)
    mco2_ds=np.zeros(Nt)
    rrs2=np.zeros(Nt)

    f_comp = dict(zip([spec.name for spec in species],np.zeros((len(species),2))))
    planet = Planet(init_m*Mearth/Msun, init_m*(1-frac_gc)*Mearth/Msun, init_m*(frac_gc)*Mearth/Msun, f_comp, 8*Rau)
    planet_evo = std_mig_comp(planet, DM, p_env, T(grid.Rc),f_plansi, dt, Nt)
    
    ms3 = [p.mass for p in planet_evo]
    mgf = [p.mg for p in planet_evo]
    mdf = [p.mc for p in planet_evo]
    mg_cof = [p.f_comp['CO'][0] for p in planet_evo]
    md_cof = [p.f_comp['CO'][1] for p in planet_evo]
    mg_hof = [p.f_comp['H2O'][0] for p in planet_evo]
    md_hof = [p.f_comp['H2O'][1] for p in planet_evo]

    print(ms3)

    plt.plot(np.arange(Nt)*dt, np.array(ms3)*Msun/Mearth, 'k--', label='total')
    plt.plot(np.arange(Nt)*dt, np.array(mgf)*Msun/Mearth, 'k-', label='total')
    plt.plot(np.arange(Nt)*dt, np.array(mg_cof)*Msun/Mearth, 'c:', label='CO gas')
    plt.plot(np.arange(Nt)*dt, np.array(md_cof)*Msun/Mearth, 'c--', label='CO dust')
    plt.plot(np.arange(Nt)*dt, np.array(mg_hof)*Msun/Mearth, 'b:', label='H2O gas')
    plt.plot(np.arange(Nt)*dt, np.array(md_hof)*Msun/Mearth, 'b--', label='H2O dust')
    plt.yscale('log')
    plt.show()

if '__main__'==__name__:
    main()
