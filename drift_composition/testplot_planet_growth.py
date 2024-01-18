from drift_composition.constants import Mearth, Msun, Rau, yr
from drift_composition.grid import Grid
from drift_composition.disc import DiscModel
from drift_composition.molecule import get_molecular_properties
from drift_composition.simple_planet import Planet, PlanetEnv
from drift_composition.atoms import atoms_in_molecule, ELEMENT_MASS
from drift_composition.simple_reduction import Evolution, atom_mass
import drift_composition.simple_planet as dumb
import matplotlib.pyplot as plt
import numpy as np

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

def plot_planet_comp(planet_evo, Nt, dt, comp='CO', title=''):
    e = Evolution(planet_evo, Nt, dt)

    fig, ax = plt.subplots()
    ax.plot(e.time, e.masses*Msun/Mearth, 'k-', label='total')
    ax.plot(e.time, e.mcs*Msun/Mearth, 'g:', label='core')
    ax.plot(e.time, e.mgs*Msun/Mearth, 'b-.', label='gas')
    ax.plot(e.time, e.f_comps[comp][0]*Msun/Mearth, 'c:', label='CO gas')
    ax.plot(e.time, e.f_comps[comp][1]*Msun/Mearth, 'c--', label='CO dust')
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_xlabel("time [yr]")
    ax.set_ylabel("mass [M_Earth]")
    ax.legend()
    #ax.savefig('frag_masses.png')
    ax.text(0, 1, title)
    plt.show()
    pass

def plot_planet(planet_evo,Nt,dt):
    e = Evolution(planet_evo, Nt, dt)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plt.plot(e.time, e.masses*Msun/Mearth, 'k-', label='mass total')
    plt.plot(e.time, e.mgs*Msun/Mearth, 'k--', label='gas total')
    plt.plot(e.time, e.mcs*Msun/Mearth, 'k:', label='dust total')
    for name, c in zip(list(planet_evo[0].f_comp.keys()),colors):
        plt.plot(e.time, e.f_comps[name][0]*Msun/Mearth, '--', c=c, label=name)
        plt.plot(e.time, e.f_comps[name][1]*Msun/Mearth, ':', c=c)
    plt.ylim(1e-5,1e3)
    plt.ylabel(r'mass [$M_{\oplus}$]')
    plt.xlabel('time [yr]')

    plt.legend()
    plt.yscale('log')
    plt.show()

    plt.plot(e.rs, e.masses*Msun/Mearth, 'k-', label='mass total')
    plt.plot(e.rs, e.mgs*Msun/Mearth, 'k--', label='gas total')
    plt.plot(e.rs, e.mcs*Msun/Mearth, 'k:', label='dust total')
    for name, c in zip(list(planet_evo[0].f_comp.keys()),colors):
        plt.plot(e.rs, e.f_comps[name][0]*Msun/Mearth, '--', c=c, label=name)
        plt.plot(e.rs, e.f_comps[name][1]*Msun/Mearth, ':', c=c)
    plt.ylim(1e-5,1e3)
    plt.xlabel('radius [au]')
    plt.ylabel(r'mass [$M_{\oplus}$]')
    plt.legend()
    plt.yscale('log')
    plt.show()
    pass

def lecture_plot():
    Mdot_gas = 1e-8
    Mdot_dust = 1e-9
    Stokes = lambda R: 0.01

    T = lambda R: 150*(R/Rau)**-0.5
    alpha = lambda R: 1e-3

    species, abundances = get_molecular_properties()
    f_comp = dict(zip([spec.name for spec in species],np.zeros((len(species),2))))
    f_comp['H2'] = np.zeros(2)
    f_comp['Si'] = np.zeros(2)
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

def plot_CtoO(planet_evo,Nt,dt,atm1,atm2,solar = 1.):
    e = Evolution(planet_evo, Nt, dt)
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharex='col', sharey='row')

    atoms = (atm1, atm2)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for (x,a1,a2) in zip((e.time,e.rs),(ax1,ax2),(ax3,ax4)):
        a1.plot(x, e.masses*Msun/Mearth, 'k-', label='mass total')
        a1.plot(x, e.mgs*Msun/Mearth, 'k--', label='gas total')
        a1.plot(x, e.mcs*Msun/Mearth, 'k:', label='dust total')
        for atom,c in zip(atoms,colors):
            a1.plot(x, e.f_atms[atom][0]*Msun/Mearth, '--', c=c, label=atom)
            a1.plot(x, e.f_atms[atom][1]*Msun/Mearth, ':', c=c)

        a2.plot(x, (e.f_atms[atm1][0]/e.f_atms[atm2][0])*(ELEMENT_MASS[atm2]/ELEMENT_MASS[atm1])/solar, label='gas only')
        a2.plot(x, ((e.f_atms[atm1][0]+e.f_atms[atm1][1])/(e.f_atms[atm2][0]+e.f_atms[atm2][1])*(ELEMENT_MASS[atm2]/ELEMENT_MASS[atm1]))/solar, label='gas+dust')

    ax3.set_xlabel('time [yr]')
    ax4.set_xlabel('distance [au]')
    ax1.set_ylabel(r'mass [$M_{\oplus}$]')
    ax3.set_ylabel(r'{}/{}'.format(atm1,atm2))

    ax1.set_ylim(1e-3,1e3)
    ax2.set_ylim(1e-3,1e3)
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    ax1.legend()
    ax3.legend()
    plt.show()
    pass

def plot_atoms(planet_evo,Nt,dt):

    e = Evolution(planet_evo, Nt, dt)
    ms3 = e.masses

    #print(atom_mass(planet_evo[0].f_comp))
    at_mass = [atom_mass(p.f_comp) for p in planet_evo]

    atoms = list(ELEMENT_MASS.keys())

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plt.plot(e.rs, ms3*Msun/Mearth, 'k-', label='mass total')
    plt.plot(e.rs, e.mgs*Msun/Mearth, 'k--', label='gas total')
    plt.plot(e.rs, e.mcs*Msun/Mearth, 'k:', label='dust total')
    plt.plot(e.rs, e.f_atms['Si'][0]*Msun/Mearth, '--', c='b', label='Si')
    plt.plot(e.rs, e.f_atms['Si'][1]*Msun/Mearth, ':', c='b')
    for atom,c in zip(atoms,colors):
        plt.plot(e.rs, e.f_atms[atom][0]*Msun/Mearth, '--', c=c, label=atom)
        plt.plot(e.rs, e.f_atms[atom][1]*Msun/Mearth, ':', c=c)


    plt.xlabel('Radius [au]')
    plt.ylabel(r'mass [$M_{\oplus}$]')

    plt.ylim(1e-3,1e3)
    plt.yscale('log')
    plt.legend()
    plt.show()
    pass

def main():
    '''
    Mdot_gas = 1e-8
    Mdot_dust = 1e-9
    Stokes = lambda R: 0.01

    T = lambda R: 150*(R/Rau)**-0.5
    alpha = lambda R: 1e-3

    species, abundances = get_molecular_properties()
    print([spec.name for spec in species])
    grid = Grid(0.1*Rau, 300*Rau, 512)

    p_env = PlanetEnv(grid, alpha(grid.Rc), 2.35, 1.0)

    dt = 5000
    Nt = 100
    f_plansi = 1e-1

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
        f_comp['H2'] = np.zeros(2)
        f_comp['Si'] = np.zeros(2)
        alpha = lambda R: 1e-3
        DM = DiscModel(grid, Mdot_gas, alpha, T)
        DM.compute_dust_surface_density(Mdot_dust, Stokes)
        species, abundances = get_molecular_properties()
        DM.compute_chemistry(species, abundances )
        p_env = PlanetEnv(grid, 1e-3, 2.35, 1.0)
        planet = Planet(mm*init_m*Mearth/Msun, mm*init_m*(1-frac_gc)*Mearth/Msun, mm*init_m*(frac_gc)*Mearth/Msun, f_comp,r*Rau)
        m,mc,mg,mco_g,mco_d,rr = dumb.std_evo(planet, DM, p_env, T(grid.Rc),f_plansi, dt, Nt, comp='CO')
        ms2=np.vstack((ms2,m))
        mcs2=np.vstack((mcs2,mc))
        mgs2=np.vstack((mgs2,mg))
        mco2_gs=np.vstack((mco2_gs,mco_g))
        mco2_ds=np.vstack((mco2_ds,mco_d))
        rrs2=np.vstack((rrs2,rr))
    '''

    #print(mco2_gs[4])
    #multi_plot(ms2[1:],mcs2[1:],mgs2[1:],mco2_gs[1:],mco2_ds[1:],rrs2[1:],Nt,dt,(8,16,32,64))
    #plot_planet_comp(ms2[4],mcs2[4],mgs2[4],mco2_gs[4],mco2_ds[4],rrs2[4],Nt,dt,'')

    Mdot_gas = 1e-8
    Mdot_dust = 1e-9
    Stokes = lambda R: 0.01

    T = lambda R: 150*(R/Rau)**-0.5
    alpha = lambda R: 1e-3
    grid = Grid(0.1*Rau, 300*Rau, 512)

    DM = DiscModel(grid, Mdot_gas, alpha, T)
    DM.compute_dust_surface_density(Mdot_dust, Stokes)
    species, abundances = get_molecular_properties()
    DM.compute_chemistry(species, abundances )

    species, abundances = get_molecular_properties()

    p_env = PlanetEnv(grid, alpha(grid.Rc), 2.35, 1.0)

    dt = 5000
    Nt = 100
    f_plansi = 1e-1

    frac_gc = 0.01
    init_m  = 5.0
    f_comp = dict(zip([spec.name for spec in species],np.zeros((len(species),2))))
    f_comp['H2'] = np.zeros(2)
    f_comp['Si'] = np.zeros(2)

    planet_ini = Planet(init_m*Mearth/Msun, init_m*(1-frac_gc)*Mearth/Msun, init_m*(frac_gc)*Mearth/Msun, f_comp, 15*Rau)

    planet_evo = dumb.std_evo_comp(planet_ini, DM, p_env, T(grid.Rc),f_plansi, dt, Nt)
    #plot_planet(planet_evo,Nt,dt)
    print(atom_mass(planet_evo[0].f_comp), '\n', atom_mass(planet_evo[-1].f_comp),'\n', planet_evo[-1].f_comp)
    plot_planet(planet_evo,Nt,dt)
    plot_atoms(planet_evo,Nt,dt)
    plot_CtoO(planet_evo,Nt,dt,'C','O')
    plot_CtoO(planet_evo,Nt,dt,'O','H', solar = 0.0005242)

    pass

if '__main__'==__name__:
    main()
