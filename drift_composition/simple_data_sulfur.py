from drift_composition.simple_data import get_species_info, set_env
from drift_composition.constants import Mearth, Rau, stefan_boltzmann, G_Msun, Msun, Lsun, yr
from drift_composition.grid import Grid
from drift_composition.disc import DiscModel
from drift_composition.molecule import get_molecular_properties
from drift_composition.simple_planet import Planet, PlanetEnv
from drift_composition.atoms import atoms_in_molecule, ELEMENT_MASS, load_protosolar_abundances, molecule_mass
from drift_composition.simple_reduction import Evolution, atom_mass, dust_to_gas, final_accretion, crit_mass
import drift_composition.simple_reduction as red
import drift_composition.simple_planet as simp

from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import numpy as np


def store_sulfur_data_range(planet_ini, DM, p_env, T, inp='test', f_plansis=np.logspace(-6,-1,10), radii = np.linspace(6.,9.,10), final_radius = 1e-3, si= True, folder='data2', final_time=1e7):

    dt_ini = 500
    #dts = np.linspace(500,1000,10)
    Nt = 5000
    header = "# #mini, mcini, mgini, rini, plans, rfin, mfin, mcfin, mgfin, mgH, mgO, mgC, mdH, mdO, mdC, m10, mg10, mc10, mgH10, mgO10, mgC10, mdH10, mdO10, mdC10, yr, mgS, mdS, mgS10, mdS10, mgO_noSi, mdO_noSi \n"
    f = open('{}/{}.txt'.format(folder,inp), 'w')
    f.write(header)
    p_ini = planet_ini
    i=0
    for fp in f_plansis:
        for rad in radii:
            i += 1
            print(i)
            fin_r =final_radius*(1+9*np.random.rand())
            p_ini.dist = rad*Rau
            planet_evo, nn = simp.std_evo_comp(p_ini, DM, p_env, T(p_env.grid.Rc),fp, dt_ini, Nt, final_radius=fin_r, final_time=final_time)
            planet_fin = planet_evo[-1]
            #print(planet_fin.dist/Rau , fin_r)
            if si:
                exc = ()
            else:
                exc = list(p_env.dust.keys())
            #print('Si',si)
            evo = Evolution(planet_evo, nn, exclude=exc)
            fin_mass, fin_mc, fin_mg, fin_comp, fin_atom = final_accretion(evo, crit_mass(evo, threshhold_mass_fraction=0.9))
            data = (str(planet_ini.mass), 
                    str(planet_ini.mc), 
                    str(planet_ini.mg), 
                    str(planet_ini.dist/Rau),
                    str(fp),
                    str(planet_fin.dist/Rau),
                    str(planet_fin.mass), 
                    str(planet_fin.mc), 
                    str(planet_fin.mg), 
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['H'][0]), 
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['O'][0]), 
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['C'][0]),
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['H'][1]), 
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['O'][1]), 
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['C'][1]),
                    str(fin_mass), 
                    str(fin_mc), 
                    str(fin_mg),
                    str(fin_atom['H'][0]),
                    str(fin_atom['O'][0]),
                    str(fin_atom['C'][0]),
                    str(fin_atom['H'][1]),
                    str(fin_atom['O'][1]),
                    str(fin_atom['C'][1]),
                    str(planet_fin.time),
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['S'][0]),
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['S'][1]),
                    str(fin_atom['S'][0]),
                    str(fin_atom['S'][1]),
                    str(atom_mass(planet_fin.f_comp,exclude=list(p_env.dust.keys()))['O'][0]),
                    str(atom_mass(planet_fin.f_comp,exclude=list(p_env.dust.keys()))['O'][1]),
                    )
            #print(data)
            f.write('  '.join(data))
            f.write('\n')
            #print(planet_ini.dist/Rau ,planet_fin.dist/Rau)
    f.close()
    pass


def solar_org_comp(atom_abund=load_protosolar_abundances()):

# Oxygen / Nitrogen / Noble gases
    mol_abund = {
        'H2O' : atom_abund['O'] / 3 - atom_abund['S'] * 2 * 0.1,
        'CO'  : atom_abund['O'] / 6,
        'CO2' : atom_abund['O'] / 12,
    
        'N2'  : atom_abund['N'] * 0.45,
        'NH3' : atom_abund['N'] * 0.1,

        #'He' : atom_abund['He'], 
        'Ar' : atom_abund['Ar'], 
        'Kr' : 1.8e-9,
        'Xe' : 1.7e-10,

        'H2S' : atom_abund['S'] * 0.45,
        'FeS' : atom_abund['S'] * 0.45,
        'SO2' : atom_abund['S'] * 0.1,
    }
    grain_abund = {
        'MgFeSiO4' : atom_abund['O'] / 12,
        'P'  : atom_abund['P'],
        #'S'  : atom_abund['S'],
        'Na' : atom_abund['Na'],
        'K'  : atom_abund['K'],
    }
    gas_abund = {
        'H2' : atom_abund['H']/2,
        'He'  : atom_abund['He'],
    }
    dust = {n: g / np.sum(list(grain_abund.values())) for n,g in grain_abund.items()}
    gas  = {n: g / np.sum(list(gas_abund.values())) for n,g in gas_abund.items()}

# Count up the total carbon/oxygen abundance
    Ctot = 0
    for mol in mol_abund:
        atoms = atoms_in_molecule(mol)
        if 'C' in atoms:
            Ctot += mol_abund[mol] * atoms['C']

    for mol in grain_abund:
        atoms = atoms_in_molecule(mol)
        if 'C' in atoms:
            Ctot += grain_abund[mol] * atoms['C']

    # Put the rest into ethane / refractory carbon:
    C_org = atom_abund['C'] - Ctot 
    mol_abund['CH4'] = C_org * 0.25 / 1 
    mol_abund['C4H10'] = C_org * 0.75 / 4
    mol_abund['CH3OH'] = C_org * 0.0

    return mol_abund, atom_abund, dust, gas


def default_sulfur(folder='data2',si=True):
    if si:
        inp = 'default_S45'
    else:
        inp = 'default_noSi_S45'
    abund, atom_ab, dust, gas = solar_org_comp(atom_abund=load_protosolar_abundances())
    print(gas, atom_ab['S']/atom_ab['O'])
    planet_ini, DM, p_env, T, f_plansis, radii = set_env(abund,
                                                         atom_ab, 
                                                         St_alp=1.,
                                                         Mdot_gas=1e-8,
                                                         Md_Mg=0.01, 
                                                         radii = np.linspace(5.5, 17.5, 15), 
                                                         f_plansis= np.logspace(-5,0, 10), 
                                                         gas=gas, 
                                                         dust=dust, 
                                                         init_m=5.0, 
                                                         T0=150.)
    store_sulfur_data_range(planet_ini, DM, p_env, T, inp = inp, f_plansis=f_plansis, radii = radii, si=si,folder=folder)
    pass

def data_sets(Mdots, Md_Mgs, St_alps, radiis, final_radius, T0, si, mdot_alp=(0.01,10), folder='data2', suffix='', final_time=1e7):

    abund, atom_ab, dust, gas = solar_org_comp(atom_abund=load_protosolar_abundances())
    species, abundances = get_species_info(abund, atom_ab)

    f = open('{}/chem{}.txt'.format(folder,suffix), 'w')
    printing = (abundances, abundances/atom_ab['H'])
    f.write('; '.join([str(spec.name) for spec in species]))
    f.write('\n')
    f.write('; '.join([str(a) for a in abundances]))
    f.write('\n')
    f.write('; '.join([str(a) for a in abundances/atom_ab['H']]))
    f.close()

    for (mdot, radii) in zip(Mdots, radiis):
        for ma in mdot_alp:
            if si:
                inp = 'mdot_{}_st{}_{}K{}'.format(mdot, ma, T0,suffix)
            print(inp)
            planet_ini, DM, p_env, T, f_plansis, radii = set_env(abund,
                                                         atom_ab,
                                                         St_alp=ma,
                                                         Mdot_gas=mdot,
                                                         Md_Mg=0.01,
                                                         radii = radii,
                                                         f_plansis= np.logspace(-5,0,50),
                                                         gas=gas,
                                                         dust=dust,
                                                         init_m=5.0,
                                                         T0=T0)
            store_sulfur_data_range(planet_ini, DM, p_env, T, inp = inp, f_plansis=f_plansis, radii = radii, final_radius=final_radius, si=True,folder=folder,final_time=final_time)
    for mdmg in Md_Mgs:
        for st_a in St_alps:
            if si:
                inp = 'dust2gas_{}_St2alp{}_{}K{}'.format(mdmg, st_a, T0, suffix)
            print(inp)
            planet_ini, DM, p_env, T, f_plansis, radii = set_env(abund,
                                                         atom_ab,
                                                         St_alp=st_a,
                                                         Mdot_gas=1e-8,
                                                         Md_Mg=mdmg,
                                                         radii = radiis[1],
                                                         f_plansis= np.logspace(-5,0,50),
                                                         gas=gas,
                                                         dust=dust,
                                                         init_m=5.0,
                                                         T0=T0)
            store_sulfur_data_range(planet_ini, DM, p_env, T, 
                                                        inp = inp, 
                                                        f_plansis=f_plansis, 
                                                        radii = radii, 
                                                        final_radius=final_radius, 
                                                        si=True, 
                                                        folder=folder,
                                                        final_time=final_time)
    pass


def main():
    T0s = (200,) #[125,136,150,165,182,200]
    m_as = (0.01, 10)
    si_sws = (True,)#True)
    Mdots = (1e-7, 1e-8, 1e-9)
    Md_Mgs = (1e-2, 5e-2, 1e-1)
    St_alps = (1e-2, 1., 10.)
    radiis = [np.linspace(5.5, 20.5, 40),
              np.linspace(5.5, 17.5, 40),
              np.linspace(5.5, 12.5, 40)
             ]
    final_radius = 0.5e-2

    for si_sw in si_sws:
        for T0 in T0s:
            data_sets(Mdots, Md_Mgs, St_alps, radiis, final_radius, T0, si=si_sw, folder='data_sulfur', suffix='_S55_short', final_time=1e6)

    pass

if '__main__'==__name__:
    main()
