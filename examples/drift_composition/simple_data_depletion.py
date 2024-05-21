from drift_composition.constants import Mearth, Rau, stefan_boltzmann, G_Msun, Msun, Lsun, yr
from drift_composition.grid import Grid
from drift_composition.disc import DiscModel
from drift_composition.molecule import get_molecular_properties
from drift_composition.simple_planet import Planet, PlanetEnv
from drift_composition.atoms import atoms_in_molecule, ELEMENT_MASS, load_protosolar_abundances, molecule_mass
from drift_composition.simple_reduction import Evolution, atom_mass, dust_to_gas, final_accretion, crit_mass
from  drift_composition.simple_data import store_data_range, create_temperature_profile, solar_org_comp, get_species_info
import drift_composition.simple_reduction as red
import drift_composition.simple_planet as simp

from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import numpy as np


def solar_org_comp(atom_abund=load_protosolar_abundances()):

# Oxygen / Nitrogen / Noble gases
    mol_abund = {
        'H2O' : atom_abund['O'] / 3,
        'CO'  : atom_abund['O'] / 6,
        'CO2' : atom_abund['O'] / 12,
    
        'N2'  : atom_abund['N'] * 0.45,
        'NH3' : atom_abund['N'] * 0.1,

        #'He' : atom_abund['He'], 
        'Ar' : atom_abund['Ar'], 
        'Kr' : 1.8e-9,
        'Xe' : 1.7e-10,
    }
    grain_abund = {
        'MgFeSiO4' : atom_abund['O'] / 12,
        'P'  : atom_abund['P'],
        'S'  : atom_abund['S'],
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
    mol_abund['C2H6'] = C_org * 0.0

    return mol_abund, atom_abund, dust, gas

def deplete_CO_abundance(abund, fraction, species):
    """Convert a given fraction of CO into a different species. Any 
    excess/deficit of oxygen is assumed to change the water abundance"""
    if not (0 < fraction < 1):
        raise ValueError(f"Fraction of CO removed must be between 0 and 1, not {fraction}.")

    # Copy the abundances
    new = {}
    for mol in abund:
        new[mol] = abund[mol]
    if species not in new:
        new[species] = 0
    
    # Deplete CO
    xi = abund['CO'] * fraction
    new['CO'] -= xi

    # Update the abundance of the new species and H2O depending on the
    # number of carbon and oxygen atoms in the molecule.
    spec_abund =  atoms_in_molecule(species)

    new[species] += xi / spec_abund['C']
    new['H2O'] -= xi*(spec_abund['O']/spec_abund['C'] - 1)

    return new

def create_disc(St_alp, Mdot_gas, Md_Mg, mu, T, grid, alp):  
    #Set up disc dynamics
    alpha = lambda R: alp
    
    DM = DiscModel(grid, Mdot_gas, alpha, T, mu)

    Mdot_dust = Mdot_gas * Md_Mg 
    Stokes = lambda R: St_alp * alp
    DM.compute_dust_surface_density(Mdot_dust, Stokes)
    print('alpha = {}, Stokes = {}, Mdot_gas = {}, Mdot_dust ={}, T[1au] = {}'.format(alp, alp*St_alp, Mdot_gas, Mdot_dust, T(grid.Rc)[np.argmin(abs(grid.Rc-Rau))]))

    return DM

def create_chemistry(mol_abund, atom_abund, DM, depl_fracs, depl_specs):
    specs, abund = get_species_info(mol_abund, atom_abund)
    abundance = deplete_CO_abundance(mol_abund, depl_fracs, depl_specs) 
    print(depl_specs, abundance[depl_specs])

    DM.compute_chemistry(specs, list(abundance.values()))
    print('cheminstry done')
    return DM, specs, abundance
    

def set_env_dp(mol_abund, atom_abund, St_alp=1.,Mdot_gas=1e-8, Md_Mg=0.1, radii = np.linspace(7.,9.,10), f_plansis= np.logspace(-6,-1,10), gas={'H2':0.912,'He':0.087}, dust={'MgFeSiO4':3.235e-5}, init_m=5.0, mu=2.35, T0=150, m_star=1.4, depl_frac=(), depl_spec=()):

    alp = 1e-3
    alpha = lambda R: alp
    grid = Grid(0.0005*Rau, 100*Rau, 512)
    #T = create_temperature_profile(grid, L_star, Mdot_gas, alpha, mu=mu)
    T = lambda R: T0*(R/Rau)**(-0.5)
    DM = create_disc(St_alp, Mdot_gas, Md_Mg, mu, T, grid, alp)

    #Set up chemistry
    DM, species, abundances = create_chemistry(mol_abund, atom_abund, DM, depl_frac, depl_spec)


    f_comp = dict(zip([spec.name for spec in species],np.zeros((len(species),2))))
    for g in gas:
        f_comp[g] = np.zeros(2)
    for d in dust:
        f_comp[d] = np.zeros(2)
    p_env = PlanetEnv(grid, alpha(grid.Rc), mu, m_star, gas, dust)

    #Set up initial planet
    frac_gc = 0.01
    planet_ini = Planet(init_m*Mearth/Msun, init_m*(1-frac_gc)*Mearth/Msun, init_m*(frac_gc)*Mearth/Msun, f_comp, 8.5*Rau)

    return planet_ini, DM, p_env, T, f_plansis, radii
    #Set up sample data

def depletion_sets(depl_fracs, depl_specs, radii, final_radius, T0, si):
    abund, atom_ab, dust, gas = solar_org_comp(atom_abund=load_protosolar_abundances())
    for (df, ds) in zip(depl_fracs, depl_specs):
        if si:
            inp = 'depl_{}_{}_{}K_Si_drift'.format(ds,df,T0)
        else:
            inp = 'depl_{}_{}_{}K_noSi_drift'.format(ds,df,T0)
        print(inp)
        planet_ini, DM, p_env, T, f_plansis, radii = set_env_dp(abund,
                                                         atom_ab, 
                                                         St_alp=10.,
                                                         Mdot_gas=1e-8,
                                                         Md_Mg=0.01, 
                                                         radii = radii, 
                                                         f_plansis= np.logspace(-5,0,30), 
                                                         gas=gas, 
                                                         dust=dust,
                                                         init_m=5.0, 
                                                         T0=T0,
                                                         depl_frac=df,
                                                         depl_spec=ds
                                                         )
        store_data_range(planet_ini, DM, p_env, T, inp = inp, f_plansis=f_plansis, radii = radii, final_radius=final_radius, si=si)
    pass


def main():    
    #default_data()
    depl_specs= ['CH3OH', 'C2H6', 'CO2', 'C4H10']
    depl_fracs= [0.9 ,0.9 ,0.9, 0.9]
    radiis = np.linspace(6.5, 15.5, 50)
             
    final_radius = 1e-3
    T0 = 150

    for si in (True,False):
        depletion_sets(depl_fracs, depl_specs, radiis, final_radius, T0, si =si)
    pass

if '__main__'==__name__:
    main()
