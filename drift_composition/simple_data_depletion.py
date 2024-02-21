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

def depletion(mol_abund, atom_abund, fraction, mol):
    print(mol)
    if fraction > 0:
        abund = deplete_CO_abundance(mol_abund, fraction, mol)
    else:
        abund = mol_abund
    spec, abund = get_species_info(mol_abund,atom_abund)
    return spec, abund

def create_disc(St_alp, Mdot_gas, Md_Mg, L_star, mu, T, grid, alp):  
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
    for (df, ds) in zip(depl_fracs,depl_specs):
        species, abundances = depletion(mol_abund, atom_abund, df, ds)

    DM.compute_chemistry(species, abundances )
    print([(spec.name) for spec in species], abundances)
    return DM, species, abundances
    

def set_env_dp(mol_abund, atom_abund, St_alp=1.,Mdot_gas=1e-8, Md_Mg=0.1, radii = np.linspace(7.,9.,10), f_plansis= np.logspace(-6,-1,10), gas={'H2':0.912,'He':0.087}, dust={'MgFeSiO4':3.235e-5}, init_m=5.0, mu=2.35, L_star=2., m_star=1.4, depl_frac=(), depl_spec=()):

    alp = 1e-3
    alpha = lambda R: alp
    grid = Grid(0.0005*Rau, 100*Rau, 512)
    T = create_temperature_profile(grid, L_star, Mdot_gas, alpha, mu=mu)
    DM = create_disc(St_alp, Mdot_gas, Md_Mg, L_star, mu, T, grid, alp)

    #Set up chemistry
    DM, species, abundances = create_chemistry(mol_abund, atom_abund, DM, depl_frac, depl_spec)


    f_comp = dict(zip([spec.name for spec in species],np.zeros((len(species),2))))
    for g in gas:
        f_comp[g] = np.zeros(2)
    for d in dust:
        f_comp[d] = np.zeros(2)
    p_env = PlanetEnv(grid, alpha(grid.Rc), mu, m_star, gas, dust)

    #Set up iniital planet
    frac_gc = 0.01
    planet_ini = Planet(init_m*Mearth/Msun, init_m*(1-frac_gc)*Mearth/Msun, init_m*(frac_gc)*Mearth/Msun, f_comp, 8.5*Rau)

    return planet_ini, DM, p_env, T, f_plansis, radii
    #Set up sample data

def data_sets(Mdots, Md_Mgs, St_alps, radiis, final_radius):
   
    abund, atom_ab, dust, gas = solar_org_comp(atom_abund=load_protosolar_abundances())
    for (mdot, radii) in zip(Mdots, radiis):
        inp = 'mdot_{}_hot2'.format(mdot)
        print(inp)
        planet_ini, DM, p_env, T, f_plansis, radii = set_env(abund,
                                                         atom_ab, 
                                                         St_alp=10.,
                                                         Mdot_gas=mdot,
                                                         Md_Mg=0.01, 
                                                         radii = radii, 
                                                         f_plansis= np.logspace(-6,-1,5), 
                                                         gas=gas, 
                                                         dust=dust, 
                                                         init_m=5.0, 
                                                         L_star=2.)
        store_data_range(planet_ini, DM, p_env, T, inp = inp, f_plansis=f_plansis, radii = radii, final_radius=final_radius)
    for mdmg in Md_Mgs:
        for st_a in St_alps:
            inp = 'dust2gas_{}_St2alp{}_hot2'.format(mdmg, st_a)
            print(inp)
            planet_ini, DM, p_env, T, f_plansis, radii = set_env(abund,
                                                         atom_ab, 
                                                         St_alp=st_a,
                                                         Mdot_gas=1e-8,
                                                         Md_Mg=mdmg, 
                                                         radii = radiis[1], 
                                                         f_plansis= np.logspace(-6,-1,5), 
                                                         gas=gas, 
                                                         dust=dust, 
                                                         init_m=5.0, 
                                                         L_star=2.)
            store_data_range(planet_ini, DM, p_env, T, inp = inp, f_plansis=f_plansis, radii = radii, final_radius=final_radius)
    pass

def depletion_sets(depl_fracs, depl_specs, radii, final_radius):
    abund, atom_ab, dust, gas = solar_org_comp(atom_abund=load_protosolar_abundances())
    for (df, ds) in zip(depl_fracs, depl_specs):
        inp = 'depl_{}_{}'.format(ds,df)
        print(inp)
        planet_ini, DM, p_env, T, f_plansis, radii = set_env_dp(abund,
                                                         atom_ab, 
                                                         St_alp=10.,
                                                         Mdot_gas=1e-8,
                                                         Md_Mg=0.01, 
                                                         radii = radii, 
                                                         f_plansis= np.logspace(-6,-1,5), 
                                                         gas=gas, 
                                                         dust=dust,
                                                         init_m=5.0, 
                                                         L_star=2.,
                                                         depl_frac=df,
                                                         depl_spec=ds
                                                         )
        store_data_range(planet_ini, DM, p_env, T, inp = inp, f_plansis=f_plansis, radii = radii, final_radius=final_radius)
    pass


def main():    
    #default_data()
    depl_specs= [['C2H6',], ['CO2',], ['C4H10']]
    depl_fracs= [[0.9] ,[0.9] ,[0.9]]
    radiis = np.linspace(7.5, 15.5, 20)
             
    final_radius = 1e-2

    depletion_sets(depl_fracs, depl_specs, radiis, final_radius)
    pass

if '__main__'==__name__:
    main()
