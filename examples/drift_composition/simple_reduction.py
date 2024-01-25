from drift_composition.constants import Mearth, Msun, Rau, yr
from drift_composition.grid import Grid
from drift_composition.disc import DiscModel
from drift_composition.molecule import get_molecular_properties
from drift_composition.simple_planet import Planet, PlanetEnv
from drift_composition.atoms import atoms_in_molecule, ELEMENT_MASS
import drift_composition.simple_planet as dumb
import matplotlib.pyplot as plt
import numpy as np

class Evolution:
    def __init__(self, planet_evo, Nt, dt):
        self.planet_evo = planet_evo
        self.Nt = Nt
        self.dt = dt
        self.masses = np.array([p.mass for p in planet_evo])
        self.mgs    = np.array([p.mg for p in planet_evo])
        self.mcs    = np.array([p.mc for p in planet_evo])
        self.rs     = np.array([p.dist for p in planet_evo])/Rau
        self.time   = np.arange(Nt)*dt
    @property
    def f_comps(self):
        names = list(self.planet_evo[0].f_comp.keys())
        comps = dict(zip(names,np.zeros((len(names),2,len(range(self.Nt))))))
        for name in names:
            comps[name][0] = np.array([p.f_comp[name][0] for p in self.planet_evo])
            comps[name][1] = np.array([p.f_comp[name][1] for p in self.planet_evo])
        return comps
    @property
    def f_atms(self):
        at_evo = np.array([atom_mass(p.f_comp) for p in self.planet_evo])
        atoms = list(ELEMENT_MASS.keys())
        abun = dict(zip(atoms,np.zeros((len(atoms),2,len(range(self.Nt))))))
        for atom in atoms:
            abun[atom][0] = np.array([am[atom][0] for am in at_evo])
            abun[atom][1] = np.array([am[atom][1] for am in at_evo])
        return abun        

def atom_mass(f_comp):
    f_atom = dict(zip(list(ELEMENT_MASS.keys()),np.zeros((len(list(ELEMENT_MASS.keys())),2))))
    mol_names = list(f_comp.keys())
    for mol_n in mol_names:
        n_atoms = atoms_in_molecule(mol_n)
        mass_mol = np.sum([n_atoms[na]*ELEMENT_MASS[na] for na in list(n_atoms.keys())])
        for na in list(n_atoms.keys()):
            f_atom[na][0] += f_comp[mol_n][0]/mass_mol*ELEMENT_MASS[na] * n_atoms[na]
            f_atom[na][1] += f_comp[mol_n][1]/mass_mol*ELEMENT_MASS[na] * n_atoms[na]
    return f_atom

def dust_to_gas(evo, cut_index):
        core_mass = evo.mcs[cut_index]
        new_mcs = np.where(evo.mcs>core_mass,core_mass,evo.mcs)
        new_mgs = evo.mgs + evo.mcs - new_mcs
        names = list(evo.f_comps.keys())
        new_comps = evo.f_comps
        for name in names:
            solid = evo.f_comps[name][1][cut_index]
            new_comps[name][1][cut_index+1:] = solid
            new_comps[name][0] += evo.f_comps[name][1] - new_comps[name][1]
        new_atoms = evo.f_atms
        atoms = list(evo.f_atms.keys())
        for at in atoms:
            solid = evo.f_atms[at][1][cut_index]       
            new_atoms[at][1][cut_index+1:] = solid
            new_atoms[at][0] += evo.f_atms[at][1] - new_atoms[at][1]
        return new_mgs, new_mcs, new_comps, new_atoms

def run_away(evo):
    return np.argmax(evo.mgs>evo.mcs)

def final_comp(evo):
    return np.argmax(evo.time>evo.time[-1]*0.9)

def crit_mass(evo, threshhold_mass):
    return np.argmax(evo.masses>threshhold_mass)

def store_data_range(planet_ini, DM, p_env, T):
    f_plansis = np.logspace(-5,0,5)
    radii = np.linspace(7.,9.,5)
    dts = np.linspace(425,8325,10)
    Nt = 1000
    header = "#mini, mcini, mgini, rini, plans, mfin, mcfin, mgfin, rfin, mgH, mgO, mgC, mdH, mdO, mdC, yr \n"
    f = open('test.txt', 'w')
    f.write(header)

    for fp in f_plansis:
        for (rad, dt) in zip(radii,dts):
            planet_ini.dist = rad*Rau
            planet_evo, nn = dumb.std_evo_comp(planet_ini, DM, p_env, T(p_env.grid.Rc),fp, dt, Nt)
            planet_fin = planet_evo[-1]
            data = (str(planet_ini.mass), 
                    str(planet_ini.mc), 
                    str(planet_ini.mg), 
                    str(planet_ini.dist/Rau),
                    str(fp),
                    str(planet_fin.mass), 
                    str(planet_fin.mc), 
                    str(planet_fin.mg), 
                    str(planet_fin.dist/Rau),
                    str(atom_mass(planet_fin.f_comp)['H'][0]), 
                    str(atom_mass(planet_fin.f_comp)['O'][0]), 
                    str(atom_mass(planet_fin.f_comp)['C'][0]),
                    str(atom_mass(planet_fin.f_comp)['H'][1]), 
                    str(atom_mass(planet_fin.f_comp)['O'][1]), 
                    str(atom_mass(planet_fin.f_comp)['C'][1]),
                    str(nn*dt)
                    )
            f.write('  '.join(data))
            f.write('\n')
    f.close()
    pass
