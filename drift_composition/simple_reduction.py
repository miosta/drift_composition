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