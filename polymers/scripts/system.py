import argparse
import espressomd
import espressomd.observables
import espressomd.accumulators
import espressomd.polymer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

required_features = ["LENNARD_JONES"]
espressomd.assert_features(required_features)

parser = argparse.ArgumentParser(epilog=__doc__)
parser.add_argument('-n', '--num_monomer', type=int, required=True)
args = parser.parse_args()

N_MONOMER = args.num_monomer
DENSITY = 5e-5
SIGMA = 2.5 # Angstrom
BOX_L = np.power(N_MONOMER / DENSITY, 1.0 / 3.0) * np.ones(3)
L_BJERRUM = 2.84 # 7.1 Angstrom
DELTA_T = 0.01


def build_system():
    system = espressomd.System(box_l=BOX_L)
    system.periodicity = [True, True, True]
    system.cell_system.skin = 0.4
    system.time_step = DELTA_T
    logging.info(f"Created Simulation with {N_MONOMER} monomers and box length L={BOX_L}.")
    system.non_bonded_inter[0, 0].wca.set_params(epsilon=0.25, sigma=1.0)
    return system

def build_polymer(system, n_monomers, fene):
    polymer_pos = espressomd.polymer.linear_polymer_positions(n_polymers=1, beads_per_chain=n_monomers, seed=42, bond_length=0.91)
    p_previous = None
    for pos in polymer_pos[0]:
        p = system.part.add(pos=pos, type=0, q=-1)
        if p_previous is not None:
            p.add_bond((fene, p_previous))
        p_previous = p

def equilibrate_polymer(system):
    logging.info("Removing overlaps ...")
    FMAX = 0.001 * 0.25 * 1. / system.time_step**2
    system.integrator.set_steepest_descent(
        f_max=FMAX,
        gamma=10,
        max_displacement=0.01)
    # FIX
    FMAX = 10
    system.integrator.run(100)
    print(np.abs(system.part.all().f).max())
    system.integrator.set_vv()
    assert np.abs(system.part.all().f).max() < FMAX, "Overlap removal did not converge!"
    logging.info("Remove overlap finished.")
    logging.info("Equilibration ...")
    system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=42)
    system.integrator.run(2000)
    system.thermostat.turn_off()
    system.galilei.kill_particle_motion()
    system.galilei.galilei_transform()
    logging.info("Equilibration finished.")

def generate_counterions(system, n_counter_ions):
    logging.info("Creating Counterions ...")
    counter_ions_pos = np.random.rand(n_counter_ions, 3) * BOX_L
    for pos in counter_ions_pos:
        system.part.add(pos=pos, q=1., type=1)
    logging.info("Equilibration")
    system.integrator.run(1000)

if __name__ == "__main__":
    system = build_system()
    fene = espressomd.interactions.FeneBond(k=30, r_0=0.91, d_r_max=1.5)
    system.bonded_inter.add(fene)
    build_polymer(system, N_MONOMER, fene)
    equilibrate_polymer(system) # SOMESH FRAGEN WARUM NICHT KONVERGIERT
    vel = espressomd.observables.ParticleVelocities(ids=[i for i in range(len(system.part.all()))])
    accumulator = espressomd.accumulators.TimeSeries(obs=vel, delta_N=2)
    system.auto_update_accumulators.add(accumulator)
    generate_counterions(system, N_MONOMER)
    print(np.sum(accumulator.time_series()**2 / 2, axis=1).sum(1))