import argparse
import espressomd
import espressomd.lb

import espressomd.observables
import espressomd.accumulators
import espressomd.polymer
import numpy as np
import logging
import tqdm

logging.basicConfig(level=logging.INFO)

required_features = ["LENNARD_JONES"]
espressomd.assert_features(required_features)

parser = argparse.ArgumentParser(epilog=__doc__)
parser.add_argument('-n', '--num_monomer', type=int, required=True)
args = parser.parse_args()

N_MONOMER = args.num_monomer
DENSITY = 5e-5
SIGMA = 2.5 # Angstrom
BOX_L = np.power((N_MONOMER / DENSITY), 1.0 / 3.0) * np.ones(3)
BOX_L= np.round(BOX_L)
L_BJERRUM = 2.84 # 7.1 Angstrom
DELTA_T = 0.01
SEED = 42
print(BOX_L)
def build_system():
    system = espressomd.System(box_l=BOX_L)
    system.periodicity = [True, True, True]
    system.cell_system.skin = 0.4
    system.time_step = DELTA_T
    logging.info(f"Created Simulation with {N_MONOMER} monomers and box length L={BOX_L}.")
    system.non_bonded_inter[0, 0].wca.set_params(epsilon=0.25, sigma=1.0)
    system.non_bonded_inter[0, 1].wca.set_params(epsilon=0.25, sigma=1.0)

    return system

def build_polymer(system, n_monomers, fene,seed):
    polymer_pos = espressomd.polymer.linear_polymer_positions(n_polymers=1, beads_per_chain=n_monomers, seed=seed, bond_length=0.91)
    p_previous = None
    for pos in polymer_pos[0]:
        p = system.part.add(pos=pos, type=0, q=-1)
        if p_previous is not None:
            p.add_bond((fene, p_previous))
        p_previous = p

def equilibrate_polymer(system,seed):
    logging.info("Removing overlaps of polymers...")
    FMAX = 0.001 * 0.25 * 1. / system.time_step**2
    system.integrator.set_steepest_descent(
        f_max=FMAX,
        gamma=10,
        max_displacement=0.01)
    # FIX
    logging.info("desired F_max polymers and counterions=" +str(FMAX))
    FMAX = 10
    system.integrator.run(1000)
    logging.info("F_max polymers="+str(np.abs(system.part.all().f).max()))
    system.integrator.set_vv()
    assert np.abs(system.part.all().f).max() < FMAX, "Overlap removal did not converge (Polymer)!"
    logging.info("Remove overlap polymer finished.")
    logging.info("Equilibration polymer ...")
    system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=seed)
    system.integrator.run(2000)
    system.thermostat.turn_off()
    # system.galilei.kill_particle_motion()
    # system.galilei.galilei_transform()
    logging.info("F_max polymers 2="+str(np.abs(system.part.all().f).max()))
    logging.info("Equilibration polymer finished.")

def generate_counterions(system, n_counter_ions,seed):
    logging.info("Creating Counterions ...")
    counter_ions_pos = np.random.rand(n_counter_ions, 3) * BOX_L
    for pos in counter_ions_pos:
        system.part.add(pos=pos, q=1., type=1)
    FMAX = 0.001 * 0.25 * 1. / system.time_step**2
    logging.info("Removing overlap Counterions ...")
    system.integrator.set_steepest_descent(
        f_max=FMAX,
        gamma=10,
        max_displacement=0.01)
    system.integrator.run(1000)
    logging.info("desired F_max polymers and counterions="+str(FMAX))
    logging.info("F_max polymers and counterions="+str(np.abs(system.part.all().f).max()))
    # FIX
    FMAX = 10
    logging.info("Equilibration with counterions")
    assert np.abs(system.part.all().f).max() < FMAX, "Overlap removal did not converge! (polzmer and counterions)"
    logging.info("Remove overlap finished.")
    logging.info("Equilibration ...")
    system.integrator.set_vv()
    system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=seed)
    system.integrator.run(2000)
    system.thermostat.turn_off()

#prefactor =l_be=jerrum*kT/e^2
def turn_on_electrostatics(l_bjerrum):
    p3m = espressomd.electrostatics.P3M(prefactor=l_bjerrum, accuracy=0.0001)
    system.electrostatics.solver = p3m
    logging.info("P3m turned on.")

def set_hydrodynamic_interaction_walberla(seed):
    logging.info("Set hydrodynamic interactions walberla")
    system.galilei.kill_particle_motion()
    system.galilei.galilei_transform()
    lb=espressomd.lb.LBFluidWalberla(
        agrid=1.0, density=0.84, kinematic_viscosity=3.0, tau=system.time_step,kT=1)
    system.lb = lb
    system.thermostat.set_lb(LB_fluid=lb, seed=seed, gamma=20.0) 

def set_hydrodynamic_interaction_gpu(seed):
    logging.info("Set hydrodynamic interactions walberla")
    system.galilei.kill_particle_motion()
    system.galilei.galilei_transform()
    lbf = espressomd.lb.LBFluidGPU(agrid=1, dens=10, visc=.1, tau=0.01)
    system.thermostat.set_lb(LB_fluid=lbf, seed=seed, gamma=20.0) 
    system.actors.add(lbf)

def simulate_with_HI(total_steps,steps_per_iteration):
    logging.info(f"Simulate with hydrodynamic interactions for {total_steps} steps")
    num_iterations = int(total_steps/steps_per_iteration)
    for _ in tqdm.tqdm(range(num_iterations)):
        system.integrator.run(steps_per_iteration)
if __name__ == "__main__":
    system = build_system()
    fene = espressomd.interactions.FeneBond(k=30, r_0=0.91, d_r_max=1.5)
    system.bonded_inter.add(fene)
    build_polymer(system, N_MONOMER, fene,SEED)
    equilibrate_polymer(system,SEED) # SOMESH FRAGEN WARUM NICHT KONVERGIERT
    vel = espressomd.observables.ParticleVelocities(ids=[i for i in range(len(system.part.all()))])
    accumulator = espressomd.accumulators.TimeSeries(obs=vel, delta_N=2)
    system.auto_update_accumulators.add(accumulator)
    generate_counterions(system, N_MONOMER,SEED)
    turn_on_electrostatics(l_bjerrum=L_BJERRUM)
    set_hydrodynamic_interaction_walberla(SEED)
    simulate_with_HI(1000,100)
    #print(np.sum(accumulator.time_series()**2 / 2, axis=1).sum(1))