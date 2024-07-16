import argparse
import espressomd
import espressomd.lb
import espressomd.version
import espressomd.electrostatics
import espressomd.observables
import espressomd.accumulators
import espressomd.polymer
import espressomd.visualization
import numpy as np
import logging
import pickle
import time
import tqdm
import matplotlib.pyplot as plt
import espressomd.io.writer.vtf
from itertools import repeat

logging.basicConfig(level=logging.INFO)

required_features = ["LENNARD_JONES"]
espressomd.assert_features(required_features)

parser = argparse.ArgumentParser(epilog=__doc__)
parser.add_argument('-n', '--num_monomer', type=int, required=True)
parser.add_argument('-steps', '--sim_steps', type=int, required=True)
parser.add_argument('-gamma', '--gamma_lb', type=float, required=False, default=20.)
parser.add_argument('-write_to_vtf', type=bool, required=False, default=False)
parser.add_argument('-p3m_eq_steps', type=int, required=False, default=1000000)
parser.add_argument('-e_field', type=bool, required=False, default=False)
args = parser.parse_args()

N_MONOMER = args.num_monomer
TOTAL_TIME_STEPS = args.sim_steps
EQULIBRATION_STEPS = int(1e6)
P3M_EQ_STEPS = args.p3m_eq_steps
WRITE_TO_VTF = args.write_to_vtf
GAMMA = args.gamma_lb
E_FIELD = 0.05 if args.e_field else 0.
logging.info(f"Setting external E-field of {E_FIELD}.")
DENSITY = 5e-5
SIGMA = 1. # 2.5 Angstrom
BOX_L = np.power((N_MONOMER / DENSITY), 1.0 / 3.0) * np.ones(3)
BOX_L = np.round(BOX_L)
L_BJERRUM = 2.84 # 7.1 Angstrom
# L_BJERRUM = 2.0 # 7.1 Angstrom
DELTA_T = 0.01
SEED = 42

def time_it(func):
    def wrapper(*args, **kwargs):
        start_t = time.perf_counter()
        func(*args, **kwargs)
        print(f"Function {func.__name__} took {time.perf_counter() - start_t:.2f}s to execute.")
    return wrapper

def build_system():
    system = espressomd.System(box_l=BOX_L)
    system.periodicity = [True, True, True]
    system.cell_system.skin = 0.4
    system.time_step = DELTA_T
    logging.info(f"Created Simulation with {N_MONOMER} monomers and box length L={BOX_L}.")
    system.non_bonded_inter[0, 0].wca.set_params(epsilon=0.25, sigma=1.0)
    system.non_bonded_inter[0, 1].wca.set_params(epsilon=0.25, sigma=1.0)
    return system

def build_polymer(system, n_monomers, fene, seed):
    polymer_pos = espressomd.polymer.linear_polymer_positions(n_polymers=1, beads_per_chain=n_monomers, seed=seed, bond_length=0.91, min_distance=0.9, start_positions=np.expand_dims(BOX_L/2., 0))
    p_previous = None
    for pos in polymer_pos[0]:
        p = system.part.add(pos=pos, type=0, q=-1)
        if p_previous is not None:
            p.add_bond((fene, p_previous))
        p_previous = p

def equilibrate_polymer(system,seed):
    logging.info("Removing overlaps of polymers...")
    FMAX = 0.001 * 0.25 * 1. / system.time_step**2
    print("BEFORE MINIMIZATION", system.analysis.energy()["total"])
    system.integrator.set_steepest_descent(
        f_max=0,
        gamma=10.,
        max_displacement=0.01)
    # FIX
    logging.info("Desired F_max polymers and counterions=" +str(FMAX))
    FMAX = 10
    system.integrator.run(1000)
    print("AFTER MINIMIZATION", system.analysis.energy()["total"])
    logging.info("F_max polymers is " + str(np.abs(system.part.all().f).max()))
    system.integrator.set_vv()
    # assert np.abs(system.part.all().f).max() < FMAX, "Overlap removal did not converge (Polymer)!"
    logging.info("Remove overlap polymer finished.")
    logging.info("Equilibration polymer ...")
    system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=seed)
    system.integrator.run(20000)
    system.thermostat.turn_off()
    # system.galilei.kill_particle_motion()
    # system.galilei.galilei_transform()
    logging.info("F_max polymers"+str(np.abs(system.part.all().f).max()))
    logging.info("Equilibration polymer finished.")

def generate_counterions(system, n_counter_ions, seed):
    logging.info("Creating Counterions ...")
    # Spawn counterions closer around polymer
    com_polymer = system.part.select(type=0).pos.mean(0)
    counter_ions_pos = ((np.random.rand(n_counter_ions, 3) * 2) - 1) * BOX_L/ 2 + com_polymer
    # counter_ions_pos = np.random.rand(n_counter_ions, 3) * BOX_L
    for pos in counter_ions_pos:
        system.part.add(pos=pos, q=1., type=1)
    FMAX = 0.001 * 0.25 * 1. / system.time_step**2
    logging.info("Removing overlap Counterions ...")
    print("BEFORE MINIMIZATION", system.analysis.energy()["total"])
    system.integrator.set_steepest_descent(
        f_max=0,
        gamma=30.,
        max_displacement=0.01)
    system.integrator.run(1000)
    print("AFTER MINIMIZATION", system.analysis.energy()["total"])
    logging.info("Desired F_max polymers and counterions="+str(FMAX))
    logging.info("F_max polymers and counterions is " + str(np.abs(system.part.all().f).max()))
    FMAX = 10
    logging.info("Equilibration with counterions")
    # assert np.abs(system.part.all().f).max() < FMAX, "Overlap removal did not converge! (polymer and counterions)"
    logging.info("Remove overlap finished.")
    logging.info("Equilibration ...")
    system.integrator.set_vv()
    system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=seed)
    system.integrator.run(20000)
    system.thermostat.turn_off()

def turn_on_e_field(system):
    system.part.select(q=-1).ext_force = [0, 0, -E_FIELD]
    system.part.select(q=1).ext_force = [0, 0, E_FIELD]
    logging.info(f"Turned on E-field of {E_FIELD}.")

#prefactor =l_be=jerrum*kT/e^2
def turn_on_electrostatics(system, l_bjerrum, seed):
    if espressomd.version.minor() == 2:
        logging.info("P3m with GPU acceleration turned on.")
        p3m = espressomd.electrostatics.P3MGPU(prefactor=l_bjerrum, accuracy=0.0001)
        system.actors.add(p3m)
    else:
        logging.info("P3m with NO GPU acceleration turned on.")
        p3m = espressomd.electrostatics.P3M(prefactor=l_bjerrum, accuracy=0.0001)
        system.electrostatics.solver = p3m
    logging.info("Equilibration ...")
    system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=seed)
    all_ion_nbhoods = []
    for _ in tqdm.tqdm(range(P3M_EQ_STEPS), desc="Equilibrating electrostatics..."):
        system.integrator.run(1)
        # _ions_pos = ions_pos.calculate().tolist()
        # IF DEVELOPER VERSION: SET POSITIONS IN NBHOOD WITH KEYWORD ARGUMENT "pos"
        # if espressomd.version.minor() == 2:
        #    all_ion_nbhoods.append([system.analysis.nbhood(_ion_pos, r_catch=2.0) for _ion_pos in _ions_pos])
        # else:
        #     all_ion_nbhoods.append([system.analysis.nbhood(pos=_ion_pos, r_catch=2.0) for _ion_pos in _ions_pos])
    system.thermostat.turn_off()
    return all_ion_nbhoods

def set_hydrodynamic_interaction_walberla(system, seed, gamma):
    logging.info("Set hydrodynamic interactions Walberla.")
    system.galilei.kill_particle_motion()
    system.galilei.galilei_transform()
    lb=espressomd.lb.LBFluidWalberlaGPU(
        agrid=1.0, density=0.864, kinematic_viscosity=3.0, tau=system.time_step, kT=1, ext_force_density=[0.0,0.0,0.0], seed=seed)
    system.thermostat.turn_off()
    system.lb = lb
    system.thermostat.set_lb(LB_fluid=lb, seed=seed, gamma=gamma)

def set_hydrodynamic_interaction_gpu(system, seed, gamma):
    logging.info("Set hydrodynamic interactions GPU.")
    system.galilei.kill_particle_motion()
    system.galilei.galilei_transform()
    lbf = espressomd.lb.LBFluidGPU(agrid=1.0, dens=0.864, visc=3.0, tau=system.time_step, seed=seed, kT=1.0)
    system.actors.add(lbf)
    # system.lb = lbf
    system.thermostat.turn_off()
    system.thermostat.set_lb(LB_fluid=lbf, seed=seed, gamma=gamma)

@time_it
def equilibration_run(system, eq_steps, all_ion_nbhoods):
    logging.info(f"Equilibrate with hydrodynamic interactions for {eq_steps} steps.")
    for i in tqdm.tqdm(range(eq_steps), desc="Running LB..."):
        system.integrator.run(1)
        # _ions_pos = ions_pos.calculate().tolist()
        # if espressomd.version.minor() == 2:
        #    all_ion_nbhoods.append([system.analysis.nbhood(_ion_pos, r_catch=2.0) for _ion_pos in _ions_pos])
        # else:
        #     all_ion_nbhoods.append([system.analysis.nbhood(pos=_ion_pos, r_catch=2.0) for _ion_pos in _ions_pos])
    return all_ion_nbhoods

@time_it
def simulate_with_HI(system, total_steps, gamma, all_ion_nbhoods, write_to_vtf=False):
    ions_pos = espressomd.observables.ParticlePositions(ids=system.part.select(type=1).id)
    system.integrator.set_vv()
    logging.info(f"Simulate with hydrodynamic interactions for {total_steps} steps.")
    for i in tqdm.tqdm(range(total_steps), desc="Running LB"):
        system.integrator.run(1)
        # _ions_pos = ions_pos.calculate().tolist()
        # if espressomd.version.minor() == 2:
        #    all_ion_nbhoods.append([system.analysis.nbhood(_ion_pos, r_catch=2.0) for _ion_pos in _ions_pos])
        # else:
        #     all_ion_nbhoods.append([system.analysis.nbhood(pos=_ion_pos, r_catch=2.0) for _ion_pos in _ions_pos])
    logging.info("Saving ion neihborhoods.")
    # with open(f"../results/n={N_MONOMER}/ion_nbhoods_{total_steps}_{P3M_EQ_STEPS}_{GAMMA}.pkl", "wb") as f:
    #     pickle.dump(all_ion_nbhoods, f)
    logging.info("Saving trajectories.")
    if write_to_vtf is True:
        if E_FIELD is None:
            path = f'../data/trajectory_{gamma=}_{N_MONOMER}_{total_steps=}.vtf'
        else:
            path = f'../data/trajectory_{gamma=}_{N_MONOMER}_{total_steps=}_E.vtf'
        with open(path, mode='w+t') as f:
            espressomd.io.writer.vtf.writevsf(system, f)
            espressomd.io.writer.vtf.writevcf(system, f)    
            for i in tqdm.tqdm(range(300000), desc="Sampling"):
                system.integrator.run(1)
                espressomd.io.writer.vtf.writevcf(system, f)
        logging.info("Saving finished.")

def save_observables(correlator1, correlator2, correlator3, polymer_velocity_mean, Q_eff):
    correlator1.finalize()
    correlator2.finalize()
    correlator3.finalize()

    corr_dict = {
        "results_monomer": correlator1.result(),
        "lag_times_monomer": correlator1.lag_times(),
        "sample_sizes_monomer": correlator1.sample_sizes(),
        "results_counterions": correlator3.result(),
        "lag_times_counterions": correlator3.lag_times(),
        "sample_sizes_counterions": correlator3.sample_sizes(),
        "results_all": correlator2.result(),
        "lag_times_all": correlator2.lag_times(),
        "sample_sizes_all": correlator2.sample_sizes(),
        "Q_eff": Q_eff
    }

    if E_FIELD is None:
        path = f"../results/n={N_MONOMER}/simulation_{TOTAL_TIME_STEPS}_{P3M_EQ_STEPS}_{GAMMA}.npz"
    else:
        path = f"../results/n={N_MONOMER}/simulation_{TOTAL_TIME_STEPS}_{P3M_EQ_STEPS}_{GAMMA}_E.npz"
    np.savez(
        path,
        correlator=corr_dict,
        chain_velocity_mean=polymer_velocity_mean.mean(),
        num_monomers=N_MONOMER,
    )

def calc_effective_charge():
    counter = 0
    monomers = system.part.select(q=-1,type=0).id
    qions = system.part.select(q=1, type=1).id
    for mon_id in monomers:
        for qion_id in qions:
            dist = system.distance(system.part.by_id(mon_id),system.part.by_id(qion_id))
            if dist <= 2.0:
                counter += 1
                qions.remove(qion_id)
                break
    Q_eff = float((len(monomers)-counter)/len(monomers))
    return Q_eff


if __name__ == "__main__":
    system = build_system()
    visualizer = espressomd.visualization.openGLLive(system)
    fene = espressomd.interactions.FeneBond(k=30, d_r_max=1.5)
    system.bonded_inter.add(fene)
    build_polymer(system, N_MONOMER, fene, SEED)
    equilibrate_polymer(system, SEED)
    polymer_pos = espressomd.observables.ParticlePositions(ids=system.part.select(type=0).id)
    accumulator1 = espressomd.accumulators.TimeSeries(obs=polymer_pos, delta_N=2)
    system.auto_update_accumulators.add(accumulator1)
    generate_counterions(system, N_MONOMER,SEED)
    ions_pos = espressomd.observables.ParticlePositions(ids=system.part.select(type=1).id)
    accumulator2 = espressomd.accumulators.TimeSeries(obs=ions_pos, delta_N=2)
    system.auto_update_accumulators.add(accumulator2)
    # system.part.all().pos = [[17.921180416018036, 21.806248942187118, 39.73593109989852], [17.371544062902615, 21.329033292490458, 39.20605448362896], [17.299584357589808, 20.63199747541704, 38.57818287426307], [16.977726979210313, 20.045208133381276, 39.331722957891856], [16.307496812524132, 19.58340721763357, 39.01722562826629], [15.948879038363971, 19.202444744687664, 38.277133825891674], [15.77762550511775, 19.07774412344922, 37.45547718713095], [15.750154762776111, 18.978012101248385, 36.54162748154526], [14.948825222057863, 18.63604199872946, 36.30919549012562], [14.656332282401141, 18.507611778640843, 35.45369827377424], [14.749309175373115, 18.369251998901554, 34.589542028235016], [14.058915265059953, 17.834966780956126, 34.284201513069014], [14.21403826553445, 17.07904248450343, 33.96916877646006], [14.232953483211237, 16.200053548868937, 33.627111011021455], [14.577878299325752, 15.553085943646023, 33.07233371357073], [4.465469076728528, 38.49577622637732, -24.57622592117454], [22.832469038286057, 38.83601025894902, 10.22075636192888], [15.908667066432356, -49.89238163190327, -32.22076198881775], [21.606752824153713, 82.1105549536146, 34.64680311294331], [83.60351250026144, 18.59611110514094, -33.982671028123384], [16.775494128706015, 89.7606148087691, 39.943975494624084], [15.990239792304207, 88.53509372545584, 38.657327694496416], [-27.20460828135039, 32.93058717694066, 31.190894005700244], [16.89966785190946, 20.654780204125128, 36.55137241295285], [62.062383544108286, 26.50595592995062, 28.435105852001964], [17.19420799897272, 87.02984926475872, 39.548392160037864], [15.923566022273011, 30.83727529915816, 25.51286186598832], [85.91414099723391, 11.961964757903887, 41.86171183874221], [11.11998576128019, 16.73856542754977, 33.116728954989696], [29.991070496692195, 69.45806177282404, 106.80747270926508]]
    # rdf = espressomd.observables.RDF(ids1=system.part.select(type=0).id, ids2=system.part.select(type=1).id, n_r_bins=100, min_r=0, max_r=BOX_L[0]/2)
    # plt.plot(rdf.bin_centers(), rdf.calculate())
    # plt.show()
    turn_on_e_field(system)
    all_ion_nbhoods = turn_on_electrostatics(system, l_bjerrum=L_BJERRUM, seed=SEED)
    if espressomd.version.minor() == 2:
        set_hydrodynamic_interaction_gpu(system, SEED, gamma=GAMMA)
    else:
        set_hydrodynamic_interaction_walberla(system, SEED, gamma=GAMMA)
    equilibration_run(system, eq_steps=EQULIBRATION_STEPS, all_ion_nbhoods=all_ion_nbhoods)

    # Defining some observables
    monomer_velocity_obs = espressomd.observables.FluxDensityProfile(
        ids=system.part.select(type=0).id,
        n_x_bins=1,
        n_y_bins=1,
        n_z_bins=1,
        min_x=0,
        min_y=0,
        min_z=0,
        max_x=BOX_L[0],
        max_y=BOX_L[1],
        max_z=BOX_L[2],
    )
    counterions_velocity_obs = espressomd.observables.FluxDensityProfile(
        ids=system.part.select(type=1).id,
        n_x_bins=1,
        n_y_bins=1,
        n_z_bins=1,
        min_x=0,
        min_y=0,
        min_z=0,
        max_x=BOX_L[0],
        max_y=BOX_L[1],
        max_z=BOX_L[2],
    )
    all_velocity_obs = espressomd.observables.FluxDensityProfile(
        ids=system.part.all().id,
        n_x_bins=1,
        n_y_bins=1,
        n_z_bins=1,
        min_x=0,
        min_y=0,
        min_z=0,
        max_x=BOX_L[0],
        max_y=BOX_L[1],
        max_z=BOX_L[2],
    )
    polymer_velocity_obs = espressomd.observables.ComVelocity(ids=system.part.select(type=0).id)
    polymer_velocity_mean = espressomd.accumulators.MeanVarianceCalculator(
        obs=monomer_velocity_obs, delta_N=1
    )
    correlator1 = espressomd.accumulators.Correlator(
        obs1=monomer_velocity_obs,
        obs2=polymer_velocity_obs,
        corr_operation="scalar_product",
        delta_N=1,
        tau_max=TOTAL_TIME_STEPS * system.time_step,  # 1e5 - 1e6
        tau_lin=16,
        compress1="discard2",
    )
    correlator2 = espressomd.accumulators.Correlator(
        obs1=all_velocity_obs,
        obs2=polymer_velocity_obs,
        corr_operation="scalar_product",
        delta_N=1,
        tau_max=TOTAL_TIME_STEPS * system.time_step,  # 1e5 - 1e6
        tau_lin=16,
        compress1="discard2",
    )
    correlator3 = espressomd.accumulators.Correlator(
        obs1=counterions_velocity_obs,
        obs2=polymer_velocity_obs,
        corr_operation="scalar_product",
        delta_N=1,
        tau_max=TOTAL_TIME_STEPS * system.time_step,  # 1e5 - 1e6
        tau_lin=16,
        compress1="discard2",
    )
    system.auto_update_accumulators.add(polymer_velocity_mean)
    system.auto_update_accumulators.add(correlator1)
    system.auto_update_accumulators.add(correlator2)
    system.auto_update_accumulators.add(correlator3)
    simulate_with_HI(system, TOTAL_TIME_STEPS, gamma=GAMMA, all_ion_nbhoods=all_ion_nbhoods, write_to_vtf=WRITE_TO_VTF)
    Q_eff = []
    for _ in tqdm.trange(100):
        Q_eff.append(calc_effective_charge())
        system.integrator.run(9000)
    save_observables(correlator1, correlator2, correlator3, polymer_velocity_mean, Q_eff)
