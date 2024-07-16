import argparse
import time

import espressomd
import espressomd.visualization
import espressomd.accumulators
import espressomd.electrostatics
import espressomd.interactions
import espressomd.lb
import espressomd.observables
import espressomd.polymer
import numpy as np
import pint
import scipy.constants as constants

parser = argparse.ArgumentParser()
parser.add_argument("N", type=int, help="Number of monomers")
parser.add_argument("--num_steps", type=int, default=int(1e7))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--efield", type=float, default=0.0)
args = parser.parse_args()

seed = args.seed

# Parameters from paper / Thesis
kT = 1.0

fene_stiffness = 30
fene_maximum_extension = 1.5

wca_epsilon = 0.25
wca_sigma = 1.0

monomer_type = 1
counterion_type = 2

polymer_bond_length = 0.91
monomer_charge = -1.0
counterion_charge = -monomer_charge
bjerrum_length = 2.84  # Corresponds to 7.1A for water at room temperature

# Taken from Thesis
fluid_viscosity = 3.0
fluid_density = 0.864
fluid_spacing_grid = 1.0
fluid_tau = 0.01
fluid_coupling_parameter = 20.0

external_field = args.efield  # 0.001

time_step = 0.01
warmup_steps_dna = int(1e3)  # 1e3
warmup_steps_ions = int(1e5)  # 5e4
warmup_steps_fluid = int(1e6)  # 1e5
production_steps = args.num_steps

# Box length -> adapted so that monomer concentration is 5mM

ureg = pint.UnitRegistry()

sigma_0 = (
    2.5 * ureg.angstrom
)  # Conversion factor between simulation and real length units
number_monomers_in_mol = args.N / constants.Avogadro * ureg.mol  # Unit: mol
monomer_concentration = 5 * ureg.millimolar  # Unit: mol/cubic meter
box_length = np.cbrt(number_monomers_in_mol / monomer_concentration)
box_length = int(round(box_length.to("angstrom") / sigma_0))  # Convert to sim units

print(
    f"Number of monomers and counterions: {args.N}\n"
    # + f"Number of salt molecules: {number_of_salt_molecules}\n"
    + f"Box length: {box_length}"
)

#
# Espresso system setup
#
system = espressomd.System(box_l=3 * [box_length])
system.time_step = time_step
system.cell_system.skin = 0.4
visualizer = espressomd.visualization.openGLLive(system)


# WCA interaction
for type1 in [monomer_type, counterion_type]:  # , salt_anion_type, salt_cation_type]:
    for type2 in [
        monomer_type,
        counterion_type,
    ]:  # , salt_anion_type, salt_cation_type]:
        system.non_bonded_inter[type1, type2].wca.set_params(
            epsilon=wca_epsilon, sigma=wca_sigma
        )

# Create polymer
# FENE potential
fene = espressomd.interactions.FeneBond(
    k=fene_stiffness, d_r_max=fene_maximum_extension
)
system.bonded_inter.add(fene)

polymer_positions = espressomd.polymer.linear_polymer_positions(
    n_polymers=1, beads_per_chain=args.N, bond_length=polymer_bond_length, seed=seed
)[0]

# For the observable
monomer_ids = []
print(f"Creating polymer with {args.N} monomers...")
monomers = system.part.add(
    pos=polymer_positions,
    type=np.full(shape=(args.N), fill_value=monomer_type),
    q=np.full(shape=(args.N), fill_value=monomer_charge),
)
previous_part = None
for part in monomers:
    monomer_ids.append(part.id)
    if previous_part is not None:
        part.add_bond((fene, previous_part))
    previous_part = part

# Remove overlap
system.integrator.set_steepest_descent(
    f_max=0, gamma=10, max_displacement=wca_sigma * 0.01
)

# fp = open("trajectory_dna.vtf", mode="w+t")
# vtf.writevsf(system, fp)

print("Running steepest descent...")
equil_count = 0
# system.integrator.run(1000)
while system.analysis.min_dist() < 0.8 * wca_sigma:
    system.integrator.run(1)
    equil_count += 1
print("AFTER MINIMIZATION", system.analysis.energy()["total"])
# vtf.writevcf(system, fp, types="all")

system.integrator.set_vv()
system.thermostat.set_langevin(kT=kT, gamma=1.0, seed=seed)

print("Equilibrating...")
for _ in range(warmup_steps_dna):
    system.integrator.run(1)

# Add counterions
print(f"Adding {args.N} counterions...")
counterion_ids = []
for _ in range(args.N):
    _part = system.part.add(
        pos=np.random.random(3) * box_length,
        type=counterion_type,
        q=counterion_charge,
    )
    counterion_ids.append(_part.id)

# Remove overlap
system.thermostat.turn_off()
system.integrator.set_steepest_descent(
    f_max=0, gamma=10, max_displacement=wca_sigma * 0.01
)

print("Running steepest descent...")
equil_count = 0
while system.analysis.min_dist() < 0.8 * wca_sigma:
    system.integrator.run(1)
    equil_count += 1

print("F_max polymers and counterions is " + str(np.abs(system.part.all().f).max()))


system.integrator.set_vv()
system.thermostat.set_langevin(kT=kT, gamma=1.0, seed=seed)

# Add P3M
print("Adding P3M...")
p3m = espressomd.electrostatics.P3M(prefactor=bjerrum_length, accuracy=1e-4)
# system.electrostatics.solver = p3m
system.actors.add(p3m)

print("Equilibrating...")
system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=seed)
for _ in range(warmup_steps_ions):
    system.integrator.run(1)
system.thermostat.turn_off()

# Switch to LB
print("Adding LB fluid and thermostat...")

lbf = espressomd.lb.LBFluidGPU(
    agrid=fluid_spacing_grid,
    dens=fluid_density,
    visc=fluid_viscosity,
    tau=fluid_tau,
    kT=kT,
    seed=seed,
)
system.actors.add(lbf)

# Turn off Langevin
system.thermostat.turn_off()
# Remove existing particle motion
system.galilei.kill_particle_motion()
# Remove existing center of mass momentum
system.galilei.galilei_transform()

system.thermostat.set_lb(LB_fluid=lbf, seed=seed, gamma=fluid_coupling_parameter)

print("Equilibrating...")
print(system.part.all().pos.tolist())
visualizer.run(1)
for _ in range(warmup_steps_fluid):
    system.integrator.run(1)

if not args.efield == 0.0:
    print("Adding external field...")
    for part in system.part.all():
        part.ext_force = [0, 0, part.q * external_field]

# Observables
monomer_velocity_obs = espressomd.observables.FluxDensityProfile(
    ids=monomer_ids,
    n_x_bins=1,
    n_y_bins=1,
    n_z_bins=1,
    min_x=0,
    min_y=0,
    min_z=0,
    max_x=box_length,
    max_y=box_length,
    max_z=box_length,
)
counterion_velocity_obs = espressomd.observables.FluxDensityProfile(
    ids=counterion_ids,
    n_x_bins=1,
    n_y_bins=1,
    n_z_bins=1,
    min_x=0,
    min_y=0,
    min_z=0,
    max_x=box_length,
    max_y=box_length,
    max_z=box_length,
)
chain_velocity_obs = espressomd.observables.ComVelocity(ids=monomer_ids)

chain_velocity_mean = espressomd.accumulators.MeanVarianceCalculator(
    obs=monomer_velocity_obs, delta_N=1
)

correlator1 = espressomd.accumulators.Correlator(
    obs1=monomer_velocity_obs,
    obs2=chain_velocity_obs,
    corr_operation="scalar_product",
    delta_N=1,
    tau_max=production_steps * system.time_step,  # 1e5 - 1e6
    tau_lin=16,
    compress1="discard2",
)
correlator2 = espressomd.accumulators.Correlator(
    obs1=counterion_velocity_obs,
    obs2=chain_velocity_obs,
    corr_operation="scalar_product",
    delta_N=1,
    tau_max=production_steps * system.time_step,  # 1e5 - 1e6
    tau_lin=16,
    compress1="discard2",
)

system.auto_update_accumulators.add(chain_velocity_mean)
system.auto_update_accumulators.add(correlator1)
system.auto_update_accumulators.add(correlator2)

print("Production run...")
_time = time.time()
for i in range(production_steps):
    system.integrator.run(1)

correlator1.finalize()
correlator2.finalize()

corr_dict = {
    "results_monomer": correlator1.result(),
    "lag_times_monomer": correlator1.lag_times(),
    "sample_sizes_monomer": correlator1.sample_sizes(),
    "results_counterion": correlator2.result(),
    "lag_times_counterion": correlator2.lag_times(),
    "sample_sizes_counterion": correlator2.sample_sizes(),
}

np.savez(
    f"simulation_{args.N}.npz",
    correlator=corr_dict,
    chain_velocity_mean=chain_velocity_mean.mean(),
    num_monomers=args.N,
    simulation_steps=time_step,
    production_steps=production_steps,
)

print(f"Finished after {time.time() - _time}s.")
