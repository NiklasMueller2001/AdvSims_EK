import argparse
import espressomd

import espressomd.lb
import espressomd.electrostatics
import espressomd.observables
import espressomd.accumulators
import espressomd.polymer
import espressomd.visualization
import espressomd.shapes
import espressomd.io
import espressomd.checkpointing
import numpy as np
import logging
import threading
import time
import tqdm
from matplotlib import pyplot as plt
import os

logging.basicConfig(level=logging.INFO)

required_features = ["LENNARD_JONES"]
espressomd.assert_features(required_features)

parser = argparse.ArgumentParser(epilog=__doc__)
parser.add_argument('-n', '--num_monomer', type=int, required=False)
args = parser.parse_args()
COM_FLAG = "GPU"
COM_FLAG = "CPU"
VERSION_FLAG = "4.2"
# VERSION_FLAG = "WAL"

corners = [[0,0,0],[0,16,0],[16,0,0],[16,16,0],[0,0,16],[0,16,16],[16,0,16],[16,16,16],]
BOX_L = [16,16,23]
SLIT_WIDTH = 12
L_BJERRUM = 0.7095 # 7.095 Angstrom
DELTA_T = 0.01
SEED = 42
KT=1
EPSILON_0=(1.428e-3)
EPSILON_R=78.54
E_FIELD=25
RHO_A = 26.18
VISCOCITY = 0.25
GAMMA = 15

CONST1 = 1.25
CONST2 = -14.75
LBB1 = 1.5
LBB2 = -14.5
WALLP1 = 1
WALLP2 = 15

E_FORCE = np.array([E_FIELD,0,0])
np.random.seed(SEED)
# assert False
def time_it(func):
    def wrapper(*args, **kwargs):
        start_t = time.perf_counter()
        func(*args, **kwargs)
        print(f"Function {func.__name__} took {time.perf_counter() - start_t:.2f}s to execute.")
    return wrapper

def build_system(sigma):
    logging.info("Build System ...")

    system = espressomd.System(box_l=BOX_L)
    system.periodicity = [True, True, True]
    system.cell_system.skin = 0.4
    system.time_step = DELTA_T
    system.non_bonded_inter[0, 1].wca.set_params(epsilon=1., sigma=1)
    system.non_bonded_inter[0, 0].wca.set_params(epsilon=1, sigma=1.)
    system.non_bonded_inter[0, 2].wca.set_params(epsilon=1., sigma=1)


    return system
# def generate_box(system,corners):
#     for corner in corners:
#         system.part.add(pos=corner, q=0, type=3)
def generate_counterions(system, n_counter_ions, seed):
    logging.info("Creating Counterions ...")
    for _ in range(n_counter_ions):
        counter_ions_pos = np.array([np.random.rand()* BOX_L[0]
                                 ,np.random.rand()* BOX_L[0],
                                     9*np.random.rand()+3.5])
        system.part.add(pos=counter_ions_pos, q=1., type=0)


def steepest_descent_counterions(system,steps):
    FMAX = 0.001 * 0.25 * 1. / system.time_step**2
    logging.info("Removing overlap Counterions ...")
    logging.info("BEFORE MINIMIZATION " +str(system.analysis.energy()["total"]))
    system.integrator.set_steepest_descent(
        f_max=0,
        gamma=30.,
        max_displacement=0.01)
    system.integrator.run(steps)
    system.integrator.set_vv()

    logging.info("AFTER MINIMIZATION "+str(system.analysis.energy()["total"]))




def generate_wall_particle(system,walld, wallu,p_per_site,n_counter_ions):
    logging.info("Creating Wall Particles ...")

    xs = np.linspace(0,16,p_per_site+1)
    xs += (xs[1]-xs[0])/2
    xss, yss = np.meshgrid(xs[:-1],xs[:-1])
    for xs,ys in zip(xss,yss):
        for x,y in zip(xs,ys):
            system.part.add(pos=[x,y,walld], q=-(n_counter_ions/2)/(p_per_site**2), type=1,fix =[True,True,True])
            system.part.add(pos=[x,y,wallu], q=-(n_counter_ions/2)/(p_per_site**2), type=1, fix =[True,True,True])
def generate_constraint(const1,const2):
    bottom_constraint = espressomd.shapes.Wall(dist=const1, normal=[0, 0, 1])
    top_constraint = espressomd.shapes.Wall(dist=const2, normal=[0, 0, -1])

    system.constraints.add(shape=bottom_constraint, particle_type=2,penetrable=False)
    system.constraints.add(shape=top_constraint, particle_type=2,penetrable=False)

if VERSION_FLAG == "4.2":
    def turn_on_electrostatics(system, l_bjerrum,com_flag, seed):#4.2#
        if com_flag == "GPU":
            logging.info("Turn on electrostatics 4.2 GPU")
            p3m = espressomd.electrostatics.P3MGPU(
            prefactor=l_bjerrum,accuracy =1e-4,verbose = False)
        elif com_flag == "CPU":
            logging.info("Turn on electrostatics 4.2 CPU")
            p3m = espressomd.electrostatics.P3M(
            prefactor=l_bjerrum,accuracy =1e-4,verbose = False)
        else:
            print("error")
        elc = espressomd.electrostatics.ELC(actor=p3m,
                                            gap_size=5,
                                            maxPWerror=1e-4,
                                            )
        system.actors.add(elc)


    def set_hydrodynamic_interaction(system,wall_l, wall_r,fd,com_flag, seed):#4.2
        system.galilei.kill_particle_motion()
        system.galilei.galilei_transform()
        bottom_wall = espressomd.shapes.Wall(dist=wall_l, normal=[0, 0, 1])
        top_wall = espressomd.shapes.Wall(dist=wall_r, normal=[0, 0, -1])
        bottom_boundary = espressomd.lbboundaries.LBBoundary(shape=bottom_wall)
        top_boundary = espressomd.lbboundaries.LBBoundary(shape=top_wall)
        system.lbboundaries.add(bottom_boundary)
        system.lbboundaries.add(top_boundary)
        if com_flag == "CPU":
            logging.info("Set hydrodynamic interactions 4.2 CPU")
            lbfunc = espressomd.lb.LBFluid
        elif com_flag =="GPU":
            logging.info("Set hydrodynamic interactions 4.2 GPU")
            lbfunc = espressomd.lb.LBFluidGPU
        if fd == 0:
            lbf = lbfunc(agrid=1,
                        dens=26.18,visc=0.25,
                        tau=system.time_step,
                        kT=1,seed=seed)
        else:
            lbf = lbfunc(agrid=1,
                dens=26.18,visc=0.25,
                tau=system.time_step,kT=1,
                seed=seed,ext_force_density=[fd,0,0])                         
        system.actors.add(lbf)
        system.thermostat.set_lb(LB_fluid=lbf, seed=seed, gamma=15)
        return lbf

elif VERSION_FLAG== "WAL":
    def set_hydrodynamic_interaction(system,wall_l, wall_r,fd, com_flag,seed):
        system.galilei.kill_particle_motion()
        system.galilei.galilei_transform()
        if com_flag == "CPU":
            logging.info("Set hydrodynamic interactions Walberla CPU")
            lbfunc = espressomd.lb.LBFluidWalberla
        elif com_flag =="GPU":
            logging.info("Set hydrodynamic interactions Walberla GPU")
            lbfunc = espressomd.lb.LBFluidWalberlaGPU

        if fd == 0:
            lb=lbfunc(
            agrid=1.0, density=26.18, 
            kinematic_viscosity=0.25, tau=system.time_step, kT=1,)
            logging.info("Set hydrodynamic interactions Walberla without external force.")
        else:
            lb=lbfunc(
            agrid=1.0, density=26.18, 
            kinematic_viscosity=0.25, tau=system.time_step, kT=1,
            ext_force_density=[fd,0,0])
            logging.info(f"WARNING: Set hydrodynamic interactions Walberla withe external force {fd}")
        system.lb = lb
        system.thermostat.set_lb(LB_fluid=lb, seed=seed, gamma=15.0)
        bottom_wall = espressomd.shapes.Wall(dist=wall_l, normal=[0, 0, 1])
        top_wall = espressomd.shapes.Wall(dist=wall_r, normal=[0, 0, -1])

        lb.add_boundary_from_shape(shape=bottom_wall)
        lb.add_boundary_from_shape(shape=top_wall)
        return lb

    def turn_on_electrostatics(system, l_bjerrum,com_flag, seed):
        logging.info("Turn on electrostatics dev CPU dev")

        p3m = espressomd.electrostatics.P3M(
            prefactor=l_bjerrum,accuracy =1e-6, verbose = False)
        elc = espressomd.electrostatics.ELC(actor=p3m,
                                            gap_size=.5,
                                            maxPWerror=1e-7)
        system.electrostatics.solver = elc




def turn_on_e_field(system,force):
    system.part.select(type=0).ext_force = force

def calculate_dens_accu(system,obs):
    #obs = obs.calculate()
    accu =espressomd.accumulators.TimeSeries(obs=obs, delta_N=100)
    system.auto_update_accumulators.add(accu)
    return accu


def generate_accu(system,observable,id):
    obs =  observable(ids=id)
    accu =espressomd.accumulators.TimeSeries(obs=obs, delta_N=5)
    system.auto_update_accumulators.add(accu)
    return accu



if __name__ == "__main__":
    logging.info(f"{VERSION_FLAG} {COM_FLAG}")
    system = build_system(1)
    #dvisualizer= espressomd.visualization.openGLLive(system)
    generate_counterions(system,128,SEED)
    generate_wall_particle(system,WALLP1,WALLP2,8,128)
    generate_constraint(CONST1,CONST2)

    

    steepest_descent_counterions(system,1000)
    turn_on_electrostatics(system,L_BJERRUM,COM_FLAG,SEED)
    turn_on_e_field(system,1*E_FORCE)
    #visualizer.run(1)

    #system.integrator.run(1000)
    #accumulator1 = generate_accu(system,espressomd.observables.ParticlePositions,system.part.select(type=0).id )

    lbf = set_hydrodynamic_interaction(system, LBB1, LBB2,0,COM_FLAG,SEED)
    # visualizer= espressomd.visualization.openGLLive(system)
    # visualizer.run()
    eq_steps = int(5e4)
    tqdm_update = 100
    logging.info('Equilibrating')
    for _ in tqdm.tqdm(range(eq_steps//tqdm_update)):
         system.integrator.run(tqdm_update)
    total_stepps = int(1e5)
    tqdm_update = 100
    accumulator5 = calculate_dens_accu(system, espressomd.observables.LBVelocityProfile(
                                        n_x_bins=1,
                                        n_y_bins=1,
                                        n_z_bins=26,
                                        min_x=0,
                                        min_y=0,
                                        min_z=0,
                                        max_x=16,
                                        max_y=16,
                                        max_z=16, 
                                        allow_empty_bins=True,
                                        sampling_offset_x=0,
                                        sampling_offset_y=0,
                                        sampling_offset_z=0,
                                        sampling_delta_x=16,
                                        sampling_delta_y=16,
                                        sampling_delta_z=1
    ))
    accumulator6 = calculate_dens_accu(system, espressomd.observables.DensityProfile(
                                        n_x_bins=1,
                                        n_y_bins=1,
                                        n_z_bins=100,
                                        min_x=0,
                                        min_y=0,
                                        min_z=0,
                                        max_x=16,
                                        max_y=16,
                                        max_z=16,
                                        allow_empty_bins=True,
                                        sampling_offset_x=0,
                                        sampling_offset_y=0,
                                        sampling_offset_z=0,
                                        sampling_delta_x=1,
                                        sampling_delta_y=1,
                                        sampling_delta_z=0.5,
                                        ids = system.part.select(type=0).id
    ))
    accumulator7 = calculate_dens_accu(system, espressomd.observables.FluxDensityProfile(
                                        n_x_bins=1,
                                        n_y_bins=1,
                                        n_z_bins=32,
                                        min_x=0,
                                        min_y=0,
                                        min_z=0,
                                        max_x=16,
                                        max_y=16,
                                        max_z=16,
                                        allow_empty_bins=True,
                                        sampling_offset_x=0,
                                        sampling_offset_y=0,
                                        sampling_offset_z=0,
                                        sampling_delta_x=1,
                                        sampling_delta_y=1,
                                        sampling_delta_z=0.5,
                                        ids = system.part.select(type=0).id
    ))
    system.integrator.run(100)

    logging.info('Prod run started')
    for _ in tqdm.tqdm(range(total_stepps//tqdm_update)):
         system.integrator.run(tqdm_update)
    # with open("{folder}/traj_short.vtf", mode ="w+t") as f:
    #         espressomd.io.writer.vtf.writevsf(system,f)
    #         espressomd.io.writer.vtf.writevcf(system,f)
    #         for _ in range():
    #             system.integrator.run(1)
                # vzs = espressomd.observables.ParticleVelocities(ids=system.part.select(type=0).id).calculate()[:,2]
                # zs = espressomd.observables.ParticlePositions(ids=system.part.select(type=0).id).calculate()[:,2]
                # print(np.min(zs), np.max(zs))
                # print(np.argmin(zs), np.argmax(zs))
                # print(np.min(vzs), np.max(vzs))
                # print(np.argmin(vzs), np.argmax(vzs))
                # espressomd.io.writer.vtf.writevcf(system,f)

                

    folder = '4_2_CPU'
    if not os.path.isdir(f"./{folder}"):
        os.mkdir(f"./{folder}")
    filename = f"{VERSION_FLAG}_{COM_FLAG}_steps_{total_stepps}_const_{CONST1}_{CONST2}_lbb_{LBB1}_{LBB2}_walp_{WALLP1}_{WALLP2}"
    # np.save(f"./{folder}/ion_density_{filename}", accumulator1.time_series())
    np.save(f"./{folder}/flow_profile_{filename}.npy", accumulator5.time_series())
    np.save(f"./{folder}/density_profile_{filename}.npy", accumulator6.time_series())
    np.save(f"./{folder}/ions_flux_density_profile_{filename}.npy", accumulator7.time_series())

    logging.info("saving finished")
    logging.info(f"walls: {filename}")


    # fig1 = plt.figure(figsize=(10, 6))
    # plt.plot(fluid_positions, lbf[:,:,:].velocity[...,0].mean(axis=(0,1)), 'o', label='simulation')
    # #plt.yscale("log")
    # plt.xlabel('Position on the $x$-axis', fontsize=16)
    # plt.ylabel('Fluid velocity in $y$-direction', fontsize=16)
    # plt.legend()
    # plt.show()














    # np.save("../data/npy_wall_slit_pore_system", accumulator1.time_series())
    # np.save("../data/npy_counter_slit_pore_system", accumulator2.time_series())
    # np.save("../data/npy_box_slit_pore_system", accumulator2.time_series()[-1])


    #visualizer = espressomd.visualization.openGLLive(system)

    # def calculate_flux_profile_fluid(dim,lbf):
#     fluid_positions = (np.arange(lbf.shape[0]) + 0.5)
#     # get all velocities as Numpy array and extract y components only
#     fluid_velocities = (lbf[:,:,:].velocity)[:,:,:,dim]
#     # average velocities in y and z directions (perpendicular to the walls)
#     fluid_velocities = np.average(fluid_velocities, axis=(1,2))
#     fig1 = plt.figure(figsize=(10, 6))
#     plt.plot(fluid_positions, fluid_velocities, 'o', label='simulation')
#     plt.yscale("log")
#     plt.xlabel('Position on the $x$-axis', fontsize=16)
#     plt.ylabel('Fluid velocity in $y$-direction', fontsize=16)
#     plt.legend()
#     plt.show()

# def plot_flux(x_bins,y_bins, z_bins, fluxs):
#     bin_list = [x_bins, y_bins, z_bins]
#     fig1 , axes = plt.subplots(1,3)
#     dims = ["x", "y", "z"]
#     i=0
#     for bin , flux , dim in zip(bin_list, fluxs, dims) :
#         axes[i].plot(bin,flux,label=f'flux in {dim}-direction')
#         axes[i].legend(loc='best')
#         axes[i].set_xlabel(f'{dim}-position')
#         axes[i].set_ylabel(f'{dim}-flux density')
#         i+=1
#     fig1.set_size_inches(w=12,h=4)
#     plt.tight_layout()
#     fig1.savefig('profile.pdf',bbox_inches = "tight" ,dpi=600)

# @time_it
# def simulate_with_HI_sampling(system, total_steps):
#     vel_list =[]
#     logging.info(f"Simulate with hydrodynamic interactions for {total_steps} steps")
#     for _ in tqdm.tqdm(range(total_steps//5)):
#         system.integrator.run(5)
#         vel_list.append(system.actors[1][:,:,:].velocity[:,:,:,0])
#         #print(np.max(system.actors[1][:,:,:].velocity[:,:,:,:]),np.min(system.actors[1][:,:,:].velocity[:,:,:,0]))
#     vel_array = np.array("vel.npz",vel_list)
#     np.savez(vel_array)
#     return vel_array
# def main_thread(visualizer):
#     system.integrator.set_vv()
#     while True:
#         system.integrator.run(1)
#         visualizer.update()

