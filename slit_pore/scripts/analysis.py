from matplotlib import pyplot as plt
import scipy
import numpy as np

EPSILON_0=(1.428e-3)
PERMITTIVITY=78.54*1.428e-3
VALENCY = 1
KT =1
WIDTH = 12
VISCOSITY_DYNAMIC = 0.25*26.18
SURFACE_CHARGE_DENSITY = -64/(16*16)
#SURFACE_CHARGE_DENSITY = -64/(12*12) so würde es gut aussehen


def transcendental_equation(c, distance, kT, sigma, valency, permittivity) -> float:
    elementary_charge = 1.0
    return c * np.tan(valency * elementary_charge * distance / (4 * kT) * c) + sigma / (permittivity)

solution = scipy.optimize.fsolve(func=transcendental_equation,
                                  x0=0.5, args=(WIDTH, KT,
                                SURFACE_CHARGE_DENSITY, VALENCY,
                                PERMITTIVITY))


print(solution)
def eof_density(x, c, permittivity, elementary_charge, valency, kT):
    return c**2 * permittivity / (2 * kT) / (np.cos(valency * elementary_charge * c / (2 * kT) * x))**2

def eof_velocity(x, c, permittivity, elementary_charge, valency, kT, ext_field, distance, viscosity):
    return 2 * kT * ext_field * permittivity / (viscosity * elementary_charge * valency) * np.log(np.cos(valency * elementary_charge * c / (2 * kT) * x) / np.cos(valency * elementary_charge * c / (2 * kT) * distance / 2))


def eof_pressure_tensor(x, c, elementary_charge, valency, kT, ext_field, permittivity):
    return permittivity * ext_field * c * np.tan(valency * elementary_charge * c / (2 * kT) * x)

positions_ana = np.linspace(-WIDTH/2, WIDTH/2, 100)
positions = positions_ana + WIDTH/2 + 1.95

analytic_density_eof = eof_density(x=positions_ana, c=solution, permittivity=PERMITTIVITY,
                                 elementary_charge=1.0, valency=VALENCY, kT=KT)
analytic_velocity_eof = eof_velocity(x=positions_ana, c=solution, permittivity=PERMITTIVITY, 
                                     elementary_charge=1.0, valency=VALENCY, kT=KT, 
                                     ext_field=25, distance=WIDTH, 
                                     viscosity=VISCOSITY_DYNAMIC)

a=0
b=-1

positions_ions = np.load("data/ion_density_prod_run_force_1.npy")
positions_ions_z = positions_ions[:,:,2][a:b,:].flatten()

print(np.min(positions_ions_z), np.max(positions_ions_z))
dist, bins = np.histogram(positions_ions_z,bins=100)
bins += bins[1]-bins[0]
VERSION_FLAG = "WAL"
VERSION_FLAG = "4.2"

COM_FLAG = "GPU"
COM_FLAG = "CPU"


COM_FLAGS = ["CPU", "GPU"]
VERSION_FLAGS = ["4.2", "WAL"]
folder = '4_2_CPU'
folder2 = 'prodrun_figs'
total_stepps = int(1e5)

CONST1 = 1.25
CONST2 = -14.75
LBB1 = 1.5
LBB2 = -14.5
WALLP1 = 1
WALLP2 = 15
filename = f"{VERSION_FLAG}_{COM_FLAG}_steps_{total_stepps}_const_{CONST1}_{CONST2}_lbb_{LBB1}_{LBB2}_walp_{WALLP1}_{WALLP2}"


flow_profile = np.load(f"./{folder}/flow_profile_{filename}.npy")
ion_flux_profile = np.load(f"./{folder}/ions_flux_density_profile_{filename}.npy")

ion_dens_profile = np.load(f"./{folder}/density_profile_{filename}.npy")






bound =0
# flow_profile_average = np.zeros(26)
flow_profile_squeeze = flow_profile.squeeze(1).squeeze(1)[:,:,0]
flow_profile_average=flow_profile.squeeze(1).squeeze(1)[:,:,0].mean(axis=0)
# flow_profile_average=np.where(flow_profile_average!=0,flow_profile_average,np.nan)
plt.plot(positions, analytic_velocity_eof)
flows_zs = np.linspace(0,16,np.shape(flow_profile_average)[0])
print('Avg:', flow_profile_average.shape, flows_zs.shape)
plt.scatter(flows_zs, flow_profile_average, label = f"{bound}")
plt.legend()
plt.title(f"LB-Flow mean{LBB1} {LBB2}")
plt.savefig(f'{folder2}/flow_profile_{filename}.pdf')
plt.show()
plt.clf()
###




# ###
ion_profile_squeeze = ion_dens_profile.squeeze(1).squeeze(1)
bound = 0
ion_dens_profile_average=ion_dens_profile.squeeze(1).squeeze(1)[bound:,:].mean(axis=0)
ion_dens_profile_average=np.where(ion_dens_profile_average!=0,ion_dens_profile_average,np.nan)
plt.plot(positions, analytic_density_eof)
zs_ion_flux_profile_average = np.linspace(0,16, ion_dens_profile_average.shape[0])
plt.scatter(zs_ion_flux_profile_average,ion_dens_profile_average, label = f"{bound}")
plt.legend()
plt.title(f"Ion density (espresso obs)")
plt.savefig(f'{folder2}/ion_dens_profile_{filename}.pdf')
plt.show()
plt.clf()

###





assert False

plt.title("ion density and flow profile")
plt.scatter(flows_zs,ion_dens_profile_average/np.nanmax(ion_dens_profile_average), label = f"ion density (espresso) ")
plt.scatter(flows_zs,flow_profile_average/np.nanmax(flow_profile_average), label = f"LB flow profile (espresso)")
plt.plot(bins[:-1],dist/np.max(dist),label ='ion density (self impelemented)')
plt.legend()
plt.ylim(-0.2,1.1)
plt.savefig(f'{folder2}/combined_{filename}.pdf')



assert False


ion_flux_profile_squeeze = ion_flux_profile.squeeze(1).squeeze(1)[:,:,0]
bound =int(0.1*ion_flux_profile_squeeze.shape[0])
ion_flux_profile_average=ion_flux_profile.squeeze(1).squeeze(1)[bound:,:,0].mean(axis=0)
ion_flux_profile_average=np.where(ion_flux_profile_average!=0,ion_flux_profile_average,np.nan)
# plt.plot(positions,analytic_velocity_eof)
print('Avg:', ion_flux_profile_average)
zs_ion_flux_profile_average = np.linspace(0,16,ion_flux_profile_average.shape[0])
plt.scatter(zs_ion_flux_profile_average,ion_flux_profile_average, label = f"{bound}")
plt.legend()
plt.title(f"Ion Flux (espresso obs)")
plt.savefig(f'{folder2}/ion_flux_profile_{filename}.pdf')
plt.show()
plt.clf()


plt.plot(xs,ys)
plt.title("Flow veclocity")
plt.show()

plt.plot(center_times,center_vel)
plt.plot(center_times,smoothed_timeseries)
plt.title("time series flow center")
plt.show()

#vels_data = np.load("./testdata/vels_4_2_LB_steps_5000_walls_1_-14.npz")
# lb_vels = vels_data["v"]
# lb_pos = vels_data["x"]

# plt.plot(lb_pos,lb_vels)
# plt.show()



plt.title("LB-Flow at the end")
plt.show()
plt.clf()

plt.title("ion density (self implemented)")
plt.plot(bins[:-1],dist)
plt.savefig(f'{folder2}/ion_density_{filename}.pdf')

plt.show()
plt.clf()