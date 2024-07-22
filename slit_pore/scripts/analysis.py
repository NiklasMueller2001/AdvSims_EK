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



positions_ions = np.load("data/ion_density_prod_run_force_1.npy")
a=-0
b=-1
print(positions_ions.shape)
positions_ions_z = positions_ions[:,:,2][a:b,:].flatten()

print(np.min(positions_ions_z), np.max(positions_ions_z))
dist, bins = np.histogram(positions_ions_z,bins=100)
bins += bins[1]-bins[0]

totalsteps = int(5e4)
wall1 = 1.5
wall2 = 14.5
filename =f'4_2_steps_{totalsteps}_walls_{wall1}_-{wall2}_25'
print("dist", dist.shape)
print("bins", bins.shape)
folder = 'prodrun'
folder2 = 'prodrunfigs'
flow_profile = np.load(f"./{folder}/flow_profile_{filename}.npy")
ion_flux_profile = np.load(f"./{folder}/ions_flux_density_profile_{filename}.npy")

ion_dens_profile = np.load(f"./{folder}/density_profile_{filename}.npy")

#flow_profile = np.load("./{folder}/flow_profile_4_2_LB_steps_5000_walls_1_-14.npy")





print(flow_profile.shape)
flow_profile_timeseries_center=flow_profile.squeeze(1).squeeze(1)[:,10,0]
flow_profile_end=flow_profile.squeeze(1).squeeze(1)[-1,:,0]


#flow_profile_average=flow_profile.squeeze(1).squeeze(1)[:,:,0].mean(axis=0)
#print(flow_profile_average.shape)

flows_zs = np.linspace(1.5,14.5,26)

center_vel = flow_profile_timeseries_center
smoothed_timeseries = scipy.ndimage.uniform_filter1d(center_vel,100)
center_times = np.arange(center_vel.shape[0])

bounds = [[1,-1],[500,1000], [1000,2000], [5000,10000],[50000,100000],[100000,150000]]
#bounds = [[100000,150000]]


print(flow_profile.shape)
flow_profile_squeeze = flow_profile.squeeze(1).squeeze(1)[:,:,0]
bound =int(0.1*flow_profile_squeeze.shape[0])
flow_profile_average=flow_profile.squeeze(1).squeeze(1)[bound:,:,0].mean(axis=0)
# flow_profile_average=np.where(flow_profile_average!=0,flow_profile_average,np.nan)
plt.plot(positions, analytic_velocity_eof)
print('Avg:', flow_profile_average)
plt.scatter(flows_zs, flow_profile_average, label = f"{bound}")
plt.legend()
plt.title(f"LB-Flow mean{wall1} {wall2}")
plt.savefig(f'{folder2}/flow_profile_{filename}.pdf')
plt.show()
plt.clf()
###

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


# ###
ion_profile_squeeze = ion_dens_profile.squeeze(1).squeeze(1)
bound = 0
ion_dens_profile_average=ion_dens_profile.squeeze(1).squeeze(1)[bound:,:].mean(axis=0)
ion_dens_profile_average=np.where(ion_dens_profile_average!=0,ion_dens_profile_average,np.nan)
plt.plot(positions, analytic_density_eof)
plt.scatter(zs_ion_flux_profile_average,ion_dens_profile_average, label = f"{bound}")
plt.legend()
plt.title(f"Ion density (espresso obs)")
plt.savefig(f'{folder2}/ion_dens_profile_{filename}.pdf')
plt.show()
plt.clf()

###





plt.plot(flows_zs,flow_profile_end)
plt.title("LB-Flow at the end")
plt.show()
plt.clf()

plt.title("ion density (self implemented)")
plt.plot(bins[:-1],dist)
plt.savefig(f'{folder2}/ion_density_{filename}.pdf')

plt.show()
plt.clf()

# assert False

plt.title("ion density and flow profile")
plt.scatter(flows_zs,ion_dens_profile_average/np.nanmax(ion_dens_profile_average), label = f"ion density (espresso) ")
plt.scatter(flows_zs,ion_flux_profile_average/np.nanmax(ion_flux_profile_average), label = f"ion flux (espresso)")
plt.scatter(flows_zs,flow_profile_average/np.nanmax(flow_profile_average), label = f"LB flow profile (espresso)")
plt.plot(bins[:-1],dist/np.max(dist),label ='ion density (self impelemented)')
plt.legend()
plt.ylim(-0.2,1.1)
plt.savefig(f'{folder2}/combined_{filename}.pdf')



assert False
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