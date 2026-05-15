import numpy as np
import pandas as pd
from Py6S import *
import argparse
import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser()
parser.add_argument("--atmos", help="Atmospheric profile type")
parser.add_argument("--aerosol", help="Aerosol model type")
parser.add_argument("--output_dir", help="Output directory for results")
args = parser.parse_args()


# Set atmospheric type
## Atmospheric profile
'''
    NoGaseousAbsorption = 0
    Tropical = 1
    MidlatitudeSummer = 2
    MidlatitudeWinter = 3
    SubarcticSummer = 4
    SubarcticWinter = 5
    USStandard1962 = 6
'''
atmos_type = str(args.atmos) if args.atmos else "1"

## Aerosol model
'''
    NoAerosols = 0
    Continental = 1
    Maritime = 2
    Urban = 3
    Desert = 5
    BiomassBurning = 6
    Stratospheric = 7
'''
aerosol_type = str(args.aerosol) if args.aerosol else "2"

output_dir = args.output_dir if args.output_dir else "."
os.makedirs(output_dir, exist_ok=True)


# Main code
s = SixS()
atmos_map = {
    "0": "NoGaseousAbsorption",
    "1": "Tropical",
    "2": "MidlatitudeSummer",
    "3": "MidlatitudeWinter",
    "4": "SubarcticSummer",
    "5": "SubarcticWinter",
    "6": "USStandard1962"
}
aerosol_map = {
    "0": "NoAerosols",
    "1": "Continental",
    "2": "Maritime",
    "3": "Urban",
    "5": "Desert",
    "6": "BiomassBurning",
    "7": "Stratospheric"
}
print(f"Selected atmospheric profile: {atmos_map[atmos_type]}")
print(f"Selected aerosol model: {aerosol_map[aerosol_type]}")


s.atmos_profile = atmos_type

# aerosol model
s.aero_profile = aerosol_type

# visibility (km)
s.visibility = 40

s.geometry = Geometry.User()

s.geometry.solar_z = 30

s.geometry.view_z = 0

s.geometry.view_a = 0

s.altitudes.set_sensor_satellite_level()
s.altitudes.set_target_sea_level()

wavelengths = np.arange(0.38, 2.6, 0.005)

irradiance = []

for wl in wavelengths:

    s.wavelength = Wavelength(wl)

    s.run()

    # solar irradiance
    irr = s.outputs.pixel_radiance

    irradiance.append(irr)

# save results
df = pd.DataFrame({
    "wavelength_um": wavelengths,
    "solar_irradiance": irradiance
})

plt.plot(df["wavelength_um"], df["solar_irradiance"])
plt.xlabel("Wavelength (um)")
plt.ylabel("Solar Irradiance")
plt.title(f"Solar Spectrum ({atmos_map[atmos_type]}, {aerosol_map[aerosol_type]})")
plt.grid()
plt.savefig(f"{output_dir}/solar_spectrum_{atmos_type}_{aerosol_type}.png")
#plt.show()

df.to_csv(f"{output_dir}/solar_spectrum_{atmos_type}_{aerosol_type}.csv", index=False)

print(df.head())