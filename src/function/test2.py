
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxspec.model.additive import Powerlaw
import numpy as np
import astropy.units as u

# Define the spectral model as a Powerlaw without absorption
spectral_model = Powerlaw()

# Define the energy range (in keV) for which to compute the flux
energies = jnp.geomspace(0.3, 10, 30)

# Define parameters for the power law model
params = {
    'powerlaw_1': {
        'alpha': 1.7,    # Power law index
        'norm': 0.001    # Normalization
    }
}

# Calculate the energy flux
energy_flux = spectral_model.energy_flux(params, energies[:-1], energies[1:], n_points=30)
photo_flux = spectral_model.photon_flux(params, energies[:-1], energies[1:], n_points=30)

total_computed_energy_flux = np.sum(energy_flux)
total_computed_photo_flux = np.sum(photo_flux)

print("Energy flux in the range 0.3 to 10 keV:", energy_flux)
print("Photon flux in the range 0.3 to 10 keV:", photo_flux)

print("Total energy flux in the range 0.3 to 10 keV:", total_computed_energy_flux)
print("Total photon flux in the range 0.3 to 10 keV:", total_computed_photo_flux)




