

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxspec.model.additive import Powerlaw
import numpy as np
import astropy.units as u
from jaxspec.model.multiplicative import Tbabs
import numpyro
from jax import config

def norm_estimation (nearby_sources_table: Table, parameters=parameters, model, index = index):
    parameters = {}
    parameters = {
        "tbabs_1": {"N_H": np.full(size, nearby_sources_table["Nh"][index]/1e22)},
        "powerlaw_1": {
            "alpha": np.full(size, nearby_sources_table["Photon Index"][index] if nearby_sources_table["Photon Index"][index] > 0.0 else 1.7),
            "norm": np.full(size, 1e-5),
        }
    }
    Total_flux = nearby_sources_table["SC_EP_8_FLUX"][index]
    target_flux = Total_flux * (u.erg / u.cm**2 / u.s)

    tolerance = 1e-5 * u.erg / (u.cm**2 * u.s)  # in erg/cm²/s
    max_iterations = 10000  # Maximum number of iterations to avoid infinite loop
    learning_rate = 1.0e6 * (u.cm**2 * u.s) / u.erg  # in (cm² s) / erg
    energies = jnp.geomspace(0.3, 10, 100)
    
    for iteration in range(max_iterations):
        energy_flux = model.energy_flux(parameters, energies[:-1], energies[1:], n_points=100) * (u.keV/ ( u.s * u.cm**2))
        total_computed_flux = np.sum(energy_flux)  # Sum across the energy bands
        total_computed_flux = total_computed_flux.to(u.erg / u.cm**2/ u.s)  # Sum across the energy bands

        # Calculate the difference between computed flux and target flux
        flux_difference = target_flux - total_computed_flux

        # Check if the computed flux is close enough to the target flux
        if np.abs(flux_difference) < tolerance:
            print(f"Matching flux found with norm: {parameters['powerlaw_1']['norm']} at iteration {iteration}")
            break

        norm = norm + flux_difference * learning_rate
        parameters['powerlaw_1']['norm'] = norm
    return parameters
