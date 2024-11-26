import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxspec.model.additive import Powerlaw
import numpy as np
import astropy.units as u
from jaxspec.model.multiplicative import Tbabs
import numpyro
from jax import config

config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")

model = Powerlaw() * Tbabs()

target_flux = 1.05845e-12 * (u.erg / u.cm**2 / u.s)


tolerance = 1e-18 * u.erg / (u.cm**2 * u.s)  # in erg/cm²/s

max_iterations = 10000  # Maximum number of iterations to avoid infinite loop

learning_rate = 1.0e6 * (u.cm**2 * u.s) / u.erg  # in (cm² s) / erg

parameters = {}

#energies = [0.35, 0.75, 1.5, 3.25, 8.25]
energies = jnp.geomspace(0.3, 10, 100)

norm = 0.00133
for iteration in range(max_iterations):
    parameters = {
    "tbabs_1": {"N_H": 5.62885420517436E+21/1e22},
    "powerlaw_1": {
    "alpha":4.10628986358643,  # Scalar value
    "norm": norm ,
    }
}
    # Compute the flux with the current norm
    #energy_flux = model.energy_flux(parameters, energies[:-1], energies[1:], n_points=100) * ((u.erg * u.s) / u.cm**2)
    energy_flux = model.energy_flux(parameters, energies[:-1], energies[1:], n_points=100) * (u.keV/ ( u.s * u.cm**2))
    total_computed_flux = np.sum(energy_flux)  # Sum across the energy bands
    total_computed_flux = total_computed_flux.to(u.erg / u.cm**2/ u.s)  # Sum across the energy bands

    # Calculate the difference between computed flux and target flux
    flux_difference = target_flux - total_computed_flux

    # Check if the computed flux is close enough to the target flux
    if np.abs(flux_difference) < tolerance:
        print(f"Matching flux found with norm: {parameters['powerlaw_1']['norm']} at iteration {iteration}")
        break

    # Adjust the norm based on the flux difference
    #parameters['powerlaw_1']['norm'] += flux_difference * learning_rate
    norm = norm + flux_difference * learning_rate

# Print the results
print(f"The final norm is {parameters['powerlaw_1']['norm']}")
print(f"The flux from the XMM Newton IRAP survey is {target_flux}")
print(f"The computed flux with the model is {total_computed_flux}")