# ------------------------------ #
        # Python's packages
        
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from typing import Tuple, Dict, Union
from termcolor import colored
from scipy.optimize import curve_fit
from astropy.units import Quantity
from tqdm import tqdm

# ---------- import function ---------- #

import function.calculation_function as c_f
import function.unique_function as u_f
import function.init_function as i_f
from function.log_function import log, RESULT
# ------------------------------------- #

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import catalog_information as dict_cat

# ------------------------------ #

# ---------- for documentation ---------- #

# import src.function.calculation_function as c_f
# import src.function.unique_function as u_f
# import src.catalog_information as dict_cat

# --------------------------------------- #


class ChandraCatalog:
    """
    A class for handling and analyzing data from the Chandra Source Catalog (CSC 2.0, read from Chandra.fits).

    This class is expertly crafted to manage and interpret data from the Chandra X-ray Observatory, one of the foremost 
    tools in modern astrophysics for observing X-ray sources. Its functions include identifying nearby astronomical sources,
    computing photon indices, and modeling astronomical data for in-depth analysis.

    The online cone search of the Chandra Source Catalog (CS_Chandra) was removed; it is kept in the git tag
    chandra-cone-search-v1 and on the branch legacy-modes.

    Important:
        - This class is tailored to work with the specific data formats and analytical needs of Chandra X-ray data.
        - It offers advanced functionalities such as variability analysis, photon index calculations, and custom source modeling.
        - The class plays a vital role in X-ray astronomy, enabling researchers to delve deeper into understanding the 
          universe's high-energy phenomena.
        - Users should have a basic understanding of X-ray astronomy and data structures commonly used in the field for 
          effective utilization of this class.

    Note:
        - ChandraCatalog is not just a data processing tool; it's a bridge between raw observational data and actionable 
          astronomical insights.
        - While designed for Chandra X-ray data, the methodologies and techniques employed are adaptable to other X-ray 
          observatories' data.
        - The class is part of a larger ecosystem of tools aimed at enhancing our understanding of the cosmos through X-ray 
          astronomy.

    Example:
        Creating an instance of ChandraCatalog with necessary parameters:
        
        >>> chandra_catalog = ChandraCatalog("path/to/chandra_catalog.fits", 5*u.arcmin, simulation_data, user_table)

    The ChandraCatalog class stands as a testament to the power of combining detailed observational data with sophisticated 
    analysis techniques, pushing the boundaries of astronomical research and discovery.
    """
    
    def __init__(self, catalog_path: str, radius: Quantity, simulation_data: Dict, user_table: Table) -> None:
        """
        Initializes the Chandra class, setting up the necessary attributes for data analysis.

        This constructor loads the Chandra catalog from the given path, finds the nearby sources, and prepares the data for
        further analysis such as photon index calculation and model generation.

        Args:
            catalog_path (str): Path to the primary Chandra catalog file. This file is used to initialize the Chandra catalog.
            radius (Quantity): The search radius for finding nearby sources. This radius is utilized for finding nearby sources.
            simulation_data (Dict): A dictionary containing simulation data. This includes 'object_data' for object-specific information and 'os_dictionary' for operating system-specific information.
            user_table (Table): An optional Astropy Table provided by the user to include in the analysis.

        Attributes:
        
        .. attribute:: ra
            :type: str
            :value: RA
            
            Right ascension key for the Chandra catalog.
            
        .. attribute:: dec
            :type: str
            :value: DEC
            
            Declination key for the Chandra catalog.
            
        .. attribute:: chandra_catalog
            :type: Table
            
            Catalog data loaded from the Chandra catalog file.
            
        .. attribute:: nearby_sources_table
            :type: Table
            
            Table of nearby sources found in the Chandra catalog.
            
        .. attribute:: nearby_sources_position
            :type: SkyCoord
            
            Positions of nearby sources found in the Chandra catalog.

        .. attribute:: photon_index
            :type: list
            
            List of photon index values for sources in the Chandra catalog.
            
        .. attribute:: model_dictionary
            :type: dict
            
            Dictionary of models for sources in the Chandra catalog.

        Important:
            The constructor performs several key operations:
            
            - Initializes coordinate keys for the Chandra catalog from a global dictionary.
            - Loads the Chandra catalog from the specified path.
            - Finds nearby sources based on the Chandra catalog and additional user-provided data.
            - Prepares the data for further analysis, including calculating photon indices and generating source models.
        """    
        
        # ---------- coord ---------- #
        self.ra = dict_cat.dictionary_coord["Chandra"]["right_ascension"]
        self.dec = dict_cat.dictionary_coord["Chandra"]["declination"]
        # --------------------------- #
        

        self.chandra_catalog = self.open_catalog(catalog_path=catalog_path)
        self.nearby_sources_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, object_data=simulation_data["object_data"], user_table=user_table)
        test_nh_path = os.path.join(simulation_data["os_dictionary"]["catalog_datapath"], "NHI_HPX.fits").replace("\\", "/")
        nh_path = i_f.get_valid_file_path(test_nh_path)
        self.nh_dictionnary = self.open_catalog(catalog_path=nh_path)
        self.neighbourhood_of_object(radius=radius, simulation_data=simulation_data)
        self.photon_index = self.get_phoindex_nh()
        self.model_dictionary = self.dictionary_model()

        

    def open_catalog(self, catalog_path: str) -> Table:
        """
        Opens a FITS file from the specified path and converts it into an Astropy Table.

        This method is designed to load a catalog from a FITS file, which is a common file format used in astronomy for storing data tables and images. 
        The method utilizes the 'fits' module from Astropy to read the file and loads the data into a table format for easier manipulation and analysis.

        Args:
            catalog_path (str): The file path to the FITS file containing the catalog data. This file should conform to the FITS standard and typically contains astronomical data.

        Returns:
            Table: 
                An Astropy Table object containing the data from the FITS file. The table format facilitates various operations like data filtering, sorting, and visualization.
        """
        
        with fits.open(catalog_path, memmap=True) as data:
            return Table(data[1].data)
        
        
    def find_nearby_sources(self, radius: Quantity, object_data: dict, user_table: Table) -> Tuple[Table, SkyCoord]:
        """
        Searches for astronomical sources near a specified object within the Chandra Source Catalog.

        This method identifies sources in the Chandra Source Catalog that are located within a specified radius of a given 
        celestial object. It uses the object's position and a defined field of view to filter sources from the catalog. 
        The method returns a table of nearby sources along with their sky coordinates.

        Args:
            radius (Quantity): The radius within which to search for sources, as an astropy Quantity object (value and unit).
            object_data (dict): A dictionary containing the object's information. It must have keys 'object_name' and 
                                'object_position', where 'object_position' should be a SkyCoord object.
            user_table (Table): An astropy Table object representing the user's data. This table is not used in the current 
                                implementation of the function.

        Returns:
            Tuple[Table, SkyCoord]: 
                A tuple containing two elements. The first element is an astropy Table with the catalog of nearby sources. 
                The second element is a SkyCoord object containing the positions of these sources.

        Note:
            - The search radius is internally expanded by 5 arcminutes to define a field of view.
            - The function prints various messages during execution, including progress updates and results summary.
            - If no sources are found, the function exits the program.
            - In case of an error, the error message is printed.

        """
    
        field_of_view = radius + 5*u.arcmin
        object_name = object_data["object_name"]
        object_position = object_data['object_position']

        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view

        small_table = Table(names= self.chandra_catalog.colnames,
                            dtype= self.chandra_catalog.dtype)

        nearby_sources_table = Table(names= self.chandra_catalog.colnames,
                                     dtype= self.chandra_catalog.dtype)
        
        log.debug(fr"{colored('Reducing Chandra catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
        for number in tqdm(range(len(self.chandra_catalog))):
            if min_ra/u.deg < self.chandra_catalog[self.ra][number] < max_ra/u.deg and min_dec/u.deg < self.chandra_catalog[self.dec][number] < max_dec/u.deg:
                small_table.add_row(self.chandra_catalog[number])
                
        sources_position = SkyCoord(ra=small_table[self.ra], dec=small_table[self.dec], unit=u.deg)

        log.debug(f"{colored(f'Find sources close to {object_name} with Chandra catalog', 'blue')}")
        for number in tqdm(range(len(small_table))):
            if c_f.ang_separation(object_position, sources_position[number]) < radius:
                nearby_sources_table.add_row(small_table[number])
                
        column_name = {"source_name": "Chandra_IAUNAME", 
                       "right_ascension": self.ra,
                       "declination": self.dec,
                       "catalog_name": "Chandra"}
            
        if len(nearby_sources_table) != 0:
            unique_table = u_f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table, column_name=column_name)
        else:
            log.debug("Nearby sources table from Chandra catalog is empty.")
            sys.exit()
            
        nearby_sources_position = SkyCoord(ra=unique_table[self.ra], dec=unique_table[self.dec], unit=u.deg)

        unique_table, nearby_sources_position = c_f.exclude_target_from_nearby_sources(
            nearby_sources_table=unique_table,
            object_position=object_position,
            object_name=object_name,
            ra_col=self.ra,
            dec_col=self.dec,
            iauname_col=dict_cat.dictionary_coord["Chandra"]["iauname"],
        )

        try :
            if len(unique_table) != 0:
                log.info(f"Chandra: {len(unique_table)} sources detected close to {object_name}")
                return unique_table, nearby_sources_position
            else:
                log.warning(f"No sources detected close to {object_name}.")
        except Exception as error:
            log.warning(f"An error occured : {error}")
            
    
    def neighbourhood_of_object(self, radius: Quantity, simulation_data: Dict) -> None:
        """
        Visualizes the neighborhood of a specified astronomical object within a given radius.

        This method creates a set of plots to visually represent the neighborhood of a celestial object. It uses data from 
        the Chandra Source Catalog and a simulation data dictionary. The plots include the positions of nearby sources, 
        The plot is saved as a PNG file and also displayed.

        Args:
            radius (Quantity): The search radius around the celestial object, specified as an astropy Quantity.
            simulation_data (Dict): A dictionary containing various simulation parameters and data. This must include keys 
                                    'os_dictionary' for output settings and 'object_data' for the celestial object's data. 
                                    'object_data' itself must be a dictionary containing the 'object_name' and 'object_position'.

        Note:
            - The method creates and displays one scatter plot of the nearby sources.
            - The object's position is marked distinctly in each plot.
            - The generated plots are saved to a file in the directory specified in 'os_dictionary' within 'simulation_data'.
            - The method does not return any value.
        
        Here is an example of the plot create by this method:
        
        .. image:: C:/Users/plamb_v00y0i4/OneDrive/Bureau/Optimal_Pointing_Point_Code/modeling_result/PSR_J0437-4715/Chandra/img/neighbourhood_of_PSR_J0437-4715.png
            :align: center

        """
        
        os_dictionary = simulation_data["os_dictionary"]
        object_data = simulation_data["object_data"]
        object_name = object_data["object_name"]
        
        csc_ra = np.array(self.nearby_sources_table[self.ra], dtype=float)
        csc_dec = np.array(self.nearby_sources_table[self.dec], dtype=float)

        figure_1, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.set_title(f"Neighbourhood of {object_name}, radius = {radius}")
        axes.scatter(csc_ra, csc_dec, s=10, c='black', marker="*", label=f"Sources close to {object_name} : {len(csc_ra)}")
        axes.scatter(object_data['object_position'].ra, object_data['object_position'].dec, marker='x', c='red', label=f"{object_name}")
        axes.set_xlabel('Right Ascension [deg]')
        axes.set_ylabel('Declination [deg]')
        axes.legend(loc='upper right')

        plt.savefig(os.path.join(os_dictionary["img"], f"neighbourhood_of_{object_name}.png".replace(" ", "_")))
        plt.show()


    def visualization_interp(self, optimization_parameters, photon_index, key) -> None:
        """
        Creates a logarithmic plot visualizing the interpolation of photon index values across different energy bands.

        This method generates a plot that visualizes the relationship between energy bands and fluxes, considering the 
        photon index values. It uses optimization parameters and photon index values for each energy band and displays the 
        results as an errorbar plot, with power-law models for absorption.

        Args:
            optimization_parameters (list): A list of tuples, each containing optimization parameters. Each tuple typically 
                                            includes flux observations, flux errors, and absorbed power-law values.
            photon_index (list): A list of photon index values corresponding to the optimization parameters.
            key (str): A key to access the energy band center values from 'dict_cat.dictionary_catalog'.

        Note:
            - The method retrieves energy band center values from 'dict_cat.dictionary_catalog' using the provided key.
            - The plot is logarithmic, displaying energy in keV and flux in erg/cm^2/s/keV.
            - Each set of flux observations is plotted with error bars, and absorbed power-law models are plotted for each photon index.
            - The plot is titled "Interpolation Photon Index plot" and includes appropriate axis labels.
        """
        
        energy_band = dict_cat.dictionary_catalog[key]["energy_band_center"]
        
        fig, axes = plt.subplots(1, 1, figsize=(15, 8))
        fig.suptitle("Interpolation Photon Index plot", fontsize=20)
        fig.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center')
        fig.text(0.04, 0.5, 'Flux $[erg.cm^{-2}.s^{-1}.keV^{-1}]$', ha='center', va='center', rotation='vertical')
        
        for item in range(len(optimization_parameters)):
            flux_obs = optimization_parameters[item][1]
            flux_obs_err = optimization_parameters[item][2]
            absorbed_power_law = optimization_parameters[item][3]
            absorb_pho_index = photon_index[item]
            
            axes.errorbar(energy_band, flux_obs, flux_obs_err, fmt='*', color='red', ecolor='black')
            axes.plot(energy_band, absorbed_power_law, label=f"$\Gamma$ = {absorb_pho_index:.8f}")
            
        axes.legend(loc="upper left", ncol=4, fontsize=6)
        axes.loglog()
        
        plt.show()


    def optim_index(self, key, table, index):
        """
        Calculates the photon index for a given source using absorbed power-law fitting.

        This method computes the photon index for an astronomical source based on its flux observations across different 
        energy bands. It uses an absorbed power-law model for the fit. The method selects the appropriate data based on the 
        provided catalog key and then performs the fitting process to determine the photon index and other optimization 
        parameters.

        Args:
            key (str): The key to identify the catalog and retrieve the relevant data from 'dict_cat.dictionary_catalog'.
            table (Table): An astropy Table containing flux data for the source in various energy bands.
            index (int): The index of the source in the table for which the photon index is to be calculated.

        Returns:
            Tuple[float, Tuple]: A tuple containing the photon index and a tuple of optimization parameters. The optimization 
                                 parameters include energy band centers, observed fluxes, flux errors, and absorbed power-law 
                                 values.

        Note:
            - The absorbed power-law fitting is done using the curve_fit function from scipy.optimize.
            - If the fitting process fails, a default photon index value of 1.7 is returned.
            - The optimization parameters are used for plotting in the 'visualization_interp' method.

        """
        
        if key == "Chandra":
            interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                           "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                           "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"],
                           "energy_band_half_width": dict_cat.dictionary_catalog[key]["energy_band_half_width"]}


        def absorbed_power_law(energy_band, constant, gamma):
            energy_band = np.array(energy_band, dtype=float)
            sigma = np.array([1e-20, 1e-22, 1e-24], dtype=float)
            return (constant * energy_band ** (-gamma)) * (np.exp(-sigma * 3e20))
        
        tab_width = 2 * interp_data["energy_band_half_width"]
        
        flux_obs = [table[band_flux][index] for band_flux in interp_data["band_flux_obs"]]
        
        flux_err = [[table[err_0][index] for err_0 in interp_data["band_flux_obs_err"][0]],
                    [table[err_1][index] for err_1 in interp_data["band_flux_obs_err"][1]]]
        
        flux_err_obs = [np.mean([err_0, err_1]) for (err_0, err_1) in zip(flux_err[0], flux_err[1])]
        
        y_array = [num/det for num, det in zip(flux_obs, tab_width)]
        yerr_array = [num/det for num, det in zip(flux_err_obs, tab_width)]
        
        try:
            popt, pcov = curve_fit(absorbed_power_law, interp_data["energy_band_center"], y_array, sigma=yerr_array, absolute_sigma=True)
            constant, photon_index = popt
            perr = np.sqrt(np.diag(pcov))
            p_err = perr[1]
            absorb_random = np.random.normal(photon_index, p_err)
    
        except Exception as error:
            # Default values if curve fitting fails
            constant = 1e-14  # Reasonable default for constant
            photon_index = 1.7  # Default for photon index
            popt = [constant, photon_index]  # Use these defaults for popt
            p_err = 0.0
        optimization_parameters = (interp_data["energy_band_center"], y_array, yerr_array, absorbed_power_law(interp_data["energy_band_center"], *popt))   
        return photon_index, optimization_parameters , p_err
        

    def get_phoindex_nh(self):
        """
        Computes and assigns photon index and hydrogen column density (Nh) values for the nearby sources.

        The photon index of each nearby source is fitted to its band fluxes (optim_index), and N_H is taken from the
        NHI_HPX map at the source position. Both are added to the table as new columns, and bad fits get the default
        photon index (c_f.apply_bad_fit_default).

        Returns:
            List[float]: Photon index values for the nearby sources (before the bad-fit default is applied).
        """
        
        key = "Chandra"

        photon_index_list, parameters_list, perr_list = [], [], []
        self.nh_list = []

        self.nearby_sources_table["Chandra_IAUNAME"] = [n.strip() for n in self.nearby_sources_table["Chandra_IAUNAME"]]
        for index, name in enumerate(list(self.nearby_sources_table["Chandra_IAUNAME"])):
            name = name.strip()  # Remove leading/trailing spaces

            if name in self.nearby_sources_table["Chandra_IAUNAME"]:
                nearby_index = list(self.nearby_sources_table["Chandra_IAUNAME"]).index(name)
                log.debug(f"Index : {index} - Name : {name}")
                #else:
                photon , params, perr = self.optim_index(key=key, table=self.nearby_sources_table, index=nearby_index)
                parameters_list.append(params)
                photon_index_list.append(photon)
                perr_list.append(perr)
                    
                ra_val = self.nearby_sources_table[self.ra][index]
                dec_val = self.nearby_sources_table[self.dec][index]
                
                distances = np.sqrt((self.nh_dictionnary['RA2000'] - ra_val)**2 + (self.nh_dictionnary["DEC2000"] - dec_val)**2)
                best_match_index = np.argmin(distances)

                nhi_value = self.nh_dictionnary['NHI'][best_match_index]
                self.nh_list.append(nhi_value)
                

                
        self.visualization_interp(optimization_parameters=parameters_list, photon_index=photon_index_list, key=key)
        self.nearby_sources_table["Photon Index"] = photon_index_list
        self.nearby_sources_table["Nh"] = self.nh_list
        self.nearby_sources_table["Photon Index Error"] = perr_list

        c_f.apply_bad_fit_default(self.nearby_sources_table, iauname_col="Chandra_IAUNAME")

        return photon_index_list
                
  
    def dictionary_model(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Constructs a dictionary with model parameters for the nearby sources.

        Each entry details the source's astrophysical model type (e.g., 'power'), model value (e.g., photon index),
        flux and column density.

        Returns:
            Dict[str, Dict[str, Union[str, float]]]: One entry per source, keyed by a unique identifier (e.g., 'src_0'),
                                                    containing the source's model, model value, flux, and column density.

        Note:
            - The method assumes that the 'Photon Index' and 'Nh' values have already been calculated and added to the table.
            - Currently, the method only handles the 'power' model type. Placeholder code exists for other model types ('black_body', 'temp'), but these are not yet implemented.
            - Each source is assigned a unique key in the dictionary (e.g., 'src_0', 'src_1', etc.).
            - The method utilizes 'dict_cat.dictionary_catalog' to access the flux observation column name.
        """
        
        model_dictionary = {}
        number_source = len(self.nearby_sources_table)

        model = np.array([], dtype=str)
        model_value = []
        for item in range(number_source):
            model = np.append(model, 'power')

        for item in range(number_source):
            if model[item] == 'power':
                model_value = np.append(model_value, self.nearby_sources_table["Photon Index"][item])
            elif model[item] == 'black_body':
                pass # Pas de valeur pour le moment...
            elif model[item] == 'temp':
                pass # Pas de valeur pour le moment... (dernier model pimms)

        flux = list(self.nearby_sources_table[dict_cat.dictionary_catalog["Chandra"]["flux_obs"]])

        for item in range(number_source):
            dictionary = {
                "model": model[item],
                "model_value": model_value[item],
                "flux": flux[item],
                "column_dentsity": self.nh_list[item]
            }
            model_dictionary[f"src_{item}"] = dictionary

        return model_dictionary

