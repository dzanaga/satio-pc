import numpy as np
import xarray as xr
from satio_pc.indices.biopar.config import biopar_config


class BioparPlanetaryComputer:

    # version='3band',parameter='FAPAR'):
    def __init__(self, biopar_name='CCC', bands=3):
        self.config = biopar_config[f'{bands}band_{biopar_name}']

    def get_angles(self, da, dict_angles):
        for out_name in dict_angles:
            input_angle = dict_angles[out_name]

            if len(input_angle) == 1:
                da_angle = da.coords[input_angle[0]].astype(np.float32)
                da_angle = np.cos(np.radians(da_angle))

            elif len(input_angle) == 2:
                da_angle1 = da.coords[input_angle[0]].astype(np.float32)
                da_angle2 = da.coords[input_angle[1]].astype(np.float32)
                da_angle = np.cos(np.radians(da_angle1 - da_angle2))

            da_angle = da_angle.expand_dims(x=da.x, y=da.y)
            da_angle = da_angle.expand_dims(dim={"band": [out_name]})
            da_angle = da_angle.assign_coords(
                gsd=('band', np.array([np.NaN])))
            da_angle = da_angle.assign_coords(
                title=('band', [out_name]))
            da_angle = da_angle.assign_coords(
                common_name=('band', [out_name]))
            da_angle = da_angle.assign_coords(
                center_wavelength=('band', np.array([np.NaN])))
            da_angle = da_angle.assign_coords(
                full_width_half_max=('band', np.array([np.NaN])))
            da = xr.concat([da, da_angle], dim="band")
        return da

    def run(self, da,
            mean_view_zenith_coord='s2:mean_view_zenith',
            mean_solar_zenith_coord='s2:mean_solar_zenith',
            mean_solar_azimuth_coord='s2:mean_solar_azimuth',
            mean_view_azimuth_coord='s2:mean_view_azimuth',
            output_scale=1):

        dict_angles = dict({
            'cos_VZA': [mean_view_zenith_coord],
            'cos_SZA': [mean_solar_zenith_coord],
            'cos_PSI': [mean_solar_azimuth_coord, mean_view_azimuth_coord]
        })

        da = self.get_angles(da, dict_angles)

        da = da.where(da.coords['band'].isin(self.config['bands']), drop=True)
        da = da.sel(band=self.config['bands'])

        # check if the number of input bands is correct
        if da.shape[1] != len(self.config['bands']):
            raise ValueError(
                f"Expected number of input bands: {len(self.config['bands'])} but received: {da.shape[1]}")

        # check if the number of dimensions is correct
        if len(da.shape) != 4:
            raise ValueError(
                f"Expected number of dimensions: 4 (time, band, y, x) but received: {len(da.shape)}")

        # check if the order of the bands is correct
        if list(da.coords['band'].values) != self.config['bands']:
            raise ValueError(
                f"Expected order of bands: {self.config['bands']} but received: {da.coords['band'].values}")

        da_biopar = xr.apply_ufunc(self._compute_biopar,
                                   da,
                                   kwargs={'output_scale': output_scale},
                                   input_core_dims=[['band', 'y', 'x']],
                                   output_core_dims=[['y', 'x']],
                                   vectorize=True)
        return da_biopar

    def _compute_biopar(self, da, output_scale=1):
        nb_reflectance = da.shape[0] - 3
        invalid_bands = np.any((da[0:nb_reflectance, ...] < 0) | (
            da[0:nb_reflectance, ...] > 1), axis=0)

        inputs = da.transpose()
        x_normalised = (
            inputs * self.config['normalization_scale']) + self.config['normalization_offset']

        layer_1 = np.tanh(
            np.matmul(x_normalised, self.config['l1_weights']) + self.config['l1_bias'])
        layer_2 = np.matmul(
            layer_1, self.config['l2_weights']) + self.config['l2_bias']

        result_float = np.float32(output_scale) * np.clip(
            (layer_2 * self.config['denormalization_scale']
             ) + self.config['denormalization_offset'],
            self.config['output_minmax'][1],
            self.config['output_minmax'][2])

        result_float = np.squeeze(result_float.transpose())
        result_float[invalid_bands] = np.nan

        return result_float


class BioparTerrascopeV200:

    # version='3band',parameter='FAPAR'):
    def __init__(self, biopar_name='CCC', bands=3):
        self.config = biopar_config[f'{bands}band_{biopar_name}']

    def __call__(self, da, output_scale=1):

        da = da.where(da.coords['band'].isin(self.config['bands']), drop=True)
        da = da.sel(band=self.config['bands'])

        # check if the number of input bands is correct
        if da.shape[1] != len(self.config['bands']):
            raise ValueError(
                f"Expected number of input bands: {len(self.config['bands'])} but received: {da.shape[1]}")

        # check if the number of dimensions is correct
        # if len(da.shape) != 4:
        #     raise ValueError(
        #         f"Expected number of dimensions: 4 (time, band, y, x) but received: {len(da.shape)}")

        # check if the order of the bands is correct
        if list(da.coords['band'].values) != self.config['bands']:
            raise ValueError(
                f"Expected order of bands: {self.config['bands']} but received: {da.coords['band'].values}")

        da_biopar = xr.apply_ufunc(self._compute_biopar,
                                   da,
                                   kwargs={'output_scale': output_scale},
                                   input_core_dims=[['band', 'y', 'x']],
                                   output_core_dims=[['y', 'x']],
                                   vectorize=True)
        return da_biopar

    def _compute_biopar(self, da, output_scale=1):
        nb_reflectance = da.shape[0] - 3
        invalid_bands = np.any((da[0:nb_reflectance, ...] < 0) | (
            da[0:nb_reflectance, ...] > 1), axis=0)

        inputs = da.transpose()
        x_normalised = (
            inputs * self.config['normalization_scale']) + self.config['normalization_offset']

        layer_1 = np.tanh(
            np.matmul(x_normalised, self.config['l1_weights']) + self.config['l1_bias'])
        layer_2 = np.matmul(
            layer_1, self.config['l2_weights']) + self.config['l2_bias']

        result_float = np.float32(output_scale) * np.clip(
            (layer_2 * self.config['denormalization_scale']
             ) + self.config['denormalization_offset'],
            self.config['output_minmax'][1],
            self.config['output_minmax'][2])

        result_float = np.squeeze(result_float.transpose())
        result_float[invalid_bands] = np.nan

        return result_float.astype(np.float32)


def biopar(da, biopar_name='CCC', bands=3,
           output_scale=1, platform='Terrascope'):
    if platform == 'PlanetaryComputer':
        Biopar = BioparPlanetaryComputer
    elif platform == 'Terrascope':
        Biopar = BioparTerrascopeV200

    biopar_processor = Biopar(biopar_name=biopar_name,
                              bands=bands)
    da_biopar = biopar_processor(da, output_scale=output_scale)
    return da_biopar
