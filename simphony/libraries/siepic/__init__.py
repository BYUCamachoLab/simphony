# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
"""This package contains parameterized models of PIC components from the SiEPIC
Electron Beam Lithography Process Development Kit (PDK) from the University of
British Columbia (UBC), which is licensed under the terms of the MIT License.

See their repository for more details (
https://github.com/SiEPIC/SiEPIC_EBeam_PDK).

Usage:

.. code-block:: python

    from simphony.libraries import siepic

    wg = siepic.waveguide()
"""

from .models import grating_coupler, waveguide, y_branch

# class GratingCoupler(SiEPIC_PDK_Base):

#     def __init__(self):

#         if _has_gf:
#             gf.clear_cache()
#             self.component = gf.components.grating_coupler_te()
#             self.component.name = self.name
#             pin_names = [pin.name for pin in self.pins]
#             for i, port in enumerate(self.component.ports.values()):
#                 port.name = pin_names[i]
#             self.component.ports = dict(zip(pin_names, self.component.ports.values()))
#             port1 = self.component.ports["pin1"]
#             self.component.ports["pin1"] = self.component.ports["pin2"]
#             self.component.ports["pin2"] = port1

#     def on_args_changed(self):
#         try:
#             if self.layout_aware:
#                 self.suspend_autoupdate()

#                 self._update_lay_aware_true()

#                 self.freq_range = (self._f[0], self._f[-1])

#                 self.enable_autoupdate()
#         except AttributeError:
#             self.suspend_autoupdate()

#             self._update_no_lay_aware()

#             self.enable_autoupdate()

#     def _update_no_lay_aware(self):
#         available = self._source_argsets()
#         normalized = []
#         for d in available:
#             polarization, thickness, deltaw = (
#                 (key, d.get(key)) for key in self._args_keys
#             )
#             normalized.append(
#                 {
#                     polarization[0]: polarization[1],
#                     thickness[0]: round(str2float(thickness[1]) * 1e-9, 15),
#                     deltaw[0]: round(str2float(deltaw[1]) * 1e-9, 15),
#                 }
#             )
#         idx = self._get_matched_args(normalized, self.args)
#         for key, value in normalized[idx].items():
#             setattr(self, key, value)

#         valid_args = available[idx]
#         params = np.genfromtxt(self._get_file(valid_args), delimiter="\t")
#         self._f = params[:, 0]
#         self._s = np.zeros((len(self._f), 2, 2), dtype="complex128")
#         self._s[:, 0, 0] = params[:, 1] * np.exp(1j * params[:, 2])
#         self._s[:, 0, 1] = params[:, 3] * np.exp(1j * params[:, 4])
#         self._s[:, 1, 0] = params[:, 5] * np.exp(1j * params[:, 6])
#         self._s[:, 1, 1] = params[:, 7] * np.exp(1j * params[:, 8])

#         # Arrays are from high frequency to low; reverse it,
#         # for convention's sake.
#         self._f = self._f[::-1]
#         self._s = self._s[::-1]
#         self.freq_range = (self._f[0], self._f[-1])

#     def _update_lay_aware_true(self):
#         pass
#         # available = self._source_argsets()
#         # thicknesses = []
#         # deltaws = []
#         # s_params = []
#         # for d in available:
#         #     _, thickness, deltaw = [(key, d.get(key)) for key in self._args_keys]
#         #     thicknesses.append(round(str2float(thickness[1]) * 1e-9, 15))
#         #     deltaws.append(round(str2float(deltaw[1]) * 1e-9, 15))

#         # if self.polarization == "TE":
#         #     for idx in range(round(len(thicknesses) / 2)):
#         #         valid_args = available[idx]
#         #         params = np.genfromtxt(self._get_file(valid_args), delimiter="\t")
#         #         self._f = params[:, 0]
#         #         s = np.zeros((len(self._f), 2, 2), dtype="complex128")
#         #         s[:, 0, 0] = params[:, 1] * np.exp(1j * params[:, 2])
#         #         s[:, 0, 1] = params[:, 3] * np.exp(1j * params[:, 4])
#         #         s[:, 1, 0] = params[:, 5] * np.exp(1j * params[:, 6])
#         #         s[:, 1, 1] = params[:, 7] * np.exp(1j * params[:, 8])

#         #         self._f = self._f[::-1]
#         #         s = s[::-1]
#         #         s_params.append(s)

#         #     s_params = np.asarray(s_params, dtype=object)
#         #     thicknesses = np.asarray(thicknesses, dtype=float)
#         #     deltaws = np.asarray(deltaws, dtype=float)

#         #     dim = len(s_params)
#         #     for dimidx in range(dim):
#         #         print(f'[s_params[dimidx][:][:][:])
#         #         s_list.append(s_params[dimidx][:][:][:])
#         #     s_list = np.asarray(s_list, dtype=complex)
#         #     self._s = interp.griddata(
#         #         (
#         #             thicknesses[0 : round(len(thicknesses) / 2)],
#         #             deltaws[0 : round(len(deltaws) / 2)],
#         #         ),
#         #         s_list,
#         #         (self.thickness, self.deltaw),
#         #         method="cubic",
#         #     )

#         # elif self.polarization == "TM":
#         #     for idx in range(round(len(thicknesses) / 2) + 1, len(thicknesses)):
#         #         valid_args = available[idx]
#         #         params = np.genfromtxt(self._get_file(valid_args), delimiter="\t")
#         #         self._f = params[:, 0]
#         #         s = np.zeros((len(self._f), 2, 2), dtype="complex128")
#         #         s[:, 0, 0] = params[:, 1] * np.exp(1j * params[:, 2])
#         #         s[:, 0, 1] = params[:, 3] * np.exp(1j * params[:, 4])
#         #         s[:, 1, 0] = params[:, 5] * np.exp(1j * params[:, 6])
#         #         s[:, 1, 1] = params[:, 7] * np.exp(1j * params[:, 8])

#         #         self._f = self._f[::-1]
#         #         s = s[::-1]
#         #         s_params.append(s)

#         #     s_params = np.asarray(s_params, dtype=object)
#         #     thicknesses = np.asarray(thicknesses, dtype=float)
#         #     deltaws = np.asarray(deltaws, dtype=float)

#         #     dim = len(s_params)
#         #     s_list = []
#         #     for dimidx in range(dim):
#         #         s_list.append(s_params[dimidx][:][:][:])
#         #     s_list = np.asarray(s_list, dtype=complex)
#         #     self._s = interp.griddata(
#         #         (
#         #             thicknesses[round(len(thicknesses) / 2) + 1, len(thicknesses)],
#         #             deltaws[round(len(thicknesses) / 2) + 1, len(thicknesses)],
#         #         ),
#         #         s_list,
#         #         (self.thickness, self.deltaw),
#         #         method="cubic",
#         #     )

#     def update_variations(self, **kwargs):
#         self.nominal_deltaw = self.deltaw
#         self.nominal_thickness = self.thickness

#         w = self.deltaw + kwargs.get("corr_w") * 1e-9
#         t = self.thickness + kwargs.get("corr_t") * 1e-9

#         self.layout_aware = True
#         self.deltaw = w
#         self.thickness = t

#     def regenerate_layout_aware_monte_carlo_parameters(self):
#         self.thickness = self.nominal_thickness
#         self.deltaw = self.nominal_deltaw


# class Waveguide(SiEPIC_PDK_Base):

#     def __init__(
#         sigma_ne=0.05,
#         sigma_ng=0.05,
#         sigma_nd=0.0001,
#         **kwargs,
#     ):

#         if _has_gf:
#             self.path: Route = None

#         self.regenerate_monte_carlo_parameters()

#     def on_args_changed(self):
#         try:
#             if self.layout_aware:
#                 self.suspend_autoupdate()

#                 available = self._source_argsets()

#                 widths = []
#                 heights = []
#                 for d in available:
#                     widths.append(d["width"])
#                     heights.append(d["height"])

#                 lam0_all = []
#                 ne_all = []
#                 ng_all = []
#                 nd_all = []
#                 for idx in range(len(available)):
#                     valid_args = available[idx]
#                     with open(self._get_file(valid_args)) as f:
#                         params = f.read().rstrip("\n")
#                     if self.polarization == "TE":
#                         lam0, ne, _, ng, _, nd, _ = params.split(" ")
#                     elif self.polarization == "TM":
#                         lam0, _, ne, _, ng, _, nd = params.split(" ")
#                         raise NotImplementedError

#                     lam0_all.append(lam0)
#                     ne_all.append(ne)
#                     ng_all.append(ng)
#                     nd_all.append(nd)

#                 widths = np.asarray(widths).astype(float)
#                 heights = np.asarray(heights).astype(float)
#                 lam0_all = np.asarray(lam0_all).astype(float)
#                 ne_all = np.asarray(ne_all).astype(float)
#                 ng_all = np.asarray(ng_all).astype(float)
#                 nd_all = np.asarray(nd_all).astype(float)

#                 self.lam0 = interp.griddata(
#                     (widths, heights),
#                     lam0_all,
#                     (self.width * 1e9, self.height * 1e9),
#                     method="cubic",
#                 )
#                 self.ne = interp.griddata(
#                     (widths, heights),
#                     ne_all,
#                     (self.width * 1e9, self.height * 1e9),
#                     method="cubic",
#                 )
#                 self.ng = interp.griddata(
#                     (widths, heights),
#                     ng_all,
#                     (self.width * 1e9, self.height * 1e9),
#                     method="cubic",
#                 )
#                 self.nd = interp.griddata(
#                     (widths, heights),
#                     nd_all,
#                     (self.width * 1e9, self.height * 1e9),
#                     method="cubic",
#                 )

#                 self.enable_autoupdate()

#         except AttributeError:
#             self.suspend_autoupdate()

#             available = self._source_argsets()
#             normalized = [
#                 {k: round(str2float(v) * 1e-9, 21) for k, v in d.items()}
#                 for d in available
#             ]
#             idx = self._get_matched_args(normalized, self.args)

#             valid_args = available[idx]
#             with open(self._get_file(valid_args)) as f:
#                 params = f.read().rstrip("\n")
#             if self.polarization == "TE":
#                 lam0, ne, _, ng, _, nd, _ = params.split(" ")
#             elif self.polarization == "TM":
#                 lam0, _, ne, _, ng, _, nd = params.split(" ")
#                 raise NotImplementedError
#             self.lam0 = float(lam0)
#             self.ne = float(ne)
#             self.ng = float(ng)
#             self.nd = float(nd)

#             # Updates parameters width and thickness to closest match.
#             for key, value in normalized[idx].items():
#                 setattr(self, key, value)

#             self.enable_autoupdate()

#     def s_parameters(self, freqs):
#         """Get the s-parameters of a waveguide.

#         Parameters
#         ----------
#         freqs : float
#             The array of frequencies to get s parameters for.

#         Returns
#         -------
#         (freqs, s) : tuple
#             Returns a tuple containing the frequency array, `freqs`,
#             corresponding to the calculated s-parameter matrix, `s`.
#         """
#         return self.calc_s_params(
#             freqs, self.length, self.lam0, self.ne, self.ng, self.nd
#         )

#     def monte_carlo_s_parameters(self, freqs):
#         """Returns a monte carlo (randomized) set of s-parameters.

#         In this implementation of the monte carlo routine, random values
#         are generated for ne, ng, and nd for each run through of the
#         monte carlo simulation. This means that all waveguide elements
#         throughout a single circuit will have the same (random) ne, ng,
#         and nd values. Hence, there is correlated randomness in the
#         monte carlo parameters but they are consistent within a single
#         circuit.
#         """
#         return self.calc_s_params(
#             freqs, self.length, self.lam0, self.rand_ne, self.rand_ng, self.rand_nd
#         )

#     def layout_aware_monte_carlo_s_parameters(self, freqs):
#         """Returns a monte carlo (randomized) set of s-parameters.

#         In this implementation of the monte carlo routine, values
#         generated for lam0, ne, ng, and nd using the Reduced Spatial
#         Correlation Matrix method are used to return a set of
#         s-parameters. This is repeated for each run of the Monte Carlo
#         analysis for every wavehuide component in the circuit.
#         """
#         return self.calc_s_params(
#             freqs, self.length, self.lam0, self.ne, self.ng, self.nd
#         )

#     def regenerate_monte_carlo_parameters(self):
#         self.rand_ne = np.random.normal(self.ne, self.sigma_ne)
#         self.rand_ng = np.random.normal(self.ng, self.sigma_ng)
#         self.rand_nd = np.random.normal(self.nd, self.sigma_nd)

#     def update_variations(self, **kwargs):
#         self.nominal_width = self.width
#         self.nominal_height = self.height

#         w = self.width + kwargs.get("corr_w") * 1e-9
#         h = self.height + kwargs.get("corr_t") * 1e-9

#         self.layout_aware = True
#         self.width = w
#         self.height = h

#     def regenerate_layout_aware_monte_carlo_parameters(self):
#         self.width = self.nominal_width
#         self.height = self.nominal_height

#     @staticmethod
#     def calc_s_params(freqs, length, lam0, ne, ng, nd):
#         # Initialize array to hold s-params
#         s = np.zeros((len(freqs), 2, 2), dtype=complex)

#         # Loss calculation
#         TE_loss = 700  # dB/m for width 500nm
#         alpha = TE_loss / (20 * np.log10(np.exp(1)))

#         w = np.asarray(freqs) * 2 * np.pi  # get angular freqs from freqs
#         w0 = (2 * np.pi * SPEED_OF_LIGHT) / lam0  # center freqs (angular)

#         # calculation of K
#         K = (
#             2 * np.pi * ne / lam0
#             + (ng / SPEED_OF_LIGHT) * (w - w0)
#             - (nd * lam0**2 / (4 * np.pi * SPEED_OF_LIGHT)) * ((w - w0) ** 2)
#         )

#         for x in range(0, len(freqs)):  # build s-matrix from K and waveguide length
#             s[x, 0, 1] = s[x, 1, 0] = np.exp(-alpha * length + (K[x] * length * 1j))

#         return s


# class YBranch(SiEPIC_PDK_Base):

#     def __init__(self):
#         if _has_gf:
#             gf.clear_cache()
#             self.component = gf.read.import_gds(
#                 os.path.join(os.path.dirname(__file__), "source_data/ebeam_y_1550.gds")
#             )
#             self.component.name = self.name
#             self.component.ports[self.pins[0].name] = gf.Port(
#                 self.pins[0].name,
#                 180,
#                 center=(-7.4, 0),
#                 width=width * 1e6,
#                 layer="PORT",
#                 parent=self.component,
#             )
#             self.component.ports[self.pins[1].name] = gf.Port(
#                 self.pins[1].name,
#                 0,
#                 center=(7.4, 2.75),
#                 width=width * 1e6,
#                 layer="PORT",
#                 parent=self.component,
#             )
#             self.component.ports[self.pins[2].name] = gf.Port(
#                 self.pins[2].name,
#                 0,
#                 center=(7.4, -2.75),
#                 width=width * 1e6,
#                 layer="PORT",
#                 parent=self.component,
#             )

#     def on_args_changed(self):
#         try:
#             if self.layout_aware:
#                 self.suspend_autoupdate()

#                 available = self._source_argsets()
#                 widths = []
#                 heights = []
#                 s_params = []
#                 for idx in range(0, len(available)):
#                     d = available[idx]
#                     widths.append(d["width"])
#                     heights.append(d["thickness"])
#                     valid_args = available[idx]
#                     sparams = parser.read_params(self._get_file(valid_args))
#                     sparams = list(
#                         filter(
#                             lambda sparams: sparams["mode"] == self.polarization,
#                             sparams,
#                         )
#                     )
#                     self._f, s = parser.build_matrix(sparams)

#                     s_params.append(s)

#                 s_params = np.asarray(s_params)

#                 widths = np.asarray(widths, dtype=float)
#                 heights = np.asarray(heights, dtype=float)

#                 dim, _, _, _ = s_params.shape
#                 s_list = []
#                 for dimidx in range(dim):
#                     s_list.append(s_params[dimidx][:][:][:])
#                 s_list = np.asarray(s_list, dtype=complex)
#                 self._s = interp.griddata(
#                     (widths, heights),
#                     s_list,
#                     (self.width * 1e9, self.thickness * 1e9),
#                     method="cubic",
#                 )

#                 self.freq_range = (self._f[0], self._f[-1])

#                 self.enable_autoupdate()

#         except AttributeError:
#             self.suspend_autoupdate()

#             available = self._source_argsets()
#             normalized = [
#                 {k: round(str2float(v) * 1e-9, 21) for k, v in d.items()}
#                 for d in available
#             ]
#             idx = self._get_matched_args(normalized, self.args)

#             valid_args = available[idx]
#             sparams = parser.read_params(self._get_file(valid_args))
#             sparams = list(
#                 filter(lambda sparams: sparams["mode"] == self.polarization, sparams)
#             )

#             for key, value in normalized[idx].items():
#                 setattr(self, key, value)
#             self._f, self._s = parser.build_matrix(sparams)
#             self.freq_range = (self._f[0], self._f[-1])

#             self.enable_autoupdate()

#     def update_variations(self, **kwargs):
#         self.nominal_width = self.width
#         self.nominal_thickness = self.thickness

#         w = self.width + kwargs.get("corr_w") * 1e-9
#         t = self.thickness + kwargs.get("corr_t") * 1e-9

#         self.layout_aware = True
#         self.width = w
#         self.thickness = t

#     def regenerate_layout_aware_monte_carlo_parameters(self):
#         self.width = self.nominal_width
#         self.thickness = self.nominal_thickness
