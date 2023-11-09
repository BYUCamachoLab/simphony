# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
"""SiEPIC models compatible with SAX circuits.

This package contains parameterized models of PIC components from the SiEPIC
Electron Beam Lithography Process Development Kit (PDK) from the University of
British Columbia (UBC), which is licensed under the terms of the MIT License.

See their repository for more details (
https://github.com/SiEPIC/SiEPIC_EBeam_PDK).

Usage:

.. code-block:: python

    from simphony.libraries import siepic

    wg = siepic.waveguide()
"""

from simphony.libraries.siepic.models import (
    bidirectional_coupler,
    directional_coupler,
    grating_coupler,
    half_ring,
    taper,
    terminator,
    waveguide,
    y_branch,
)

__all__ = [
    "bidirectional_coupler",
    "directional_coupler",
    "grating_coupler",
    "half_ring",
    "taper",
    "terminator",
    "waveguide",
    "y_branch",
]

# class GratingCoupler(SiEPIC_PDK_Base):
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

#     def regenerate_monte_carlo_parameters(self):
#         self.rand_ne = np.random.normal(self.ne, self.sigma_ne)
#         self.rand_ng = np.random.normal(self.ng, self.sigma_ng)
#         self.rand_nd = np.random.normal(self.nd, self.sigma_nd)


# class YBranch(SiEPIC_PDK_Base):
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
