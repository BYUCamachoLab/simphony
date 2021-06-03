# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

from simphony import Model
from SiPANN import scee

class SiPANN_Base(Model):
    """Class that wraps SCEE models for use in simphony.

    Model passed into class CANNOT have varying geometries, as a device such as this
    can't be cascaded properly.

    Parameters
    -----------
    model : DC
        Chosen compact model from ``SiPANN.scee`` module. Can be any model that inherits from
        the DC abstract class
    sigmas : dict, optional
        Dictionary mapping parameters to sigma values for use in monte_carlo simulations. Note sigmas should
        be in values of nm. Defaults to an empty dictionary.
    """

    pins = ("n1", "n2", "n3", "n4")  #: The default pin names of the device
    freq_range = (
        182800279268292.0,
        205337300000000.0,
    )  #: The valid frequency range for this model.

    def __init__(self, model, sigmas=dict()):
        super().__init__()

        self.model = model
        self.sigmas = sigmas

        # save actual parameters for switching back from monte_carlo
        self.og_params = self.model.__dict__.copy()
        self.rand_params = dict()

        # make sure there's no varying geometries
        args = self.model._clean_args(None)
        if len(args[0]) != 1:
            raise ValueError(
                "You have changing geometries, use in simphony doesn't make sense!"
            )

        self.regenerate_monte_carlo_parameters()

    def s_parameters(self, freq):
        """Get the s-parameters of SCEE Model.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        # convert wavelength to frequency
        wl = freq2wl(freq) * 1e9

        return self.model.sparams(wl)

    def monte_carlo_s_parameters(self, freq):
        """Get the s-parameters of SCEE Model with slightly changed parameters.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        # perturb params and get sparams
        self.model.update(**self.rand_params)
        sparams = self.model.sparams(wl)

        # restore parameters to originals
        self.model.update(**self.og_params)

        return sparams

    def regenerate_monte_carlo_parameters(self):
        """Varies parameters based on passed in sigma dictionary.

        Iterates through sigma dictionary to change each of those
        parameters, with the mean being the original values found in
        model.
        """
        # iterate through all params that should be tweaked
        for param, sigma in self.sigmas.items():
            self.rand_params[param] = np.random.normal(self.og_params[param], sigma)


# TODO
class BidirectionalCoupler(SiPANN_Base):
    pass


class DirectionalCoupler(SiPANN_Base):
    pass


class GratingCoupler(SiPANN_Base):
    pass


class HalfRing(SiPANN_Base):
    def __init__(self, width, thickness, radius, gap, sw_angle=90, sigmas=dict()):
        """This class will create half of a ring resonator.

        It takes in a radius and gap along with usual waveguide parameters. Ports are numbered as::

            |         2 \     / 4          |
            |            \   /             |
            |             ---              |
            |         1---------3          |

        Parameters
        ----------
            width : float or ndarray
                Width of the waveguide in nm (Valid for 400nm-600nm)
            thickness : float or ndarray
                Thickness of waveguide in nm (Valid for 180nm-240nm)
            radius : float or ndarray
                Distance from center of ring to middle of waveguide in nm.
            gap : float or ndarray
                Minimum distance from ring waveguide edge to straight waveguide edge in nm. (Must be > 100nm)
            sw_angle : float or ndarray, optional
                Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
        """
        super().__init__(scee.HalfRing(width, thickness, radius, gap, sw_angle), sigmas)

class Terminator(SiPANN_Base):
    pass


class Waveguide(SiPANN_Base):
    pass


class YBranch(SiPANN_Base):
    pass
