from simphony.component.pcell import PCell
from simphony.component.port import Port
from simphony.libraries.ideal.modulators import (
    DirectedOpticalModulator,
    OpticalModulator,
)
from simphony.libraries.ideal.s_parameters import optical_s_parameter_placeholder
from simphony.libraries.old_ideal import coupler, waveguide
from simphony.simulation.simulation import SimulationParameters


class MZI(PCell):
    r"""
    o2 ---\        /---[ϕ]---\        /--- o3
           --------           --------
           --------           --------
    o0 ---/        \---[ϕ]---/        \--- o1

    or if `partial` is true:

    o2 ---[ϕ]---\        /--- o3
                 --------
                 --------
    o0 ---[ϕ]---/        \--- o1

    """
    _arms = (
        ("bot", "o1", "o0"),
        ("top", "o3", "o2"),
    )

    ports = [
        Port(name="o0", type="optical", directionality="bidirectional"),
        Port(name="o1", type="optical", directionality="bidirectional"),
        Port(name="o2", type="optical", directionality="bidirectional"),
        Port(name="o3", type="optical", directionality="bidirectional"),
        Port(
            name="e0",
            type="electrical",
            directionality="input",
        ),
        Port(
            name="e1",
            type="electrical",
            directionality="input",
        ),
    ]

    @classmethod
    def _arm_connections(cls, partial: bool, modulators: bool):
        connections = {}

        for arm_name, splitter_port, combiner_port in cls._arms:
            waveguide_name = f"{arm_name}_wg"
            modulator_name = f"{arm_name}_mod"
            combiner = f"combiner,{combiner_port}"

            if partial:
                if modulators:
                    connections[f"{waveguide_name},o1"] = f"{modulator_name},o0"
                    connections[f"{modulator_name},o1"] = combiner
                else:
                    connections[f"{waveguide_name},o1"] = combiner
            else:
                if modulators:
                    connections[f"splitter,{splitter_port}"] = f"{modulator_name},o0"
                    connections[f"{modulator_name},o1"] = f"{waveguide_name},o0"
                else:
                    connections[f"splitter,{splitter_port}"] = f"{waveguide_name},o0"

                connections[f"{waveguide_name},o1"] = combiner

        return connections

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        splitter_settings: dict = None,
        combiner_settings: dict = None,
        top_wg_settings: dict = None,
        bot_wg_settings: dict = None,
        top_phase_shifter_settings: dict = None,
        bot_phase_shifter_settings: dict = None,
        modulators: bool = True,
        partial: bool = False,
        # TODO: maybe inherit a getter or setter or just don't do group ids
        group_id="default",  # Setting to None will disable grouping, not setting will use MZI class id
    ):
        """`partial` builds only the arms and combiner, without the input
        splitter.

        `modulators` controls whether each arm includes a tunable phase
        shifter.
        """
        if splitter_settings is None:
            splitter_settings = {}
        if combiner_settings is None:
            combiner_settings = {}
        if top_wg_settings is None:
            top_wg_settings = {"length": 50.0}
        if bot_wg_settings is None:
            bot_wg_settings = {"length": 20.0}
        if top_phase_shifter_settings is None:
            top_phase_shifter_settings = {}
        if bot_phase_shifter_settings is None:
            bot_phase_shifter_settings = {}

        self.netlist = {
            "instances": {
                "combiner": "coupler",
                "top_wg": "waveguide",
                "bot_wg": "waveguide",
            },
            "ports": {
                "o1": "combiner,o1",
                "o3": "combiner,o3",
            },
            "connections": self._arm_connections(partial, modulators),
        }
        self.settings = {
            "top_wg": {
                "sax_settings": top_wg_settings,
            },
            "bot_wg": {
                "sax_settings": bot_wg_settings,
            },
            "combiner": {
                "sax_settings": combiner_settings,
            },
        }

        if modulators:
            self.netlist["instances"].update(
                {
                    "top_mod": "modulator",
                    "bot_mod": "modulator",
                }
            )
            self.netlist["ports"]["e0"] = "top_mod,e0"
            self.netlist["ports"]["e1"] = "bot_mod,e0"
            self.settings["top_mod"] = top_phase_shifter_settings
            self.settings["bot_mod"] = bot_phase_shifter_settings

        if partial:
            self.netlist["ports"]["o0"] = "bot_wg,o0"
            self.netlist["ports"]["o2"] = "top_wg,o0"
        else:
            self.netlist["instances"]["splitter"] = "coupler"
            self.netlist["ports"]["o0"] = "splitter,o0"
            self.netlist["ports"]["o2"] = "splitter,o2"
            self.settings["splitter"] = {"sax_settings": splitter_settings}

        self.models = {
            "coupler": coupler,
            "waveguide": waveguide,
        }
        if modulators:
            if simulation_parameters.directed:
                self.models["modulator"] = DirectedOpticalModulator
            else:
                self.models["modulator"] = OpticalModulator

        from simphony.simulation.simulation import SimulationMode

        if simulation_parameters.simulation_mode == SimulationMode.SAMPLE_MODE:
            pass
        elif simulation_parameters.simulation_mode == SimulationMode.BLOCK_MODE:
            coupler_directionality = {
                "o0": "input",
                "o2": "input",
                "o1": "output",
                "o3": "output",
            }

            waveguide_directionality = {
                "o0": "input",
                "o1": "output",
            }

            self.models["coupler"] = optical_s_parameter_placeholder(
                coupler, coupler_directionality, simulation_parameters.mode_identifiers
            )
            self.models["waveguide"] = optical_s_parameter_placeholder(
                waveguide,
                waveguide_directionality,
                simulation_parameters.mode_identifiers,
            )

            # self.settings['top_wg'] = {"sax_settings": self.settings['top_wg']}
            # self.settings['bot_wg'] = {"sax_settings": self.settings['bot_wg']}
            # if not partial:
            #     self.settings['splitter'] = {"sax_settings": self.settings['splitter']}

            # self.settings['combiner'] = {"sax_settings": self.settings['combiner']}

        elif simulation_parameters.simulation_mode == SimulationMode.S_PARAMETER:
            pass
        else:
            raise ValueError(
                f"{self} has does not support the simulation type {simulation_parameters.simulation_mode}"
            )

        s_parameter_models_to_group = ["top_wg", "bot_wg", "splitter", "combiner"]

        if partial:
            s_parameter_models_to_group.remove("splitter")

        if group_id == "default":
            group_id = id(MZI)

        for instance_name in s_parameter_models_to_group:
            self.settings[instance_name]["group_id"] = group_id


def mzi_lattice_filter(
    order: int = 3,
    modulators: bool = True,
):
    """Create a PCell class for a bidirectional MZI lattice filter.

    The returned class expands into `order` cascaded `MZI` stages. The first
    stage is a full MZI with an input splitter; later stages are partial MZIs
    that connect the two optical paths from one stage into the next.

    Parameters
    ----------
    order:
        Number of MZI stages in the lattice.
    modulators:
        If true, include phase modulators in each MZI stage and expose two
        electrical ports per stage.

    Returns
    -------
    type[PCell]
        A parameterized component class exposing optical ports `o0`, `o1`,
        `o2`, and `o3`, plus two electrical phase-shifter ports per stage when
        `modulators` is true.
    """
    optical_ports = [
        Port(name="o0", type="optical", directionality="bidirectional"),
        Port(name="o1", type="optical", directionality="bidirectional"),
        Port(name="o2", type="optical", directionality="bidirectional"),
        Port(name="o3", type="optical", directionality="bidirectional"),
    ]

    electrical_ports = (
        [
            Port(name=f"mzi{i}_e{j}", type="electrical", directionality="input")
            for i in range(order)
            for j in (0, 1)
        ]
        if modulators
        else []
    )

    class MZILatticeFilter(PCell):
        ports = optical_ports + electrical_ports

        def __init__(
            self,
            simulation_parameters: SimulationParameters,
            *,
            delay_lengths: dict = None,
            coupling_coeffs: list = None,
            ### TODO: Add a way to adjust the phase modulator settings
            ### TODO: Implement method for import standard presets
            # preset: str = None,
        ):
            """"""
            instances = {}
            connections = {}
            ports = {}
            models = {}

            MZI_MODEL_NAME = "mzi"
            MZI_INSTANCE_NAME_BASE = MZI_MODEL_NAME

            def mzi_instance_name(index):
                return f"{MZI_INSTANCE_NAME_BASE}{index}"

            models["mzi"] = MZI
            for mzi_index in range(order):
                instances[mzi_instance_name(mzi_index)] = {
                    "component": MZI_MODEL_NAME,
                    # "settings": {},
                }

            connections = {
                f"{mzi_instance_name(i)},{src}": f"{mzi_instance_name(i+1)},{dst}"
                for i in range(order - 1)
                for src, dst in (("o1", "o0"), ("o3", "o2"))
            }

            # connections = {
            #     f"{mzi_instance_name(i)},o1": f"{mzi_instance_name(i+1)},o0",
            #     f"{mzi_instance_name(i)},o3": f"{mzi_instance_name(i+1)},o2",
            #     for i in range(order-1)
            # }

            # ports["o0"] = f"{mzi_instance_name(0)},o0"
            # ports["o1"] = f"{mzi_instance_name(order-1)},o1"

            # ports = {
            #     "o0": f"{mzi_instance_name(0)},o0",
            #     "o1": f"{mzi_instance_name(order-1)},o1",
            #     "o2": f"{mzi_instance_name(0)},o2",
            #     "o3": f"{mzi_instance_name(order-1)},o3",
            # }
            ports = {}
            for port in self.ports:
                if port.type == "electrical":
                    instance_name, port_name = port.name.split("_", maxsplit=1)
                    ports[port.name] = f"{instance_name},{port_name}"
            ports["o0"] = f"{mzi_instance_name(0)},o0"
            ports["o1"] = f"{mzi_instance_name(order-1)},o1"
            ports["o2"] = f"{mzi_instance_name(0)},o2"
            ports["o3"] = f"{mzi_instance_name(order-1)},o3"

            self.netlist = {
                "instances": instances,
                "ports": ports,
                "connections": connections,
            }

            self.models = models

            self.settings = {
                instance_name: {
                    "partial": True,
                    "modulators": modulators,
                }
                for instance_name in instances.keys()
            }
            self.settings[mzi_instance_name(0)]["partial"] = False

            # self.instantiated_netlist = instantiate_netlist(netlist, models, settings={})

            # Grouping should happen when the instantiated_netlist is returned
            # Just Overwrite the getter for instantiated_netlist
            # for mzi_index in range(order):
            #     group_name = f"{MZILatticeFilter}_{id(self)}_group_{mzi_index}"
            #     instances = []
            #     netlist = group_instances(netlist, )

    return MZILatticeFilter


def mzi_lattice_passband(
    order: int = 4,
    modulators: bool = False,
):
    """Create a PCell class for a one-input MZI lattice passband filter.

    The returned class builds a cascade of `MZI` stages with one primary input,
    each stage's unused branch output, and the final cascade output exposed as
    optical top-level ports. Each stage uses a common bottom-arm length and a
    configurable top-arm delay difference.

    Parameters
    ----------
    order:
        Number of MZI stages in the passband cascade.
    modulators:
        If true, include phase modulators in each MZI stage and expose two
        electrical tuning ports per stage.

    Returns
    -------
    type[PCell]
        A parameterized component class exposing input `o0`, branch outputs
        `mzi{i}_o3`, and final output `o1`. Constructor settings include
        `base_length` and `mzi_delay_differences`.
    """
    optical_ports = [
        Port(name="o0", type="optical", directionality="bidirectional"),
        *[
            Port(
                name=f"mzi{i}_o3",
                type="optical",
                directionality="bidirectional",
            )
            for i in range(order)
        ],
    ]

    electrical_ports = (
        [
            Port(name=f"mzi{i}_e{j}", type="electrical", directionality="input")
            for i in range(order)
            for j in (0, 1)
        ]
        if modulators
        else []
    )

    class MZILatticePassband(PCell):
        ports = (
            optical_ports
            + electrical_ports
            + [Port(name="o1", type="optical", directionality="bidirectional")]
        )

        def __init__(
            self,
            simulation_parameters: SimulationParameters,
            *,
            mzi_delay_differences: list = None,
            base_length: float = 10.0,
        ):
            """Build a one-input passband cascade.

            `mzi_delay_differences` are added to `base_length` for the
            top arm of each full MZI stage.
            """

            instances = {}
            connections = {}
            self.settings = {}
            self.models = {"mzi": MZI}
            ports = {}
            if mzi_delay_differences is None:
                mzi_delay_differences = [10.0] * order
            elif len(mzi_delay_differences) != order:
                raise ValueError(
                    "`mzi_delay_differences` must contain one value per MZI stage"
                )

            for i in range(order):
                instances[f"mzi{i}"] = "mzi"
                if i < order - 1:
                    connections[f"mzi{i},o1"] = f"mzi{i+1},o2"
                self.settings[f"mzi{i}"] = {
                    "partial": False,
                    "modulators": modulators,
                    "top_wg_settings": {
                        "length": base_length + mzi_delay_differences[i]
                    },
                    "bot_wg_settings": {"length": base_length},
                }

            ports = {
                "o0": f"mzi{0},o2",
                **{f"mzi{i}_o3": f"mzi{i},o3" for i in range(order)},
            }
            if modulators:
                ports.update(
                    {
                        f"mzi{i}_e{j}": f"mzi{i},e{j}"
                        for i in range(order)
                        for j in (0, 1)
                    }
                )
            ports["o1"] = f"mzi{order - 1},o1"
            self.netlist = {
                "instances": instances,
                "ports": ports,
                "connections": connections,
            }

    return MZILatticePassband
