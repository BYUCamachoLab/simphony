from simphony.component.pcell import PCell
from simphony.simulation.simulation import SimulationParameters
from simphony.component.port import Port

from simphony.libraries.ideal.modulators import DirectedOpticalModulator

from simphony.simulation.block_mode import BlockModeSimulationParameters 
from simphony.libraries.old_ideal import waveguide, coupler
from simphony.libraries.ideal.s_parameters import optical_s_parameter
from inspect import isfunction
from simphony.libraries.ideal.special import Terminator
import numpy as np
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
        Port(
            name="o0",
            type="optical",
            directionality = "bidirectional"
        ),
        Port(
            name="o1",
            type="optical",
            directionality = "bidirectional"
        ),
        Port(
            name="o2",
            type="optical",
            directionality = "bidirectional"
        ),
        Port(
            name="o3",
            type="optical",
            directionality = "bidirectional"
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
        partial: bool = False,
        # TODO: maybe inherit a getter or setter or just don't do group ids
        group_id = "default", # Setting to None will disable grouping, not setting will use MZI class id
        modulators: bool = True,

    ):
        """
        `partial` builds only the arms and combiner, without the input splitter.
        `modulators` controls whether each arm includes a tunable phase shifter.
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
            "top_wg": top_wg_settings,
            "bot_wg": bot_wg_settings,
            "combiner": combiner_settings,
        }

        if modulators:
            self.netlist["instances"].update({
                "top_mod": "modulator",
                "bot_mod": "modulator",
            })
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
            self.settings["splitter"] = splitter_settings

        self.models = {
            "coupler": coupler,
            "waveguide": waveguide,
        }
        if modulators:
            self.models["modulator"] = DirectedOpticalModulator
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

            self.models = {
                "coupler": optical_s_parameter(coupler, coupler_directionality, simulation_parameters.mode_identifiers),
                "waveguide": optical_s_parameter(waveguide, waveguide_directionality, simulation_parameters.mode_identifiers),
            }
            if modulators:
                self.models["modulator"] = DirectedOpticalModulator
            self.settings['top_wg'] = {"sax_settings": self.settings['top_wg']}
            self.settings['bot_wg'] = {"sax_settings": self.settings['bot_wg']}
            if not partial:
                self.settings['splitter'] = {"sax_settings": self.settings['splitter']}
            if not partial:
                self.settings['splitter'] = {"sax_settings": self.settings['splitter']}
            self.settings['combiner'] = {"sax_settings": self.settings['combiner']}

        elif simulation_parameters.simulation_mode == SimulationMode.S_PARAMETER:
            pass
        else:
            raise ValueError(f"{self} has does not support the simulation type {simulation_parameters.simulation_mode}")
        

        s_parameter_models_to_group = ["top_wg", "bot_wg", "splitter", "combiner"]
        if partial:
            del self.settings["splitter"]
            s_parameter_models_to_group.remove("splitter")

        if group_id is "default":
            group_id = id(MZI)
        
        for instance_name in s_parameter_models_to_group:
            self.settings[instance_name]["group_id"] = group_id


def mzi_lattice_filter(
    order: int = 3,
    # expose_modulators: bool = True,      
):
    class MZILatticeFilter(PCell):
        ports = [
            Port(
                name="o0",
                type="optical",
                directionality = "bidirectional"
            ),
            Port(
                name="o1",
                type="optical",
                directionality = "bidirectional"
            )
        ] + [
            Port(
                name=f"mzi{i}_e{j}",
                type="electrical",
                directionality = "input"
            )
            for i in range(order)
            for j in (0, 1)
        ]

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
            """
            """
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

            ports = {
                "o0": f"{mzi_instance_name(0)},o0",
                "o1": f"{mzi_instance_name(order-1)},o1",
                "o2": f"{mzi_instance_name(0)},o2",
                "o3": f"{mzi_instance_name(order-1)},o3",
            }

            self.netlist = {
                "instances": instances,
                "ports": ports,
                "connections": connections,
            }

            self.models = models

            self.settings = {instance_name: {"partial": True} for instance_name in instances.keys()}
            self.settings[mzi_instance_name(0)]["partial"] = False

            pass
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
    optical_ports = [
        Port(name="o0", type="optical", directionality="bidirectional"),
        Port(name="o1", type="optical", directionality="bidirectional"),
    ]

    electrical_ports = [
        Port(name=f"mzi{i}_e{j}", type="electrical", directionality="input")
        for i in range(order)
        for j in (0, 1)
    ] if modulators else []

    class MZILatticePassband(PCell):
        ports = optical_ports + electrical_ports
        
        def __init__(
            self,
            simulation_parameters: SimulationParameters,
            *,
            mzi_delay_differences:list = None,
            base_length: float = 10.0 
        ):
            """Build a one-input passband cascade.

            `mzi_delay_differences` are added to `base_length` for the top arm
            of each full MZI stage.
            """

            instances = {}
            connections = {}
            self.settings = {}
            self.models = {"mzi": MZI, "terminator": Terminator}
            ports = {}
            if mzi_delay_differences is None:
                mzi_delay_differences = []
                for i in range(order):
                    mzi_delay_differences.append(10.0)
            
            for i in range(order):
                instances[f"mzi{i}"] = "mzi"
                instances[f"upt{i}1"] = "terminator"
                instances[f"upt{i}2"] = "terminator"
                if i < order-1:
                    connections[f"mzi{i},o1"] = f"mzi{i+1},o2"
                connections[f"mzi{i},o3"] = f"upt{i}1,o0"
                connections[f"mzi{i},o0"] = f"upt{i}2,o0"
                self.settings[f"mzi{i}"] = {
                    "partial": False,
                    "modulators": modulators,
                    "top_wg_settings":{"length":base_length + mzi_delay_differences[i]},
                    "bot_wg_settings":{"length": base_length}
                }
                
            ports = {
                "o0": f"mzi{0},o2",
                "o1": f"mzi{order-1},o1",
            }
            if modulators:
                ports.update({
                    f"mzi{i}_e{j}": f"mzi{i},e{j}"
                    for i in range(order)
                    for j in (0, 1)
                })
            self.netlist = {
                "instances": instances,
                "ports": ports,
                "connections": connections,
            }
            
    return MZILatticePassband
