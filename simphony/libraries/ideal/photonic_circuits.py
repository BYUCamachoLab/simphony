from simphony.component.component import SteadyStateComponent
from simphony.component.pcell import PCell
from simphony.simulation.simulation import SimulationParameters
from simphony.component.port import Port

from simphony.libraries.ideal.modulators import OpticalModulator, DirectedOpticalModulator
from simphony.circuit.netlist import instantiate_netlist

### TODO: Find a better way to deal with old sax libraries
from simphony.libraries.old_ideal import waveguide, coupler
from simphony.libraries.ideal.s_parameters import optical_s_parameter

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
        Port(
            name="e0",
            type="electrical",
            directionality = "input"
        ),
        Port(
            name="e1",
            type="electrical",
            directionality = "input"
        ),
    ]
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
    ):
        """
        `partial` boolean value which decides 
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

        if not partial:
            self.netlist = {
                "instances": {
                    "splitter": "coupler",
                    "combiner": "coupler",
                    "top_wg": "waveguide",
                    "bot_wg": "waveguide",
                    "top_mod": "modulator",
                    "bot_mod": "modulator",
                },
                "connections": {
                    "splitter,o1": "bot_mod,o0",
                    "bot_mod,o1": "bot_wg,o0",
                    "bot_wg,o1": "combiner,o0",
                    "splitter,o3": "top_mod,o0",
                    "top_mod,o1": "top_wg,o0",
                    "top_wg,o1": "combiner,o2",
                },
                "ports": {
                    "o0": "splitter,o0",
                    "o1": "combiner,o1",
                    "o2": "splitter,o2",
                    "o3": "combiner,o3",
                    "e0": "top_mod,e0",
                    "e1": "bot_mod,e0",
                },
            }
        else:
            self.netlist = {
                "instances": {
                    # "splitter": "coupler",
                    "combiner": "coupler",
                    "top_wg": "waveguide",
                    "bot_wg": "waveguide",
                    "top_mod": "modulator",
                    "bot_mod": "modulator",
                },
                "connections": {
                    # "splitter,o1": "bot_mod,o0",
                    "bot_mod,o1": "bot_wg,o0",
                    "bot_wg,o1": "combiner,o0",
                    # "splitter,o3": "top_mod,o0",
                    "top_mod,o1": "top_wg,o0",
                    "top_wg,o1": "combiner,o2",
                },
                "ports": {
                    "o0": "bot_wg,o0",
                    "o1": "combiner,o1",
                    "o2": "top_wg,o0",
                    "o3": "combiner,o3",
                    "e0": "top_mod,e0",
                    "e1": "bot_mod,e0",
                },
            }

        self.settings = {
            "top_wg": top_wg_settings,
            "bot_wg": bot_wg_settings,
            "splitter": splitter_settings,
            "combiner": combiner_settings,
            "top_mod": top_phase_shifter_settings,
            "bot_mod": bot_phase_shifter_settings,
        }
        from simphony.simulation.simulation import SimulationMode
        if simulation_parameters.simulation_mode == SimulationMode.SAMPLE_MODE:
            self.models = {
                "coupler": coupler,
                "waveguide": waveguide,
                "modulator": OpticalModulator,
            }
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
                "modulator": DirectedOpticalModulator,
            }
            self.settings['top_wg'] = {"sax_settings": self.settings['top_wg']}
            self.settings['bot_wg'] = {"sax_settings": self.settings['bot_wg']}
            self.settings['splitter'] = {"sax_settings": self.settings['splitter']}
            self.settings['combiner'] = {"sax_settings": self.settings['combiner']}

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




# class VoltageFollower(SteadyStateComponent):
#     electrical_ports = ["e0", "e1"]

# class OpAmp(SteadyStateComponent):
#     electrical_ports = ["ninv","inv","vp","vn","vout"]
    
