######## Importing libraries ########
import copy
from pathlib import Path
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax
from simphony.libraries import ideal
from simphony.plugins.lumerical import load_sparams, df_to_sdict
from simphony.utils import resample, wlum2freq, dict_to_matrix

import pandas as pd
import os

MODULE_DIR = Path(__file__).resolve().parent
SPARAM_DIR = MODULE_DIR / "sparams"

######## Ensuring the directory correct ########
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

load_sparams((SPARAM_DIR / "every gm configuration" / "cc_122p4_S.dat").resolve())



######## Defining layers ########

# Start with the edge couplers
# | | | | | | | |

port_dict = {}
inst_dict = {}
for i in range(8):
    inst_dict[f'coupler{i}'] = 'coupler'
    port_dict[f'in{i}'] = f'coupler{i},o0'
    port_dict[f'out{i}'] = f'coupler{i},o1'


edge = {
    'instances' : inst_dict,
    'connections' : {},
    'ports' : port_dict,
    'placements' : {},   
}

# Layer for routing waveguides to edge couplers
# | | | | | | | |
# \ / \ / / / / /

port_dict = {}
inst_dict = {}
for i in range(8):
    inst_dict[f'ewg{i}'] = 'ewg'
    port_dict[f'in{i}'] = f'ewg{i},o0'
    port_dict[f'out{i}'] = f'ewg{i},o1'
    
edge_routing = {
    'instances' : inst_dict,
    'connections' : {},
    'ports' : port_dict,
    'placements' : {},
}

# Layer for routing waveguides
# | | | | | | | |
# / / \ / \ / \ /

port_dict = {}
inst_dict = {}
for i in range(8):
    inst_dict[f'wg{i}'] = 'wg'
    port_dict[f'in{i}'] = f'wg{i},o0'
    port_dict[f'out{i}'] = f'wg{i},o1'
    
routing = {
    'instances' : inst_dict,
    'connections' : {},
    'ports' : port_dict,
    'placements' : {},
}

# Layer of four 50/50 splitters. 
#  X  X  X  X 

port_dict = {}
inst_dict = {}
for i in [0,2,4,6]:
    inst_dict[f'splitter{i}{i+1}'] = 'splitter'
    port_dict[f'in{i}'] = f'splitter{i}{i+1},o0'
    port_dict[f'in{i+1}'] = f'splitter{i}{i+1},o2'
    port_dict[f'out{i}'] = f'splitter{i}{i+1},o1'
    port_dict[f'out{i+1}'] = f'splitter{i}{i+1},o3'
    
splitter_4 = {
    'instances' : inst_dict,
    'connections' : {},
    'ports' : port_dict,
    'placements' : {},
}

# layer with three crossover couplers, with row 0 and 7 being special waveguides
# | X  X  X |

port_dict = {}
inst_dict = {}

# special waveguides
for i in [0,7]:
    inst_dict[f'xwg{i}'] = 'xwg'
    port_dict[f'in{i}'] = f'xwg{i},o0'
    port_dict[f'out{i}'] = f'xwg{i},o1'

# crossovers
for i in [1,3,5]:
    inst_dict[f'crossover{i}{i+1}'] = 'crossover'
    port_dict[f'in{i}'] = f'crossover{i}{i+1},o0'
    port_dict[f'in{i+1}'] = f'crossover{i}{i+1},o2'
    port_dict[f'out{i}'] = f'crossover{i}{i+1},o1'
    port_dict[f'out{i+1}'] = f'crossover{i}{i+1},o3'
    
crossover_3 = {
    'instances' : inst_dict,
    'connections' : {},
    'ports' : port_dict,
    'placements' : {},
}

# layer with 2 crossover couplers, with the two being in the middle rows
# | | X  X | |

port_dict = {}
inst_dict = {}

# special waveguides
for i in [0,1,6,7]:
    inst_dict[f'xwg{i}'] = 'xwg'
    port_dict[f'in{i}'] = f'xwg{i},o0'
    port_dict[f'out{i}'] = f'xwg{i},o1'
    
# crossovers
for i in [2,4]:
    inst_dict[f'crossover{i}{i+1}'] = 'crossover'
    port_dict[f'in{i}'] = f'crossover{i}{i+1},o0'
    port_dict[f'in{i+1}'] = f'crossover{i}{i+1},o2'
    port_dict[f'out{i}'] = f'crossover{i}{i+1},o1'
    port_dict[f'out{i+1}'] = f'crossover{i}{i+1},o3'
    
crossover_2_middle = {
    'instances' : inst_dict,
    'connections' : {},
    'ports' : port_dict,
    'placements' : {},
}

# Layer with 2 crossover couplers, with the two being split (one on rows 1&2, the other on 5&6)
# | X | | X |

port_dict = {}
inst_dict = {}

# special waveguides
for i in [0,3,4,7]:
    inst_dict[f'xwg{i}'] = 'xwg'
    port_dict[f'in{i}'] = f'xwg{i},o0'
    port_dict[f'out{i}'] = f'xwg{i},o1'
    
# crossovers
for i in [1,5]:
    inst_dict[f'crossover{i}{i+1}'] = 'crossover'
    port_dict[f'in{i}'] = f'crossover{i}{i+1},o0'
    port_dict[f'in{i+1}'] = f'crossover{i}{i+1},o2'
    port_dict[f'out{i}'] = f'crossover{i}{i+1},o1'
    port_dict[f'out{i+1}'] = f'crossover{i}{i+1},o3'
    
crossover_2_split = {
    'instances' : inst_dict,
    'connections' : {},
    'ports' : port_dict,
    'placements' : {},
}

# Try to put it all together correctly

layers = ['edge', 'edge_routing', 'splitter_4', 'routing', 'crossover_3', 
          'routing', 'crossover_2_middle', 'routing', 'crossover_3', 
          'routing', 'splitter_4', 'routing', 'crossover_2_split', 'routing', 
          'splitter_4', 'edge_routing', 'edge']
#reverse the layers
# layers = ['edge', 'edge_routing', 'splitter_4', 'crossover_2_middle', 'splitter_4', 'crossover_3', 'crossover_2_split', 'crossover_3', 'splitter_4' 'edge_routing', 'edge']


port_dict = {}
conn_dict = {}
inst_dict = {}

for i, layer in enumerate(layers):
    inst_dict[f'layer{i}'] = layer
    if i > 0:
        for j in range(8):
            conn_dict[f'layer{i},in{j}'] = f'layer{i-1},out{j}'
            
for i in range(8):
    port_dict[f'in{i}'] = f'layer0,in{i}'
    port_dict[f'out{i}'] = f'layer{len(layers)-1},out{i}'
    
rec_gm = {
    'overall' : {
        'instances' : inst_dict,
        'connections' : conn_dict,
        'ports' : port_dict,
        'placements' : {},
    },
    'edge' : edge,
    'edge_routing' : edge_routing,
    'routing' : routing,
    'splitter_4' : splitter_4,
    'crossover_3' : crossover_3,
    'crossover_2_middle' : crossover_2_middle,
    'crossover_2_split' : crossover_2_split
}


def weave_layers_into_netlist(layer_sequence, layer_definitions):
    """Flatten the recursive layer stack into a single netlist of components."""

    flat_instances = {}
    flat_connections = {}
    flat_ports = {}

    first_layer_inputs = None
    prev_layer_outputs = None

    for idx, layer_name in enumerate(layer_sequence):
        if layer_name not in layer_definitions:
            raise KeyError(f"Layer '{layer_name}' not found in layer_definitions")

        layer = layer_definitions[layer_name]
        layer_instances = layer.get('instances', {})
        layer_connections = layer.get('connections', {})
        layer_ports = layer.get('ports', {})

        prefix = f"{layer_name}_{idx}"
        inst_name_map = {}

        for inst_name, model_name in layer_instances.items():
            new_name = f"{prefix}__{inst_name}"
            flat_instances[new_name] = model_name
            inst_name_map[inst_name] = new_name

        def remap_endpoint(endpoint):
            inst, port = endpoint.split(',')
            return f"{inst_name_map[inst]},{port}"

        for dest, src in layer_connections.items():
            flat_connections[remap_endpoint(dest)] = remap_endpoint(src)

        curr_inputs = {}
        curr_outputs = {}
        other_ports = {}
        for port_name, endpoint in layer_ports.items():
            if port_name.startswith('in') and port_name[2:].isdigit():
                curr_inputs[int(port_name[2:])] = remap_endpoint(endpoint)
            elif port_name.startswith('out') and port_name[3:].isdigit():
                curr_outputs[int(port_name[3:])] = remap_endpoint(endpoint)
            else:
                other_ports[port_name] = remap_endpoint(endpoint)

        if idx == 0:
            first_layer_inputs = curr_inputs.copy()

        if prev_layer_outputs is not None:
            for key, src_endpoint in prev_layer_outputs.items():
                if key not in curr_inputs:
                    raise ValueError(
                        f"Layer '{layer_name}' missing input '{key}' required by previous layer"
                    )
                flat_connections[curr_inputs[key]] = src_endpoint

        prev_layer_outputs = curr_outputs

        for name, endpoint in other_ports.items():
            flat_ports[f"{layer_name}_{idx}_{name}"] = endpoint

    if first_layer_inputs is None or prev_layer_outputs is None:
        raise ValueError("Layer sequence must contain at least one layer with inputs and outputs")

    for idx, endpoint in first_layer_inputs.items():
        flat_ports[f'in{idx}'] = endpoint
    for idx, endpoint in prev_layer_outputs.items():
        flat_ports[f'out{idx}'] = endpoint

    return {
        'instances': flat_instances,
        'connections': flat_connections,
        'ports': flat_ports,
    }
layer_definitions = {
    'edge': edge,
    'edge_routing': edge_routing,
    'routing': routing,
    'splitter_4': splitter_4,
    'crossover_3': crossover_3,
    'crossover_2_middle': crossover_2_middle,
    'crossover_2_split': crossover_2_split,
}


gm_flat_netlist = weave_layers_into_netlist(layers, layer_definitions)


######## Model Instantiation ########


def _resolve_data_path(path_like: Path | str) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (MODULE_DIR / path).resolve()
    return path

#gets the sparam dict from the data folder
def sdict_from_file(filepath: Path = ""):
    resolved_path = _resolve_data_path(filepath)
    _, sdf = load_sparams(str(resolved_path))
    f, sdict = df_to_sdict(sdf)
    def model(wl = 1.55):
        s_new = resample(wlum2freq(wl), f, copy.deepcopy(sdict))
        return s_new
    return model

# Creates a model based on a chosen splitter and crossover
# the crossover waveguide is based on the crossover coupler
def get_model(splitter, crossover, thickness = None, gap = None, angle = "", folder = "./sparams/varying width", taperChoice = 0, MA = None):
    MAstring = "" if MA == None else f"MA_{MA}"
    defaultTaperThicc = "240"
    defaultTaperGap = "0.034"
    thiccstr = "" if thickness == None else f"_{thickness}z"
    gapstr = "" if gap == None else f"_{gap}g"
    DCfolder = "../data/vary width"
    CCfolder = "../data/vary width"
    XWGfolder = "../data/vary width"
    folder_path = _resolve_data_path(folder)
    models = {
        # "ewg": partial(ideal.waveguide, neff=1.587838),
        # "wg": partial(ideal.waveguide, neff=1.587838),
        "ewg": ideal.waveguide,
        "wg": ideal.waveguide,
        # "coupler": partial(ideal.waveguide, length=10, neff=2),
        # "ewg": partial(ideal.waveguide, neff=2),
        # "wg": partial(ideal.waveguide, neff=2),
        "splitter": sdict_from_file(folder_path / f"dc_{str(splitter).replace('.', 'p')}{thiccstr}{gapstr}{angle}{MAstring}_S.dat"),
        "xwg": sdict_from_file(folder_path / f"xwg_{str(crossover).replace('.', 'p')}{thiccstr}{gapstr}{angle}{MAstring}_S.dat"),
        "crossover": sdict_from_file(folder_path / f"cc_{str(crossover).replace('.', 'p')}{thiccstr}{gapstr}{angle}{MAstring}_S.dat"),
    }
    #the taper really does not impact the simulation very much. different thickness and gaps and angles for the taper have some impact, but not much.
    #so give options. 0 is an ideal waveguide, 1 is a default taper of hard coded width and thickness and 90 angle, 2 you can choose thicc and width, 3 you can also choose angle
    # but really our sims show only small differences between the different types of tapers, so 1 is probably fine, other factors drive the simulation outcome.
    if taperChoice == 3:
        models["coupler"] = sdict_from_file(
            _resolve_data_path(
                f"./sparams/tapers/taper250.0_{str(splitter).replace('.', 'p')}{thiccstr}{gapstr}{angle}_S.dat"
            )
        )
    elif taperChoice == 2:
        models["coupler"] = sdict_from_file(
            _resolve_data_path(
                f"./sparams/tapers/taper250.0_{str(splitter).replace('.', 'p')}{thiccstr}{gapstr}_S.dat"
            )
        )
    elif taperChoice == 1:
        models["coupler"] = sdict_from_file(
            _resolve_data_path(
                f"./sparams/tapers/taper250.0_{defaultTaperThicc}z_{defaultTaperGap}g_S.dat"
            )
        )
    elif taperChoice == 0:
        models["coupler"] = partial(ideal.waveguide, length=10, neff=1.587838)
    else:
        raise ValueError("Invalid taperChoice")

    return models

# List a bunch of splitter and crossover pairs
configs = [
    (86.7, 122.4),
    (87.8, 125.5),
    (88.7, 128.0),
    (89.4, 130.0),
    (90.0, 131.5),
    (90.5, 132.7),
    (90.9, 133.6),
    (91.2, 134.4),
    (91.5, 135.0),
    (91.7, 135.5),
    (92.0, 136.1),
    (92.4, 136.8),
    (92.8, 137.6),
    (93.2, 138.7),
    (93.8, 140.2),
]

# Create netlist
gm_netlist = sax.RecursiveNetlist.parse_obj(rec_gm)