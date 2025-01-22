import numpy as np
import matplotlib.pyplot as plt
import sax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from simphony.libraries import ideal
from simphony.utils import dict_to_matrix,pole_residue_to_time_system
from simphony.time_domain.utils import gaussian_pulse
from simphony.time_domain.pole_residue_model import IIRModelBaseband
from simphony.time_domain.utils import pole_residue_to_time_system
from simphony.libraries import siepic


class TimeDomainSim:
    def __init__(self, netlist:dict, models:dict):
        self.netlist = netlist
        self.models = models
        self.instances = {}

        for instance_name, model_name in self.netlist['instances'].items():
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' is not defined in self.models.")
            self.instances[instance_name] = self.models[model_name]

        used_model_names = set(self.netlist['instances'].values())
        all_model_names = set(self.models.keys())
        unused_model_names = all_model_names - used_model_names

        if unused_model_names:
            print(f"Warning: The following models are never called in the netlist: "
                  f"{unused_model_names}")
            
        self.connections = {instance: {} for instance in self.instances.keys()}
        for designation_a, designation_b in self.netlist['connections'].items():
            instance_a, port_a = map(str.strip, designation_a.split(','))
            instance_b, port_b = map(str.strip, designation_b.split(','))
            
            self.connections[instance_a][port_a] = (instance_b, port_b)
            self.connections[instance_b][port_b] = (instance_a, port_a)
        
        self.ports = {}
        for circuit_port, designation in self.netlist['ports'].items():
            instance_name, instance_port = map(str.strip, designation.split(','))
            self.ports[circuit_port] = (instance_name, instance_port)
            
        
            
    def build_model(self, 
                    wvl = np.linspace(1.5, 1.6, 200),
                    center_wvl = 1.55,
                    model_order = 50,
                    ):
        self.active = {}
        if 'active_components' in self.netlist:
            for active_name in self.netlist['active_components'].items():
                print("hello World")
                self.active[active_name] = active_name
                              
        else:
            circuit, info = sax.circuit(netlist = self.netlist, models= self.models)
            s = circuit(wl = wvl)
            self.S = np.asarray(dict_to_matrix(s))
            model = IIRModelBaseband(wvl,center_wvl,self.S, model_order)
            self.time_system = pole_residue_to_time_system(model)
    
            
        