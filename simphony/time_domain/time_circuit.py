from jax.typing import ArrayLike
import jax.numpy as jnp

class TimeCircuit:
    def __init__(self, netlist: dict, models: dict):
        # DEFAULT_TIME_STEP = 1e-14
        # self.time_step = DEFAULT_TIME_STEP
        self.netlist = netlist
        # self.connections = netlist['connections']
        self.models = models
        #self.connections = {}
        # self.ports = netlist['ports']
        # self.model_types = models
    
    def instantiate(self, dt, clear_on_reset=True, **kwargs):
        self.dt = dt
        instantiated_models = {}
        self.clear = clear_on_reset
        for model_name, model in kwargs.items():
            if model_name in self.models:
                instantiated_models[model_name] = model
            else:
                raise ValueError(f"Model '{model_name}' is not defined in models.")
            
        self.instances = {}
        for instance, model_name in self.netlist['instances'].items():
            self.instances[instance] = instantiated_models[model_name]
            # self.instance_outputs[instance] = {port: jnp.array([0+0j]) for port in instantiated_models[model_name].ports}
        
        self.connections = {instance: {} for instance in self.instances.keys()}
        for designation_a, designation_b in self.netlist['connections'].items():
            instance_a, port_a = map(str.strip, designation_a.split(','))
            instance_b, port_b = map(str.strip, designation_b.split(','))
            # self.instance_input_addresses[instance_a] = (instance_b, port_b) # doesn't work since it does not specify the a port
            self.connections[instance_a][port_a] = (instance_b, port_b)
            self.connections[instance_b][port_b] = (instance_a, port_a)
        

        self.ports = {}
        for circuit_port, designation in self.netlist['ports'].items():
            instance_name, instance_port = map(str.strip, designation.split(','))
            self.ports[circuit_port] = (instance_name, instance_port)
            
        # we need to create a list containing time systems and a corresponding list with connections
        # The index of the two lists will correlate

        print(f"Instances created with dt={dt}: {self.instances}")
    

    def run_sim(self, t: ArrayLike, inputs: dict)->dict:
        if self.clear:
            for instance_name, time_system in self.instances.items():
                time_system.clear()
        
        self.inputs = inputs
        self.outputs = {port: jnp.array([]) for port in self.ports}
        self.instance_outputs = {}
        for instance_name, time_system in self.instances.items():
            self.instance_outputs[instance_name] = {port: jnp.array([0+0j]) for port in time_system.ports}
        
        for _ in t:
            self.step()
            pass
            
        return self.outputs


    def step(self):
        for instance_name, time_system in self.instances.items():
            instance_inputs = {}
            for port in time_system.ports:
                # TODO: Contrust the input in order to run the response function
                #       Then update the instance_outputs
                # Looking in connections
                if instance_name in self.connections and port in self.connections[instance_name]:
                    print(f"Looking in connections")
                    source_name, source_port = self.connections[instance_name][port]
                    instance_inputs[port] = self.instance_outputs[source_name][source_port]

                else:
                    # Looking in Circuit Ports
                    print(f"Looking in Circuit Ports")
                    instance_inputs[port] = self.ports
                    designation = (instance_name, port)
                    circuit_port = next((k for k, v in self.ports.items() if v == designation), None)
                    instance_inputs[port] = jnp.array(self.inputs[circuit_port][0])
                    self.inputs[circuit_port] = self.inputs[circuit_port][1:]
                    
            # Compute Resposne for each instance and submit result to instance_outputs
            outputs = time_system.response(instance_inputs)
            for port_name in outputs:
                self.instance_outputs[instance_name][port_name] = outputs[port_name]
            pass
        


        for circuit_port, (instance, instance_port) in self.ports.items():
            self.outputs[circuit_port] = jnp.concatenate([self.outputs[circuit_port], self.instance_outputs[instance][instance_port]])



