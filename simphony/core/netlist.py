# netlist.py

# This file contains everything related to netlist generation and modeling.

# from simphony.components import BaseComponent, create_component_by_name

from typing import List#, Tuple

from simphony.core import ComponentModel, ComponentInstance
from simphony.core import connect as rf

# LLNetlist = List[Tuple[ComponentInstance, int, ComponentInstance, int]]

class Netlist:
    """Represents a netlist.

    Maintains a list of connected components as well as a net counter for 
    automatically assigning id's to connections as they are made.

    Notes
    -----
    The convention for net id numbering follows these rules:
    * All non-negative integers are considered internal nets connecting 
      components to each other within the netlist.
    * All negative integers are external nets that will be considered the
      inputs or outputs when a simulation is run on the netlist. Other
      components or netlists can be connected to these external ports.
    * Numbers are never skipped; if a number is missing, the netlist is
      corrupted and will not work properly.
    """

    def __init__(self, components: List[ComponentInstance]=None):
        """Creates a Netlist object for storing components.

        Netlist can be instantiated with an already complete list of 
        components. The netlist, however, performs no error checking on this
        list and assumes that the ComponentInstances contained within it
        already have nets assigned correctly and following the appropriate 
        numbering convention.

        Parameters
        ----------
        components : List[ComponentInstance]
            A list of pre-initialized components.
        """
        self._internal_net = -1
        self._external_net = 0
        self.components = [] if components is None else components

    # def get_external_components(self):
    #     return [component for component in self.components if (any(int(x) < 0 for x in component.nets))]

    def _next_internal(self):
        """Returns the next available internal net ID number.

        Returns
        -------
        int
            The next available internal net ID number.
        """
        self._internal_net += 1
        return self._internal_net

    def _next_external(self):
        """Returns the next available external net ID number.

        Returns
        -------
        int
            The next available external net ID number.
        """
        self._external_net -= 1
        return self._external_net

    def load(self, data, formatter='ll'):
        """Loads formatted component data into a netlist.

        Parameters
        ----------
        data
            The data to be loaded into the netlist.
        format : str
            Specification for the way the data is formatted.

        'format' can be any one of the following:
        
        =====   =====
        string  description
        -----   -----
        'll'    (list of lists)
        =====   =====
        """
        loader = None
        if formatter == 'll':
            loader = self._ll_loader
        loader(data)

    def _ll_loader(self, connection_list):
        """Loads the netlist with data formatted as a list of lists.

        Format for the data is a list of lists, as follows:
        [[instance_1, port_m, instance_2, port_n],
         [instance_3, port_m, instance_4, port_n],
         [...                                   ]]

        Ports are 0-indexed.

        Parameters
        ----------
        connection_list : List[List]
            A list of lists containing the connections for the netlist.
        """
        # Maintaining a list is required to make the ordering deterministic.
        components = []

        # If the data was zipped together, unzip it.
        if type(connection_list) == zip.__class__:
            connection_list = list(connection_list)

        # For each connection, assign net numbers to the appropriate 
        # components and save references to the components in a list.
        for connection in connection_list:
            c1, p1, c2, p2 = connection
            net_id = self._next_internal()
            c1.nets[p1] = net_id
            c2.nets[p2] = net_id
            components.append(c1)
            components.append(c2)
        
        # Any unassigned nets after all connections have been listed are 
        # external ports.
        for component in components:
            component.nets = [net if net is not None else self._next_external() for net in component.nets]

        # We do however, want to remove duplicate components and only track
        # one reference per component.
        self.components = list(set(components))

    @property
    def net_count(self):
        """Returns the number of internal nets in the Netlist.

        Finds the number of internal nets by iterating through the components
        and finding the max net number. Since internal net id's are assigned 
        beginning from '0', the total number of nets is always max(nets) + 1.

        Returns
        -------
        int
            The total count of internal nets.
        """
        # https://stackoverflow.com/a/29244327/11530613
        nets = [net for sublist in [comp.nets for comp in self.components] for net in sublist]
        return max(nets) + 1
