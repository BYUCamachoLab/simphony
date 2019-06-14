

from simphony.core import base

class ebeam_bdc_te1550(base.Component):
    
    def __init__(self):
        # Calculate s parameters
        super().__init__(component_type=type(self).__name__, s_parameters=None)
    
