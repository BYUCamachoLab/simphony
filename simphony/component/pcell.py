class PCell:
    """
    Some circuit components will not have a well defined, internal structure 
    until after they are parameterized. For this case, we use parameterized 
    cells, or PCells. 

    The Circuit objects will display the PCell as a single component. 
    
    All PCell classes must have well defined external ports, prior to instantiation.

    Similar to the Component Classes, the PCell class is a baseclass for special 
    Component objects that require a dynamic structure at the simulation runtime.
    Therefore, the __init__() function should be called by the InstantiatedCircuit class, and 
    not the designer of the PCell. 

    The PCell designer should simply inherit from the PCell baseclass and overwrite the 
    appropriate class fields and methods. 
    
    """
    def __init__():
        pass