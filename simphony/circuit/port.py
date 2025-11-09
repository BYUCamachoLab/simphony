class Port:
    """
    Represents a port on a component.

    Attributes:
        name (str): The name of the port.
        type (str): The type of signal carried by the port, "Electrical", "Optical", or "Logic"
        directionality (str): Either "Input", "Output", or "Bidirectional".
        position (str): The side of the component where the port is displayed ("Left", "Right", "Up", or "Down").
        location (float): Position along the specified side, ranging from 0 to 1.
    """

    def __init__(
        self, 
        name: str, 
        directionality: str = "bidirectional", 
        type: str = "optical",
        position: str = None, 
        location: float = None
    ):
        self.name = name
        self.type = type
        self.directionality = directionality
        self.position = position
        self.location = location