class Port:
    """Represents a port on a component.

    Attributes:
        name (str): The name of the port.
        type (str): The type of signal carried by the port, "electrical", "optical", or "logic"
        directionality (str): Either "input", "output", or "bidirectional".
        position (str): The side of the component where the port is displayed ("left", "right", "Up", or "Down").
        location (float): Position along the specified side, ranging from 0 to 1.
    """

    def __init__(
        self,
        name: str,
        directionality: str = "bidirectional",
        type: str = "optical",
        position: str = None,
        location: float = None,
    ):
        self.name = name
        self.type = type
        self.directionality = directionality
        self.position = position
        self.location = location

    def __repr__(self):
        return f"<Port {self.name!r} ({self.type}, {self.directionality})>"
