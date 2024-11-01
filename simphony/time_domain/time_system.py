from abc import ABC, abstractmethod
from jax.typing import ArrayLike

class TimeSystem(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def response(self, input_signal) -> ArrayLike:
        """Compute the system response."""
        pass
    