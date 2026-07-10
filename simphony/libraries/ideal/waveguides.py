from simphony.circuit.circuit import BlockModeComponent, SteadyStateComponent


class Waveguide(SteadyStateComponent, BlockModeComponent):
    pass


class Fiber(SteadyStateComponent, BlockModeComponent):
    pass


class GRINFiber(SteadyStateComponent, BlockModeComponent):
    pass
