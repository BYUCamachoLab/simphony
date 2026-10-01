"""Sample-rate inference for multirate circuits.

A component's sample rate is a property of where it sits in the graph:

* every *source* (a component without incoming connections), and every part of
  the circuit that is not separated from a source by a rate changer, runs at
  the reference rate `1 / simulation_parameters.dt`; `num_time_steps` counts
  samples at that rate, so `num_time_steps * dt` is the simulated duration;
* every rate changer (`simphony.component.component.RateChanger`, e.g. a
  decimator or an interpolator) multiplies the rate of its output ports by its
  `rate_ratio`.

Rates propagate through connections, so each region's rate is the reference
rate times the product of the rate ratios along any path from a source. Paths
that disagree are an error. Each region also has a *phase*: the time of its
first sample, shifted by decimator/interpolator sample offsets.

Simulators turn every region into a `RateDomain` and hand each component a copy
of the simulation parameters with that region's `dt`, `num_time_steps` and
`time_offset` (`local_parameters`). In sample mode, the circuit is advanced on a
base tick (the gcd of all sample periods) and each component is only evaluated
on the ticks where its own region has a sample.
"""

import math
import warnings
from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from simphony.component.component import RateChanger

IN = "in"
OUT = "out"
SINGLE = ""


@dataclass(frozen=True)
class RateDomain:
    """Sampling of one rate region.

    `period` and `phase` are exact multiples of the reference `dt`; `dt`,
    `num_time_steps` and `time_offset` are the values handed to components.
    `period_ticks` / `phase_ticks` express the same sampling in base ticks.
    """

    period: Fraction
    phase: Fraction
    dt: float
    num_time_steps: int
    time_offset: float
    period_ticks: int
    phase_ticks: int


class RateSchedule:
    """Result of `infer_sample_rates`."""

    def __init__(self, domains, rate_changers, tick, num_ticks, hyperperiod):
        self.domains = domains  # (instance, domain_name) -> RateDomain
        self.rate_changers = rate_changers
        self.tick = tick  # base tick, in units of the reference dt
        self.num_ticks = num_ticks
        self.hyperperiod = hyperperiod

    @property
    def is_multirate(self) -> bool:
        return any(d.period != 1 or d.phase != 0 for d in self.domains.values())

    def instance_domain(self, instance) -> RateDomain:
        """Domain in which `instance` is evaluated (rate changers: output)."""
        if instance in self.rate_changers:
            return self.domains[(instance, OUT)]
        return self.domains[(instance, SINGLE)]

    def input_domain(self, instance) -> RateDomain:
        if instance in self.rate_changers:
            return self.domains[(instance, IN)]
        return self.domains[(instance, SINGLE)]

    def port_domain(self, instance, port) -> RateDomain:
        if instance in self.rate_changers:
            return self.domains[
                (instance, _changer_domain(self.rate_changers[instance], port))
            ]
        return self.domains[(instance, SINGLE)]


def local_parameters(simulation_parameters, domain: RateDomain):
    """Copy of `simulation_parameters` describing one rate region."""
    return simulation_parameters.replace(
        dt=domain.dt,
        num_time_steps=domain.num_time_steps,
        time_offset=domain.time_offset,
    )


def _changer_domain(component, port_name) -> str:
    port = component._port_lookup_table[port_name]
    if port.directionality == "input":
        return IN
    if port.directionality == "output":
        return OUT
    raise ValueError(
        f"{type(component).__name__}: rate changers need directed ports; port "
        f"{port_name!r} is {port.directionality!r}"
    )


def _gcd(values):
    """Greatest common divisor of positive Fractions."""
    result = Fraction(0)
    for v in values:
        if v == 0:
            continue
        if result == 0:
            result = Fraction(v)
            continue
        num = math.gcd(
            result.numerator * v.denominator, v.numerator * result.denominator
        )
        result = Fraction(num, result.denominator * v.denominator)
    return result


def infer_sample_rates(instantiated_circuit, simulation_parameters) -> RateSchedule:
    """Infer the sample period and phase of every region of the circuit."""
    instances = {
        name: data["model"]
        for name, data in instantiated_circuit.instantiated_flat_netlist[
            "instances"
        ].items()
    }
    graph = instantiated_circuit.graph
    changers = {n: c for n, c in instances.items() if isinstance(c, RateChanger)}

    def node(instance, port):
        if instance in changers:
            return (instance, _changer_domain(changers[instance], port))
        return (instance, SINGLE)

    nodes = []
    for name in instances:
        nodes += [(name, IN), (name, OUT)] if name in changers else [(name, SINGLE)]

    # Constraints: rate[v] = rate[u] * ratio, stored in both directions.
    neighbours = {n: [] for n in nodes}

    def constrain(u, v, ratio, why):
        neighbours[u].append((v, Fraction(ratio), why))
        neighbours[v].append((u, 1 / Fraction(ratio), why))

    for src, dst, data in graph.edges(data=True):
        u, v = node(src, data["src_port"]), node(dst, data["dst_port"])
        constrain(u, v, 1, f"{src},{data['src_port']} -> {dst},{data['dst_port']}")
    for name, changer in changers.items():
        ratio = Fraction(changer.rate_ratio)
        constrain((name, IN), (name, OUT), ratio, f"{name} (rate x {ratio})")

    sources = {n for n in instances if n not in changers and graph.in_degree(n) == 0}

    # Relative rates within each connected region (BFS), then anchor on sources.
    rate = {}
    for seed in nodes:
        if seed in rate:
            continue
        region = {seed: Fraction(1)}
        how = {seed: "start"}
        queue = deque([seed])
        while queue:
            u = queue.popleft()
            for v, ratio, why in neighbours[u]:
                r = region[u] * ratio
                if v not in region:
                    region[v], how[v] = r, why
                    queue.append(v)
                elif region[v] != r:
                    raise ValueError(
                        f"Inconsistent sample rates at {v[0]!r}: reached at rate x{region[v]} "
                        f"(via {how[v]}) and at rate x{r} (via {why}). Every path into a "
                        "component must have the same net up/down-sampling factor."
                    )
        anchors = {region[(s, SINGLE)] for s in sources if (s, SINGLE) in region}
        if len(anchors) > 1:
            names = sorted(s for s in sources if (s, SINGLE) in region)
            raise ValueError(
                f"Sources {names} sit at different net up/down-sampling factors; all "
                "sources run at the reference rate 1/dt."
            )
        reference = anchors.pop() if anchors else Fraction(1)
        for n, r in region.items():
            rate[n] = r / reference

    period = {n: 1 / r for n, r in rate.items()}

    # Phases: propagate forward from sources along directed edges.
    phase = {}
    queue = deque()
    for s in sorted(sources):
        phase[(s, SINGLE)] = Fraction(0)
        queue.append((s, SINGLE))

    def visit(v, p, why):
        if v not in phase:
            phase[v] = p
            queue.append(v)
        elif phase[v] != p:
            warnings.warn(
                f"{v[0]!r} is fed by same-rate signals with different sample phases "
                f"({phase[v]} and {p} reference samples, via {why}); using {phase[v]}."
            )

    while queue:
        u = queue.popleft()
        instance, dom = u
        if dom == IN:
            visit(
                (instance, OUT),
                Fraction(changers[instance].output_phase(phase[u], period[u])),
                instance,
            )
        for _, dst, data in graph.out_edges(instance, data=True):
            if node(instance, data["src_port"]) == u:
                visit(node(dst, data["dst_port"]), phase[u], f"{instance} -> {dst}")
    for n in nodes:
        phase.setdefault(n, Fraction(0))

    tick = _gcd(list(period.values()) + [p for p in phase.values() if p != 0])
    duration = Fraction(simulation_parameters.num_time_steps)
    dt = Fraction(simulation_parameters.dt)
    num_ticks = duration / tick
    period_ticks = {n: int(p / tick) for n, p in period.items()}
    hyperperiod = math.lcm(*period_ticks.values()) if period_ticks else 1

    lcm_period = hyperperiod * tick
    if num_ticks.denominator != 1 or num_ticks % hyperperiod != 0:
        suggestion = math.ceil(duration / lcm_period) * lcm_period
        raise ValueError(
            f"num_time_steps={simulation_parameters.num_time_steps} is not a whole "
            f"number of the circuit's longest sample period ({lcm_period} reference "
            f"samples); use e.g. num_time_steps={suggestion}."
        )

    domains = {}
    for n in nodes:
        p = period[n]
        domains[n] = RateDomain(
            period=p,
            phase=phase[n],
            dt=float(p * dt),
            num_time_steps=int(duration / p),
            time_offset=float(phase[n] * dt),
            period_ticks=period_ticks[n],
            phase_ticks=int(phase[n] / tick) % period_ticks[n],
        )
    return RateSchedule(domains, changers, tick, int(num_ticks), hyperperiod)
