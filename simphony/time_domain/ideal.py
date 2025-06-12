"""Ideal time-domain models."""

from queue import Queue

import jax.numpy as jnp
import numpy as np
import sax
from jax.typing import ArrayLike

import simphony.libraries.ideal as fd
from simphony.time_domain.time_system import (
    BlockModeSystem,
    SampleModeSystem,
    TimeSystem,
)
from simphony.utils import dict_to_matrix, mul_polar


class TimeCoupler(TimeSystem):
    def __init__(
        self,
        coupling: float = 0.5,
        loss: float = 0.0,
        phi: float = jnp.pi / 2,
    ) -> None:
        super().__init__()
        self.s_params = dict_to_matrix(
            fd.coupler(coupling=coupling, loss=loss, phi=phi)
        )
        self.num_ports = 4
        self.ports = ["o0", "o1", "o2", "o3"]
        pass

    def response(self, inputs: dict) -> dict:
        N = inputs["o0"].shape
        response = {
            "o0": jnp.zeros((N), dtype=complex),
            "o1": jnp.zeros((N), dtype=complex),
            "o2": jnp.zeros((N), dtype=complex),
            "o3": jnp.zeros((N), dtype=complex),
        }

        for i in range(self.num_ports):
            response[f"o{i}"] = (
                inputs["o0"] * self.s_params[0, i, 0]
                + inputs["o1"] * self.s_params[0, i, 1]
                + inputs["o2"] * self.s_params[0, i, 2]
                + inputs["o3"] * self.s_params[0, i, 3]
            )
        return response

    def clear(self) -> None:
        pass


class TimeWaveguide(TimeSystem):
    def __init__(
        self,
        # wl: ArrayLike | float = 1.55,
        dt: float,
        wl0: float = 1.55,
        neff: float = 2.34,
        ng: float = 3.4,
        length: float = 10.0,
        loss: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_ports = 2
        self.ports = ["o0", "o1"]
        c = 299792458
        group_velocity = c / ng
        omega = c / (1.55e-6)

        delay = (length * 1e-6) / (group_velocity)
        self.num_delay_indices = round(delay / dt)

        phase_shift = jnp.mod(omega * delay, 2 * jnp.pi)
        loss_mag = loss / (10 * jnp.log10(jnp.exp(1)))
        alpha = loss_mag * 1e-4
        amplitude = jnp.asarray(jnp.exp(-alpha * length / 2), dtype=complex)

        self.transmission = amplitude * jnp.exp(1j * phase_shift)
        self.forward_wave = Queue()
        self.backward_wave = Queue()

        for i in range(self.num_delay_indices):
            self.forward_wave.put(0 + 0j)
            self.backward_wave.put(0 + 0j)

    def run(self, inputs: dict) -> dict:
        N = inputs["o0"].shape[0]
        o0_response = jnp.zeros((N), dtype=complex)
        o1_response = jnp.zeros((N), dtype=complex)

        for i in range(N):
            a = complex(inputs["o0"][i])
            b = complex(inputs["o1"][i])
            o0_response = o0_response.at[i].set(
                self.transmission * self.backward_wave.get()
            )
            o1_response = o1_response.at[i].set(
                self.transmission * self.forward_wave.get()
            )

            self.forward_wave.put(a)
            self.backward_wave.put(b)

        response = {
            "o0": o0_response,
            "o1": o1_response,
        }

        return response

    def clear(self) -> None:
        while not self.forward_wave.empty():
            self.forward_wave.get()
        while not self.backward_wave.empty():
            self.backward_wave.get()
        for i in range(self.num_delay_indices):
            self.forward_wave.put(0 + 0j)
            self.backward_wave.put(0 + 0j)


class Modulator(SampleModeSystem, BlockModeSystem):
    # … your __init__ stays as before (but remove any internal “self.countstep” updates) …
    def __init__(
        self,
        mod_signal: ArrayLike | float = 0.0,
        k_p: float = 1.0,
    ) -> None:
        super().__init__()

        self.num_ports = 2
        self.ports = ["o0", "o1"]
        self.phase_sequence = mod_signal * k_p

    def init_state(self, **kwargs):
        # Return whatever you want the initial state to be.
        # For example, if you have a JAX array of per‐time phases, just return index = 0:

        return jnp.int32(0)

    def step(self, prev_idx: jnp.ndarray, inputs: tuple, **kwargs):
        """
        A _pure_ function—no side‐effects!—that returns (new_idx, (out0, out1)).
        E.g.:
           phase = self.phase_sequence[prev_idx]
           coeff = jnp.exp(1j * phase)
           out0 = input1 * coeff
           out1 = input0 * coeff
           return prev_idx + 1, (out0, out1)
        """
        phase = self.phase_sequence[prev_idx]
        coeff = jnp.exp(1j * phase)
        out0 = inputs[1] * coeff
        out1 = inputs[0] * coeff
        return prev_idx + 1, (out0, out1)

    def run(self, inputs: dict, **kwargs) -> dict:
        N = inputs["o0"].shape[0]
        o0_response = jnp.zeros((N), dtype=complex)
        o1_response = jnp.zeros((N), dtype=complex)

        for i in range(N):
            o0_response = o0_response.at[i].set(
                inputs["o1"][i] * self.s_mod[self.countstep]
            )
            o1_response = o1_response.at[i].set(
                inputs["o0"][i] * self.s_mod[self.countstep]
            )
        self.countstep += 1
        response = {
            "o0": o0_response,
            "o1": o1_response,
        }


class PhaseModulator(SampleModeSystem, BlockModeSystem):
    # … your __init__ stays as before (but remove any internal “self.countstep” updates) …
    def __init__(self, time: ArrayLike, voltage: ArrayLike) -> None:
        super().__init__()

        self.num_ports = 2
        self.ports = ["o0", "o1"]

        self.time = time
        self.voltage = voltage

    def init_state(self, **kwargs):
        # Return whatever you want the initial state to be.
        # For example, if you have a JAX array of per‐time phases, just return index = 0:
        self._voltage

        return jnp.int32(0)

    def step(self, prev_idx: jnp.ndarray, input0, input1, **kwargs):
        """
        A _pure_ function—no side‐effects!—that returns (new_idx, (out0, out1)).
        E.g.:
           phase = self.phase_sequence[prev_idx]
           coeff = jnp.exp(1j * phase)
           out0 = input1 * coeff
           out1 = input0 * coeff
           return prev_idx + 1, (out0, out1)
        """
        phase = self.phase_sequence[prev_idx]
        coeff = jnp.exp(1j * phase)
        out0 = input1 * coeff
        out1 = input0 * coeff
        return prev_idx + 1, (out0, out1)

    def run(self, inputs: dict, **kwargs) -> dict:
        N = inputs["o0"].shape[0]
        o0_response = jnp.zeros((N), dtype=complex)
        o1_response = jnp.zeros((N), dtype=complex)

        for i in range(N):
            o0_response = o0_response.at[i].set(
                inputs["o1"][i] * self.s_mod[self.countstep]
            )
            o1_response = o1_response.at[i].set(
                inputs["o0"][i] * self.s_mod[self.countstep]
            )
        self.countstep += 1
        response = {
            "o0": o0_response,
            "o1": o1_response,
        }


# class Modulator(TimeSystem):
#     def __init__(
#             self,
#             mod_signal: ArrayLike|float = 0.0,
#             k_p: float = 1.0,
#      ) -> None:
#         super().__init__()

#         self.num_ports = 2
#         self.ports = ['o0','o1']
#         phase_shift = k_p * mod_signal
#         self.s_mod = jnp.exp(1j * phase_shift)
#         self.countstep = 0

#     def response(self, inputs:dict) -> dict:
#         N = inputs['o0'].shape[0]
#         o0_response = jnp.zeros((N),dtype = complex)
#         o1_response = jnp.zeros((N), dtype=complex)

#         for i in range(N):
#             o0_response = o0_response.at[i].set(inputs['o1'][i] * self.s_mod[self.countstep])
#             o1_response = o1_response.at[i].set(inputs['o0'][i] * self.s_mod[self.countstep])
#         self.countstep += 1
#         response = {
#             "o0": o0_response,
#             "o1": o1_response,
#         }


#         return response

#     def append(self, mod_signal: ArrayLike|float) -> None:
#         self.s_mod = jnp.append(self.s_mod, jnp.exp(1j * mod_signal))


#     def reset(self) -> None:
#         self.countstep = 0


class MMI(TimeSystem):
    def __init__(
        self,
        r: int = 2,
        s: int = 2,
        length: float = 10.0,
        loss: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_ports = r + s
        self.length = length
        self.loss = loss
        self.r = r
        self.s = s
        self.ports = [f"o{i}" for i in range(self.num_ports)]
        N_size = self.r + self.s
        phases = jnp.zeros((N_size, N_size))
        N = self.r
        for i in range(1, self.r + 1):
            for j in range(1, self.s + 1):
                if (i + j) % 2 == 0:
                    num = -(jnp.pi / (4 * N)) * (j - i) * (2 * N + i - j)
                else:
                    num = -(jnp.pi / (4 * N)) * (i + j - 1) * (2 * N - i - j + 1)

                phases = phases.at[i - 1, N_size - j].set(num)
                phases = phases.at[N_size - j, i - 1].set(num)

        loss_mag = self.loss / (10 * jnp.log10(jnp.exp(1)))
        alpha = loss_mag * 1e-4
        amplitude = jnp.asarray(jnp.exp(-alpha * self.length / 2), dtype=complex)
        s_dict_time = {}

        for i in range(self.r):  # inputs 0 … r-1
            for j in range(self.s):  # outputs r … r+s-1
                phi = phases[i, j + self.r]
                s_dict_time[(f"o{i}", f"o{j+self.r}")] = (
                    amplitude / jnp.sqrt(self.s) * jnp.exp(1j * phi)
                )

        for i in range(self.s):  # inputs r … r+s-1
            for j in range(self.r):  # outputs 0 … r-1
                phi = phases[i + self.s, j]
                s_dict_time[(f"o{i+self.r}", f"o{j}")] = (
                    amplitude / jnp.sqrt(self.r) * jnp.exp(1j * phi)
                )

        # now build S as [output, input]
        ports = self.ports
        N = len(ports)
        S = np.zeros((N, N), dtype=complex)
        for (inp, out), val in s_dict_time.items():
            ci = ports.index(inp)
            rj = ports.index(out)
            S[rj, ci] = val

        self.s_dict_time = S

    def response(self, inputs: dict) -> dict:

        response = {}
        N = len(self.ports)
        for j, port in enumerate(self.ports):
            # sum over all inputs
            resp = sum(inputs[f"o{l}"] * self.s_dict_time[j, l] for l in range(N))
            response[port] = resp
        return response

    def reset(self) -> None:
        pass
