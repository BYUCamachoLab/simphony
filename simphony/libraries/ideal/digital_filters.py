from simphony.component.component import BlockModeComponent, SampleModeComponent, Component
from simphony.signal.block_mode import BlockModeOpticalSignal
from simphony.signal.sample_mode import SampleModeOpticalSignal
from simphony.component.port import Port
import jax.numpy as jnp
from scipy.signal import lfilter
from scipy.constants import speed_of_light as SPEED_OF_LIGHT
import jax
from dataclasses import replace
from simphony.simulation.simulation import SimulationParameters
from simphony.time_domain.vector_fitting.z_domain import state_space_response_discrete

class OpticalDiscreteFilter( 
    SampleModeComponent,
    BlockModeComponent,
):
    ports = [
        Port(
            name="in",
            type="optical",
            directionality="input",
        ),
        Port(
            name="out",
            type="optical",
            directionality="output",
        )
    ]
    
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        *,
        b: jnp.ndarray = jnp.asarray([0.0, 1.0]),
        a: jnp.ndarray = jnp.asarray([1.0]),
        center_wl = 1.55e-6,
        delay_compensation = 0,
    ):
        """
        b: of shape (num_modes, len_b)
        a: of shape (num_modes, len_a)

        If for each mode, the a coefficients are of length 1, 
        then the filter will be optimized as a fir filter
        """
        # We need at least 1 filter for each mode
        b, a = jnp.atleast_2d(b), jnp.atleast_2d(a)
        self.filter_coefficients = b/a[:, 0], a/a[:, 0]
        
        if a.shape[1] == 1:
            self.sample_mode_step = self._sample_mode_initial_state_fir
            self.sample_mode_step = self._sample_mode_step_fir
        else:
            self.sample_mode_step = self._sample_mode_initial_state_iir
            self.sample_mode_step = self._sample_mode_step_iir

        self.center_wl = center_wl
        self.delay_compensation = 0

    def _sample_mode_initial_state_iir(self, simulation_parameters):
        b, a = self.filter_coefficients
        M, L = simulation_parameters.num_optical_modes, simulation_parameters.optical_baseband_wavelengths.shape[0]
        # weight_x = jnp.zeros((M, L), dtype=complex)
        x_hist = jnp.zeros((L, M, b.shape[1]), dtype=complex)
        # weight_y = jnp.zeros((M, L), dtype=complex)
        y_hist = jnp.zeros((L, M, a.shape[1]), dtype=complex)
        state = (x_hist, y_hist)
        return state
    
    def _sample_mode_step_iir(self, inputs: dict,  state: jax.Array, simulation_parameters):
        delay_compensation = self.delay_compensation
        
        x_hist, y_hist = state

        ###
        # TODO: Modulated the inputs by delta_omega
        ###

        input_signal = inputs["in"]
        input_amplitude = input_signal.amplitude
        input_wavelength = input_signal.wavelength

        output_amplitude = jnp.zeros_like(input_amplitude)
        output_wavelength = input_wavelength
        
        x_hist = jnp.roll(x_hist, shift=1, axis=2)
        x_hist = x_hist.at[:, :, 0].set(input_amplitude)
        y_hist = jnp.roll(y_hist, shift=1, axis=2)

        b, a = self.filter_coefficients
        for mode_idx in range(simulation_parameters.num_optical_modes):
            b_single_mode, a_single_mode = b[mode_idx], a[mode_idx]
            for wl_idx, wl in enumerate(input_wavelength):
                f_center = SPEED_OF_LIGHT / self.center_wl
                f = SPEED_OF_LIGHT/wl
                delta_omega = 2*jnp.pi*(f - f_center) / f_center
                
                weight_x = b_single_mode@x_hist[wl_idx, mode_idx]
                weight_y = a_single_mode[1:]@y_hist[wl_idx, mode_idx, 1:]
                y_single_mode = weight_x - weight_y
                output_amplitude = output_amplitude.at[wl_idx, mode_idx].set(y_single_mode)

        y_hist = y_hist.at[:, :, 0].set(output_amplitude) 

        outputs = {
            "in": SampleModeOpticalSignal(
                amplitude=jnp.zeros_like(output_amplitude),
                wavelength=output_wavelength
            ),
            "out": SampleModeOpticalSignal(
                amplitude=output_amplitude,
                wavelength=output_wavelength
            ),
        }

        state = (x_hist, y_hist)
        return outputs, state
    
    def block_mode_response(self, inputs: dict, simulation_parameters):
        input_signal = inputs['in']
        input_amplitude = input_signal.amplitude
        input_wavelength = input_signal.wavelength
        
        output_amplitude = jnp.zeros_like(input_signal.amplitude)
        output_wavelength = input_wavelength

        n = jnp.arange(0, output_amplitude.shape[0], 1)

        b, a = self.filter_coefficients
        for mode_idx in range(simulation_parameters.num_optical_modes):
            b_single_mode, a_single_mode = b[mode_idx], a[mode_idx]
            for wl_idx, wl in enumerate(input_wavelength):
                f_center = SPEED_OF_LIGHT / self.center_wl
                f = SPEED_OF_LIGHT/wl
                delta_omega = 2*jnp.pi*(f - f_center) / f_center
                x_single_mode = jnp.exp(-delta_omega*n*1j)*input_amplitude[:, wl_idx, mode_idx]
                y_single_mode = jnp.exp(delta_omega*n*1j)*lfilter(b_single_mode, a_single_mode, x_single_mode)
                output_amplitude = output_amplitude.at[:, wl_idx, mode_idx].set(y_single_mode)

        outputs = {
            'in': BlockModeOpticalSignal(
                amplitude=jnp.zeros_like(output_amplitude),
                wavelength=output_wavelength,
            ),
            'out': BlockModeOpticalSignal(
                amplitude=output_amplitude,
                wavelength=output_wavelength
            )
        }

        return outputs
    
def bidirectional_discrete_state_space(
    num_inputs,
    num_outputs
)->type[Component]:
    """
    A is nxn, where n is the number of model poles
    B is nxm, where m is the number of ports
    C is mxn, and D is mxm
    """
    
    if not num_inputs == num_outputs:
        raise ValueError("number of inputs and outputs must match for bidirectional state space models")
    raise NotImplementedError("bidirectional state space model not implemented")

def discrete_state_space(
    num_inputs,
    num_outputs,
    input_port_names = None,
    output_port_names = None,
) -> type[Component]:
    """
    Constructor for creating state space models of arbitrary dimension
    """

    if input_port_names is None:
        input_port_names = [f"port{i}_in" for i in range(num_inputs)]
    if output_port_names is None:
        output_port_names = [f"port{i}_out" for i in range(num_outputs)]


    # if not input_port_names is None:
    #     if True:
    #         pass

    class DiscreteStateSpace( 
        SampleModeComponent,
    ):
        """
        A is nxn, where n is the number of poles
        B is nxm, where m is the number of inputs
        C is pxn, where p is the number of outputs
        and D is pxm

        Port names are assigned dynamically starting at 'port0_in' / 'port0_out' 
        and ending in 'p{m-1}_in' / 'p{p-1}_out'

        All input signals should be located in mode 0 (usually called the TE MODE), 
        otherwise they will be ignored

        """
        ports = [
            Port(
                name=port_name, 
                type="optical", 
                directionality = 'input'
            ) 
            for port_name in input_port_names
        ] + [
            Port(
                name=port_name,
                type="optical",
                directionality="output"
            )
            for port_name in output_port_names
        ]
        
        def __init__(
            self,
            simulation_parameters: SimulationParameters,
            *,
            A = None,
            B = None,
            C = None,
            D = None,
            center_wl = 1.55e-6,
            sampling_period = None,
            delay_compensation = 0,
            mode = 0,
        ):
            self.state_space_matrices = A, B, C, D
            self.center_wl = center_wl
            self.center_frequency = SPEED_OF_LIGHT / center_wl
            self.delay_compensation = delay_compensation
            self.mode = mode

        def sample_mode_initial_state(self, simulation_parameters):
            A, B, C, D = self.state_space_matrices
            x = jnp.zeros((len(simulation_parameters.optical_baseband_wavelengths), A.shape[0]), dtype=complex)
            return x
        
        def sample_mode_step(self, inputs: dict,  state: jax.Array, simulation_parameters):
            x = state
            A, B, C, D = self.state_space_matrices
            u = jnp.zeros(
                (len(simulation_parameters.optical_baseband_wavelengths), len(input_port_names)),
                dtype=complex
            )

            for port, signal in inputs.items():
                port_idx = self.port_order[port]
                wavelength = inputs[port].wavelength
                u = u.at[:, port_idx].set(signal.amplitude[:, self.mode])
            
            new_x = jnp.zeros_like(x)
            y = jnp.zeros((len(simulation_parameters.optical_baseband_wavelengths), len(output_port_names)),dtype=complex)

            for i, f in enumerate(SPEED_OF_LIGHT/simulation_parameters.optical_baseband_wavelengths):
                new_x = new_x.at[i].set(A@x[i] + B@u[i])
                y = y.at[i].set(C@x[i] + D@u[i])

            outputs = {}
            for i, port in enumerate(output_port_names):
                A_t = y[:, i]
                outputs[port] = SampleModeOpticalSignal(
                    amplitude = A_t.reshape((len(simulation_parameters.optical_baseband_wavelengths), 1)),
                    wavelength = simulation_parameters.optical_baseband_wavelengths
                )

            return outputs, new_x

        # TODO: MAKE eveyrthing say input_signals and not inputs
        def block_mode_response(self, input_signals, simulation_parameters):
            """
            We assume that all signals are on a common mode
            """
            #TODO: MAKE SURE THAT THE MATRIX ELEMENTS MATCH PORT ORDER
            _input_amplitude = list(input_signals.values())[0].amplitude
            wavelengths = list(input_signals.values())[0].wavelength
            N = _input_amplitude.shape[0]
            L = _input_amplitude.shape[1]
            M = 1 # We assume all inputs are on a common mode
            
            
            # TODO: Make it so that the state space model only has M input ports and N output ports and not NXN
            num_inputs
            u = jnp.zeros((N, L, num_inputs), dtype=complex)
            for i, port_name in enumerate(input_port_names):
                # u = u.at[i, :, :].set(input_signals[port_name].get("amplitude", jnp.zeros((N, L, 1), dtype=complex))[:,:,0])
                u = u.at[:, :, i].set(input_signals[port_name].amplitude[:, :, 0])

            y = jnp.zeros((N, L, num_outputs), dtype=complex)
            A, B, C, D = self.state_space_matrices
            for i, wl in enumerate(wavelengths):
                # TODO: modulate inputs based on the delta f
                _y, _ = state_space_response_discrete(A, B, C, D, u[:, i, :])
                y = y.at[:, i, :].set(_y)

            pass


        def to_fir_filter(
            self,
        ) -> Component:
            
            ### TODO: Actually return a fir filter
            return  None
    
    return DiscreteStateSpace
    

def mimo_fir_filter(
    impulse_response: jax.Array = jnp.array([[[]]]),
    center_wl = 1.55e-6,
    sampling_period = None,        
) -> type[Component]:
    """
    """
    
    class MIMOFIRFilter(
        SampleModeComponent,
    ):
        """
        If a 
        """
        ports = ...