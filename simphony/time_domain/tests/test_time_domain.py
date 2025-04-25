import os
import pickle
import pytest
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from simphony.time_domain.simulation import TimeSim,TimeResult
from simphony.time_domain.utils import  gaussian_pulse, smooth_rectangular_pulse
from simphony.libraries import siepic
from simphony.time_domain.ideal import Modulator

# Let's assume your main code is in `my_module.main_code`:
# from my_module.main_code import run_simulation

def run_simulation():
    T = 2.5e-11
    dt = 1e-14      # Total time duration (40 ps)
    t = jnp.arange(0, T, dt) # Time array
    t0 = 1.0e-11  # Pulse start time


    f_mod =0
    m = f_mod * jnp.ones(len(t),dtype = complex)
    f_mod2 =jnp.pi/4 
    # m2 = f_mod2 * jnp.ones(len(t),dtype = complex)

    x = jnp.linspace(0, 3.14, len(t))

    mu = 1.30  # center the Gaussian in the middle of the interval
    sigma = 0.15     # adjust sigma for desired width
    x = jnp.linspace(0, 3.14, len(t))
    # Compute the Gaussian function

    gaussian = np.pi*jnp.exp(-0.5*((x - mu) / sigma) ** 2)

    # Optionally, normalize so the area under the curve is 1
    #gaussian = gaussian / jnp.trapezoid(gaussian, x)
    zero = 0*x
    timePhaseInstantiated = Modulator(mod_signal=gaussian)

    netlist={
        "instances": {
            "wg": "waveguide",
            "wg2": "waveguide",
            "pm": "phase_modulator",
            "y": "y_branch",
            "y2": "y_branch",
        },
        "connections": {
            "wg,o0":"y,port_2",
            "wg,o1":"pm,o0",
            "y2,port_2":"pm,o1",
            "wg2,o0":"y,port_3",
            "y2,port_3":"wg2,o1",
        },
        "ports": {
            "o0":"y,port_1",
            "o1":"y2,port_1",
        },
    }
    models={
        "waveguide": siepic.waveguide,
        "y_branch": siepic.y_branch,
        "bidirectional": siepic.bidirectional_coupler,
        "phase_modulator": timePhaseInstantiated,
    }
    active_components = {
        "pm", "pm2"
    }


    time_sim = TimeSim(
        netlist=netlist,
        models=models,
        active_components=active_components,
        )

    num_measurements = 200
    model_order = 50
    center_wvl = 1.548
    wvl = np.linspace(1.5, 1.6, num_measurements)
    options = {'wl':wvl,'wg':{"length": 10.0, "loss": 100}, 'wg2':{"length": 10.0, "loss": 100}}

    
    time_sim.build_model(model_parameters=options, dt = dt)
    

    num_outputs = 2

    inputs = {
                f'o{i}': smooth_rectangular_pulse(t,0.5e-11,1.5e-11) if i == 0 else jnp.zeros_like(t)
                for i in range(num_outputs)
            }



    modelResult =time_sim.run(t, inputs)
    
    
    return modelResult.outputs

def run_simulation2():
    T = 2.0e-11
    newT = 4.0e-11
    dt = 0.5e-14      # Total time duration (40 ps)
    t = jnp.arange(0, T, dt)
    ddt = 1e-14
    newt = jnp.arange(0,newT, ddt) 

    x = jnp.linspace(0, 3.14, len(t))

    mu = 1.30  # center the Gaussian in the middle of the interval
    sigma = 0.15     # adjust sigma for desired width
    x = jnp.linspace(0, 3.14, len(t))

    gaussian = np.pi*jnp.exp(-0.5*((x - mu) / sigma) ** 2)

    timePhaseInstantiated = Modulator(mod_signal=gaussian)

    netlist={
        "instances": {
            "wg": "waveguide",
            "wg2": "waveguide",
            "pm": "phase_modulator",
            "y": "y_branch",
            "y2": "y_branch",
        },
        "connections": {
            "wg,o0":"y,port_2",
            "wg,o1":"pm,o0",
            "y2,port_2":"pm,o1",
            "wg2,o0":"y,port_3",
            "y2,port_3":"wg2,o1",
        },
        "ports": {
            "o0":"y,port_1",
            "o1":"y2,port_1",
        },
    }
    models={
        "waveguide": siepic.waveguide,
        "y_branch": siepic.y_branch,
        "bidirectional": siepic.bidirectional_coupler,
        "phase_modulator": timePhaseInstantiated,
    }
    active_components = {
        "pm", "pm2"
    }


    time_sim = TimeSim(
        netlist=netlist,
        models=models,
        active_components=active_components,
        )

    num_measurements = 200
    model_order = 50
    center_wvl = 1.548
    wvl = np.linspace(1.5, 1.6, num_measurements)
    options = {'wl':wvl,'wg':{"length": 10.0, "loss": 100}, 'wg2':{"length": 10.0, "loss": 100}}

    
    time_sim.build_model(model_parameters=options, dt = dt)

    num_outputs = 2



    inputs = {
                f'o{i}': smooth_rectangular_pulse(newt,0.5e-11,1.5e-11) if i == 0 else jnp.zeros_like(t)
                for i in range(num_outputs)
            }


    modelResult =time_sim.run(newt, inputs)
    
    
    return modelResult.outputs

def run_simulation3():
    T = 2.0e-11
    dt = 0.5e-14      # Total time duration (40 ps)
    t = jnp.arange(0, T, dt)

    netlist={
        "instances": {
            "wg": "waveguide",
            "wg2": "waveguide",
            "y": "y_branch",
            "hr":"half_ring",
            "hr2":"half_ring",
            "y2": "y_branch",
        },
        "connections": {
            "hr,port_1":"hr2,port_1",
            "hr,port_3":"hr2,port_3",

        },
        "ports": {
            "o0":"hr,port_2",
            "o2":"hr2,port_4",
            "o1":"hr,port_4",
            "o3":"hr2,port_2",
        },
    }
    models={
        "waveguide": siepic.waveguide,
        "y_branch": siepic.y_branch,
        "half_ring": siepic.half_ring,
        "bidirectional": siepic.bidirectional_coupler,
    }



    time_sim = TimeSim(
        netlist=netlist,
        models=models,
        
        )

    num_measurements = 200
    model_order = 50
    center_wvl = 1.548
    wvl = np.linspace(1.5, 1.6, num_measurements)
    options = {'wl':wvl,'wg':{"length": 10.0, "loss": 100}, 'wg2':{"length": 10.0, "loss": 100}}


    time_sim.build_model(model_parameters=options, dt = dt,max_size = 6)



    num_outputs = 4

    inputs = {
                f'o{i}': smooth_rectangular_pulse(t,0.0e-11,4.0e-11) if i == 0 else jnp.zeros_like(t)
                for i in range(num_outputs)
            }


    modelResult =time_sim.run(t, inputs)
    
    
    return modelResult.outputs


def run_simulation4():
    T = 2.5e-12
    dt = 1e-15      # Total time duration (40 ps)
    t = jnp.arange(0, T, dt) # Time array
    t0 = 1.0e-11  # Pulse start time


    f_mod =0
    m = f_mod * jnp.ones(len(t),dtype = complex)

    # m2 = f_mod2 * jnp.ones(len(t),dtype = complex)

    x = jnp.linspace(0, 3.14, len(t))

    mu = 1.70  # center the Gaussian in the middle of the interval
    sigma = 0.15     # adjust sigma for desired width
    x = jnp.linspace(0, 3.14, len(t))
    # Compute the Gaussian function

    gaussian = np.pi*jnp.exp(-0.5*((x - mu) / sigma) ** 2)


    # Optionally, normalize so the area under the curve is 1
    #gaussian = gaussian / jnp.trapezoid(gaussian, x)
    zero = 0*x
    timePhaseInstantiated = Modulator(mod_signal=gaussian)

    netlist={
        "instances": {
            "wg": "waveguide",
            "wg2": "waveguide",
            "pm": "phase_modulator",
            "y": "y_branch",
            "y2": "y_branch",
        },
        "connections": {
            "wg,o0":"y,port_2",
            "wg,o1":"pm,o0",
            "y2,port_2":"pm,o1",
            "wg2,o0":"y,port_3",
            "y2,port_3":"wg2,o1",
        },
        "ports": {
            "o0":"y,port_1",
            "o1":"y2,port_1",
        },
    }
    models={
        "waveguide": siepic.waveguide,
        "y_branch": siepic.y_branch,
        "bidirectional": siepic.bidirectional_coupler,
        "phase_modulator": timePhaseInstantiated,
    }
    active_components = {
        "pm", "pm2"
    }


    time_sim = TimeSim(
        netlist=netlist,
        models=models,
        active_components=active_components,
        )

    num_measurements = 200
    model_order = 50
    center_wvl = 1.548
    wvl = np.linspace(1.5, 1.6, num_measurements)
    options = {'wl':wvl,'wg':{"length": 10.0, "loss": 100}, 'wg2':{"length": 10.0, "loss": 100}}


    time_sim.build_model(model_parameters=options, dt = dt)


    num_outputs = 2

    inputs = {
                f'o{i}': smooth_rectangular_pulse(t,0.0e-12,1.5e-12) if i == 0 else jnp.zeros_like(t)
                for i in range(num_outputs)
            }


    modelResult =time_sim.run(t, inputs)
    
    return modelResult.outputs

@pytest.mark.simulation
def test_compare_simulation_results_active_MZI():
    """
    This test runs the simulation, then compares it to a known
    'golden' result stored in a pickle file.
    """

    # 1) Run the simulation to get new results
    new_results = run_simulation()

    # 2) Load the reference (golden) data from disk
    #    Adjust the path to where your .pkl file actually is.
    reference_path = os.path.join(
        os.path.dirname(__file__),        # directory of THIS test file
        "test_comparison_results/",                 # subfolder for reference data
        "simulation_results.pkl"          # the actual pickle file name
    )

    # If you haven't created the file yet, run your code once to generate it.
    # Then rename it or copy it into reference_data as the "golden" result.
    with open(reference_path, "rb") as f:
        golden_results = pickle.load(f)

    # 3) Compare dictionary keys
    assert new_results.keys() == golden_results.keys(), "Mismatch in dictionary keys."

    # 4) Compare each item in the dictionary
    for key in new_results.keys():
        # Compare arrays
        # If you want an exact match, use jnp.array_equal
        # For numerical tolerance, use jnp.allclose
        assert jnp.allclose(new_results[key], golden_results[key], rtol=1e-4, atol=1e-6), (
            f"Mismatch for key: {key}"
            )

    # If we reach here, everything matched!
    print("All results match expected data!")


@pytest.mark.simulation
def test_compare_simulation_results_active_MZI_time_change():
    """
    This test runs the simulation, then compares it to a known
    'golden' result stored in a pickle file.
    """

    # 1) Run the simulation to get new results
    new_results = run_simulation2()

    # 2) Load the reference (golden) data from disk
    #    Adjust the path to where your .pkl file actually is.
    reference_path = os.path.join(
        os.path.dirname(__file__),        # directory of THIS test file
        "test_comparison_results/",                 # subfolder for reference data
        "simulation_results2.pkl"          # the actual pickle file name
    )

    # If you haven't created the file yet, run your code once to generate it.
    # Then rename it or copy it into reference_data as the "golden" result.
    with open(reference_path, "rb") as f:
        golden_results = pickle.load(f)

    # 3) Compare dictionary keys
    assert new_results.keys() == golden_results.keys(), "Mismatch in dictionary keys."

    # 4) Compare each item in the dictionary
    for key in new_results.keys():
        # Compare arrays
        # If you want an exact match, use jnp.array_equal
        # For numerical tolerance, use jnp.allclose
        assert jnp.allclose(new_results[key], golden_results[key], rtol=1e-4, atol=1e-6), (
            f"Mismatch for key: {key}"
            )

    # If we reach here, everything matched!
    print("All results match expected data!")

@pytest.mark.simulation
def test_compare_simulation_results_only_passive_also_port_swap():
    """
    This test runs the simulation, then compares it to a known
    'golden' result stored in a pickle file.
    """

    # 1) Run the simulation to get new results
    new_results = run_simulation3()

    # 2) Load the reference (golden) data from disk
    #    Adjust the path to where your .pkl file actually is.
    reference_path = os.path.join(
        os.path.dirname(__file__),        # directory of THIS test file
        "test_comparison_results/",                 # subfolder for reference data
        "simulation_results3.pkl"          # the actual pickle file name
    )

    # If you haven't created the file yet, run your code once to generate it.
    # Then rename it or copy it into reference_data as the "golden" result.
    with open(reference_path, "rb") as f:
        golden_results = pickle.load(f)

    # 3) Compare dictionary keys
    assert new_results.keys() == golden_results.keys(), "Mismatch in dictionary keys."

    # 4) Compare each item in the dictionary
    for key in new_results.keys():
        # Compare arrays
        # If you want an exact match, use jnp.array_equal
        # For numerical tolerance, use jnp.allclose
        assert jnp.allclose(new_results[key], golden_results[key], rtol=1e-4, atol=1e-6), (
            f"Mismatch for key: {key}"
            )

    # If we reach here, everything matched!
    print("All results match expected data!")


@pytest.mark.simulation
def test_compare_simulation_results_small_time_frame():
    """
    This test runs the simulation, then compares it to a known
    'golden' result stored in a pickle file.
    """

    # 1) Run the simulation to get new results
    new_results = run_simulation4()

    # 2) Load the reference (golden) data from disk
    #    Adjust the path to where your .pkl file actually is.
    reference_path = os.path.join(
        os.path.dirname(__file__),        # directory of THIS test file
        "test_comparison_results/",                 # subfolder for reference data
        "simulation_results4.pkl"          # the actual pickle file name
    )

    # If you haven't created the file yet, run your code once to generate it.
    # Then rename it or copy it into reference_data as the "golden" result.
    with open(reference_path, "rb") as f:
        golden_results = pickle.load(f)

    # 3) Compare dictionary keys
    assert new_results.keys() == golden_results.keys(), "Mismatch in dictionary keys."

    # 4) Compare each item in the dictionary
    for key in new_results.keys():
                assert jnp.allclose(new_results[key], golden_results[key], rtol=1e-4, atol=1e-6), (
            f"Mismatch for key: {key}"
            )
    # If we reach here, everything matched!
    print("All results match expected data!")
