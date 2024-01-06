# Simphony Documentation

Simphony allows you to define photonic circuits, then run fast simulations on
them, all in Python.

- Simphony is free and open-source
- Runs on Windows, MacOS, and Linux
- Uses a SPICE-like method for defining photonic circuits
- Subnetwork growth algorithms, giving 20x speedup over other photonic modeling
  software
- Includes libraries for circuit components (known as models)
- Provides a simple framework for defining new models 

**To install Simphony**, simply use the following in a Python 3 environment:

```bash
pip install simphony
```

:::{note} 
We recommend installing two libraries,
[matplotlib](https://matplotlib.org/) and
[SiPANN](https://sipann.readthedocs.io/en/latest/ ), alongside Simphony.
Matplotlib provides a way to visualize the results from your simulations, and
SiPANN provides additional models for use in your circuits. View the links for
installation instructions and find out more. 
:::

If you are using a machine with a GPU, you should follow the instructions for
[JAX installation](https://jax.readthedocs.io/en/latest/installation.html) in 
order to get the dependencies required to use the GPU version of JAX, instead 
of just the CPU version, which is installed by default.

**To get started using Simphony**, check out the
[Introduction](tutorials/intro). Tutorials and API references are accessible
through the sidebar navigation.

Simphony is primarily developed and maintained by members of
[CamachoLab](https://camacholab.byu.edu) at Brigham Young University. Feedback
is welcome: if you find errors or have suggestions for the Simphony project,
let us know by raising an issue on
[GitHub](https://github.com/BYUCamachoLab/simphony). If you want to contribute,
even better! See [Contributing](dev/contributing) to learn how.

**Citing this work**

::::{tab-set}

:::{tab-item} Text
> S. Ploeg, H. Gunther and R. M. Camacho, "Simphony: An Open-Source Photonic 
> Integrated Circuit Simulation Framework," in Computing in Science & 
> Engineering, vol. 23, no. 1, pp. 65-74, 1 Jan.-Feb. 2021, doi: 10.1109/MCSE.2020.3012099.
:::

:::{tab-item} BibTeX
```bibtex
@article{Ploeg2021,
  author="Ploeg, Sequoia and Gunther, Hyrum and Camacho, Ryan M.",
  journal="Computing in Science \& Engineering", 
  title="Simphony: An Open-Source Photonic Integrated Circuit Simulation Framework", 
  year=2021,
  volume="23",
  number="1",
  pages="65--74",
  doi="10.1109/MCSE.2020.3012099"
}
```
:::

::::
