Self-Instantiated Stress–Energy: A Predictive Framework for Matter and Metric in Closed-System Cosmology. Modeling
==============================================================================================================

Overview
--------

We present a technical validation for complete, testable cosmological framework in which the Universe begins in a globally coherent quantum state and the first local act of decoherence inside the closed system creates its own environment.
This repository contains a Python-based research module designed to explore quantum entanglement and decoherence as drivers for the expansion and materialization of field modes in closed quantum systems.
The modeling focuses on a theoretical cosmological model where the stress-energy tensor emerges dynamically from decoherence processes.

Requirements
------------

To ensure computational accuracy and reproducibility, the following dependencies must be installed:

- `Python <https://www.python.org/>`_: >= 3.8
- `NumPy <https://numpy.org/>`_: >= 2.3.3
- `Matplotlib <https://matplotlib.org/>`_: >= 3.10.7

Installation
------------

Clone this repository and set up a Python virtual environment as follows:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/worktif/matter_framework.git
   cd matter_framework

   # Create and activate a virtual environment
   python3 -m venv env
   source env/bin/activate

   # Install dependencies
   pip install .

Usage
-----

To run the modeling, use the provided ``analysis.py`` script in ``src`` folder.
Below is an example of initializing a simulation and producing results:

.. code-block:: python

   from src.modeling.physics_model import CosmologyParams, compute_background_and_spectra, k_phys_to_code, plot_charts

   # Define cosmological parameters
   P = CosmologyParams(
       Omega_r0=9.2e-5, Omega_m0=0.315, Omega_L0=0.684,
       Omega_ent0=1e-5,  # smaller, so the background doesn't roll off too quickly into de Sitter
       epsilon=0.01,  # change w_ent more slowly
       DeltaN=50.0,  # long plateau => almost const ε_H
       N0=-3.0,  # act before pivot, so the pivot lies on a plateau
       c_s_scalar=1.0,
       n0=0.4, k0=k_phys_to_code(0.05), sigma_ln_k=0.4,
       Gamma_over_H=3.0, A_ring=0.02, phi_ring=0.0
   )

   # Finalize internal parameter scaling
   P.finalize()

   # Run the simulation
   result = compute_background_and_spectra(
       P,
       N_min=-10.0, N_max=6.0, nN=4001,
       kmin=5e-4, kmax=1.0, nk=256
   )

   # Plot and save simulation results
   plot_charts(result)