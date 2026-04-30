# dynamics of Morris and Lecar

Code for comparing the ODEs defined by Morris and Lecar in 1981. The Hodgkin-Huxley model is a simplified version of the model defined by Hodgkin and Huxley in 1952. The Morris-Lecar model, defined by Morris and Lecar, include the assumption that Ca2+ is fast, and so M=M_infty.

ODE_hodgkinhuxley.py gives the function for the Hodgkin-Huxley model, as well as some key functions used in both.

ODE_morrislecar.py contains the functions used exclusively for the Morris Lecar model.

generate_graphs.ipynb is a Jupyter notebook contating example plots for phase planes and state variable plots comparing the two models.
