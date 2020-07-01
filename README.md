# Lauderdale_etal_2020_PNAS
[![DOI](https://zenodo.org/badge/207910435.svg)](https://zenodo.org/badge/latestdoi/207910435)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/seamanticscience/Lauderdale_etal_2020_PNAS?color=1b3370)
![GitHub last commit](https://img.shields.io/github/last-commit/seamanticscience/Lauderdale_etal_2020_PNAS?color=f44323)
![GitHub License](https://img.shields.io/github/license/seamanticscience/Lauderdale_etal_2020_PNAS?color=ffa500)
<a href="https://doi.org/10.1073/pnas.1917277117"><img src="http://img.shields.io/badge/paper%20link-doi:10.1073%2Fpnas.1917277117-lightgrey.svg" alt="Link to paper at https://doi.org/10.1073/pnas.1917277117"></a>

Box model code, processing routines, and model ensemble data for the paper "Microbial feedbacks optimize ocean iron availability" by Jonathan Maitland Lauderdale, Rogier Braakman, Gaël Forget, Stephanie Dutkiewicz, and Michael J. Follows in Proceedings of the National Academy of Sciences.

Model parameters are set in `comdeck.h`, while the main model routines are in `boxmodel.f`. Compile the model with `f2py` to generate a python module:

```
>>f2py -c -m nutboxmod --verbose boxmodel.f fe_equil.f transport.f insol.f
```

To replot the figures from the paper, then run `boxmodel.py` or the Jupyter Notebook `boxmodel.ipynb`, which will read model ensemble data (10,000 members) from  `boxmodel_input.csv` and `boxmodel_output.csv`. Note:
1. Changing `RUNMODEL` to `True` will re-run the ensemble and possibly overwrite the input and output files - this has been known to take on the order of 2 days because it is not parallelized.
1. The model is compared to data from external servers. If you do not have files for World Ocean Atlas 2013 annual Nitrate or Phosphate climatologies or the GEOTRACES IDP 2017 v2, then the default values from the paper will be used.

Any questions or comments, please get in contact!
