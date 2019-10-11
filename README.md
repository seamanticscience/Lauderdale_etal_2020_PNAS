# Lauderdale_ligand_iron_microbe_feedback
Box model, processing routines, and large model ensemble data for ligand-iron-microbe feedback paper

Model parameters are set in `comdeck.h`, while the main model routines are in `boxmodel.f`. Compile the model with `f2py` to generate a python module:

```
>>f2py -c -m nutboxmod --verbose boxmodel.f fe_equil.f transport.f insol.f
```

To replot the figures from the paper, then run `boxmodel.py`, which will read model ensemble data (10,000 members) from  `boxmodel_input.csv` and `boxmodel_output.csv`. Note:
1. Changing `RUNMODEL` to `True` will re-run the ensemble and possibly overwrite the input and output files - this has been known to take on the order of 2 days because it is not parallelized.
1. The model is compared to data from external servers. If you do not have files for World Ocean Atlas 2013 annual Nitrate or Phosphate climatologies or the GEOTRACES IDP 2017 v2, then the default values from the paper will be used.

Any questions or comments, please get in contact!
