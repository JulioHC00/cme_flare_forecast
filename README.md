# TITLE OF PAPER

This repository contains the data, scripts to process the data, scripts to train the model and the model weights from the paper (**PAPER HERE**). Below we provide extensive details of how the data was processed, normalized and used to train the models referred to in the paper.

## Data processing

The main data source for the features used in forecasting events are SHARP keywords which we obtain from SWAN-SF **SAY WHERE**. Event times are obtained from the GOES flare list (for flares), also from **SAY WHERE**, Solar Demon (for dimmings) **LINK** and our own CME source region catalogue. Then, the data is processed as follows.

### Feature processing

Data from SWAN is loaded into an sqlite database. For convenience this is already done and can be found in the file `data/features.db`, table `SWAN`. To the original columns of this table we add a `IS_VALID` boolean flag indicating whether that particular entry contains valid data. The validity is based on meeting the following two criteria:

1. `IS_TMFI`, which indicates whether the magnetic field features can be trusted (see **CITE** for details of what this means). Its value is 1 when they can be trusted and 0 otherwise. We require a value of 1.
2. No SHARP keyword must have a missing value

If both these conditions are met then `IS_VALID=1`. Else `IS_VALID=0` and that row won't be used in training.

Additionally, we calculate a series of event history parameters as in **CITE**. These consist of three columns `dec`, `hist` and `hist1d` corresponding to an exponentially decaying history that exponentially decays to 1/e over 12 hours (see **CITE** for details), the total number of events and the number of events over the last day. These three columns are calculated independently for flares of class B, C, M and X (e.g. columns `Mdec`, `Mhis`, `Mhis1d`) along with CMEs. Additionally, columns `Edec` and `logEdec` are similar to the individual flare class columns but events are weighted by their X-ray flux (see **CITE**).

The extra history features along with the `IS_VALID` flare are calculated using `scripts/pad_features.py`.
