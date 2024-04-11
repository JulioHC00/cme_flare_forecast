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

### Datasets

Datasets may be created once the above is obtained. The logic to construct them can be found in `scripts/flare_and_cme_dataset.py` and follows what's described in the paper. To use one must specify the following arguments

- `T`: The forecasting horizon in hours. For the paper `T=24`
- `L`: The length of the observation period in hours. For the paper `L=24`
- `S`: The step by which the observation window is moved every step in hours. For the paper `S=0.2` (12 minutes, SHARPs cadence).
- `MIN_FLARE_CLASS=$MIN_FLARE_CLASS`: The minimum flare class to consider. For the paper this is X class, which corresponds to 30 (A=0, B=10, C=20, M=30, X=40)
- `ALLOW_OVERLAPS=$ALLOW_OVERLAPS`: Whether events can happen in an observation period. For the paper this is set to `False` i.e. we allow events during observation periods.
- `B`: If not allowing overlaps, this is the time after a flare has happened during which no observation is allowed in hours. For the paper `B=0` as we allow overlaps.

In `scripts/run_all` is the setup used to generate the datasets in the paper as well as the splits for cross-validation. The result is a list of individual data samples to be used for the model which details a start and end time. These times can be used to pull data from the processed SHARP keywords in the database as the input to the model. Additionally, several columns detail the label of that particular entry (e.g. `has_flare_above_threshold` is 1 if there's a flare above the `MIN_FLARE_CLASS` in the next `T` hours from the end time of the sample). This is all integrated with custom `PyTorch` datasets and dataloaders and there's no need to manually load the data to run the final training scripts. **We also note that although the cadence of the SHARP keywords is 12 minutes, the input to the model uses every other point to obtain a cadence of 24 minutes.**

### Data normalization

Normalization of the data is performed using Z-normalization for each feature

$$
  x \rightarrow \frac{x - \mu}{\sigma}
$$

However, the mean and standard deviation must be calculated using only the training data, using the same values for the validation data. This is to ensure that the model doesn't have access to any aspect of the validation data during training (leaking validation statistics into the training data). Thus, our custom datasets create a temporary file every time a model is trained with a particular set of splits for training and testing. In this file, data is normalized using only the training data statistics and is then accessed by the dataset. We chose this dynamic normalization instead of pre-generating these files for each fold for greater flexibility in the choice of splits to be used for training.

## Model

The model follows the transformer architecture as detailed in the paper. We make use of rotary positional encodings which are implemented in `src/models/parts/rotary_mha.py` using the implementation by [lucidrains/rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch). The basic transformer block is implemented in `src/models/parts/rotary_transformer_block.py` and the final model in `src/models/rotary_transformer.py`. We also make use of masked attention. This way, data entries with `IS_VALID=0` are not used at any point in the model and no attention is given to them. All configurations regarding the setup used to obtain the results of the paper can be found in the `configs` folder.
