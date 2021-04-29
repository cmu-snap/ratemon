""" Creates training, validation, and test data. """

import math
from os import path

import numpy as np
from numpy.lib import recfunctions
import torch

import defaults
import features
import models
import utils


def get_dataloaders(args, net):
    """
    Builds training, validation, and test sets, which are returned as
    dataloaders.
    """
    out_dir = args["out_dir"]
    dat_flp = path.join(
        out_dir,
        defaults.DATA_PREFIX + utils.args_to_str(
            args, order=sorted(defaults.DEFAULTS.keys()), which="data") +
        ".npz")
    scl_prms_flp = path.join(out_dir, "scale_params.json")
    # Check for the presence of both the data and the scaling
    # parameters because the resulting model is useless without the
    # proper scaling parameters.
    if (not args["regen_data"] and path.exists(dat_flp) and
            path.exists(scl_prms_flp)):
        print("Found existing data!")
        trn, val, tst = utils.load_parsed_data(dat_flp)
    else:
        print("Regenerating data...")
        trn, val, tst, scl_prms = get_bulk_data(args, net)
        # Save the processed data so that we do not need to process it again.
        utils.save_parsed_data(dat_flp, trn, val, tst)
        # Save scaling parameters. We always need to save the scaling parameters,
        # because the trained model cannot be used without them.
        utils.save_scl_prms(args["out_dir"], scl_prms)
    return create_dataloaders(args, trn, val, tst)


def get_bulk_data(args, net):
    """
    Loads bulk training, validation, and test data splits from disk. Returns
    a tuple of the form:
        ( (training dataloader, validation dataloader, test dataloader),
          scaling parameters )
    """
    data_dir = args["data_dir"]
    sample_frac = args["sample_percent"] / 100
    trn, val, tst = [
        get_split(data_dir, name, sample_frac, net)
        for name in ["train", "val", "test"]]

    # Validate scaling groups.
    assert trn[3] == val[3] == tst[3], "Scaling groups do not agree."

    if isinstance(net, models.HistGbdtSklearnWrapper):
        # The HistGbdtSklearn model does not require feature scaling because it
        # is a decision tree.
        scl_prms = np.zeros((0,))
    else:
        # Scale input features. Do this here instead of in process_exp() because
        # all of the features must be scaled using the same parameters. trn[0]
        # is the training input data. trn[3] is the scaling groups.
        trn[0], scl_prms = scale_fets(trn[0], trn[3], args["standardize"])

    return trn[:3], val[:3], tst[:3], scl_prms


def get_split(data_dir, name, sample_frac, net):
    """ Constructs a split from many subsplits on disk. """
    # Load the split's subsplits.
    subsplits = utils.load_subsplits(data_dir, name)
    # Optionally select a fraction of each subsplit. We always use all of the
    # test split.
    if name in {"train", "val"} and sample_frac < 1:
        subsplits = [
            subsplit[:math.ceil(subsplit.shape[0] * sample_frac)]
            for subsplit in subsplits]
    # Merge the subsplits into a split.
    split = np.concatenate(subsplits)
    # Optionally shuffle the split.
    if name == "train" and len(subsplits) > 1:
        np.random.default_rng().shuffle(split)
    # Extract features from the split.
    return extract_fets(split, name, net)


def extract_fets(dat, split_name, net):
    """
    Extracts net's the input and output features from dat. Returns a tuple of
    the form:
        (dat_in, dat_out, dat_extra, scaling groups).
    """
    # Split each data matrix into two separate matrices: one with the input
    # features only and one with the output features only. The names of the
    # columns correspond to the feature names in in_spc and out_spc.
    assert net.in_spc, f"{net.name}: Empty in spec."
    num_out_fets = len(net.out_spc)
    # This is not a strict requirement from a modeling point of view,
    # but is assumed to make data processing easier.
    assert num_out_fets == 1, \
        (f"{net.name}: Out spec must contain a single feature, but actually "
         f"contains: {net.out_spc}")

    # Remove samples where the ground truth output is unknown.
    len_before = dat.shape[0]
    dat = dat[dat[list(net.out_spc)] != -1][0]
    removed = dat.shape[0] - len_before
    if removed > 0:
        print(
            f"Removed {removed} rows with unknown out_spc from split "
            f"\"{split_name}\".")

    dat_in = recfunctions.repack_fields(dat[list(net.in_spc)])
    dat_out = recfunctions.repack_fields(dat[list(net.out_spc)])
    # Create a structured array to hold extra data that will not be used as
    # features but may be needed by the training/testing process.
    dtype_extra = (
        # The "raw" entry is the unconverted out_spc.
        [("raw",
          [typ for typ in dat.dtype.descr if typ[0] in net.out_spc][0][1])] +
        [typ for typ in dat.dtype.descr if typ[0] in features.EXTRA_FETS])
    dat_extra = np.empty(shape=dat.shape, dtype=dtype_extra)
    dat_extra["raw"] = dat_out
    for typ in features.EXTRA_FETS:
        dat_extra[typ] = dat[typ]
    dat_extra = recfunctions.repack_fields(dat_extra)

    is_dt = isinstance(net, models.HistGbdtSklearnWrapper)
    if not is_dt:
        # Verify that there are no NaNs or Infs in the data.
        for fet in dat_in.dtype.names:
            assert (not (
                np.isnan(dat_in[fet]).any() or
                np.isinf(dat_in[fet]).any())), \
                ("Warning: NaNs or Infs in input feature for split "
                 f"\"{split_name}\": {fet}")
        assert (not (
            np.isnan(dat_out[features.LABEL_FET]).any() or
            np.isinf(dat_out[features.LABEL_FET]).any())), \
            f"Warning: NaNs or Infs in ground truth for split \"{split_name}\"."

    if dat_in.shape[0] > 0:
        # Convert all instances of -1 (feature value unknown) to either the mean for
        # that feature or NaN.
        bad_fets = []
        for fet in dat_in.dtype.names:
            invalid = dat_in[fet] == -1
            if invalid.all():
                bad_fets.append(fet)
                continue
            dat_in[fet][invalid] = (
                float("NaN") if is_dt
                else np.mean(dat_in[fet][np.logical_not(invalid)]))
            assert (dat_in[fet] != -1).all(), \
                f"Found \"-1\" in split \"{split_name}\" feature: {fet}"
        assert not bad_fets, \
            (f"Features in split \"{split_name}\" contain only \"-1\" "
             f"({len(bad_fets)}): {bad_fets}")

    # Convert output features to class labels.
    dat_out = net.convert_to_class(dat_out)

    # Verify data.
    assert dat_in.shape[0] == dat_out.shape[0], \
        "Input and output should have the same number of rows."
    # Find the uniques classes in the output features and make sure that they
    # are properly formed. Assumes that dat_out is a structured numpy array
    # containing a single column specified by features.LABEL_FET.
    for cls in np.unique(dat_out[features.LABEL_FET]).tolist():
        assert 0 <= cls < net.num_clss, f"Invalid class: {cls}"

    # Transform the data as required by this specific model.
    # TODO: Refactor this to be compatible with bulk data splits.
    # dat_in, dat_out, dat_extra, scl_grps = net.modify_data(
    #     exp, dat_in, dat_out, dat_extra, sequential=sequential)
    scl_grps = list(range(len(dat_in.dtype.names)))

    return dat_in, dat_out, dat_extra, scl_grps


def scale_fets(dat, scl_grps, standardize=False):
    """
    Returns a copy of dat with the columns normalized. If standardize
    is True, then the scaling groups are normalized to a mean of 0 and
    a variance of 1. If standardize is False, then the scaling groups
    are normalized to the range [0, 1]. Also returns an array of shape
    (number of unique scaling groups, 2) where row i contains the scaling
    parameters of column i in dat. If standardize is True, then the
    scaling parameters are the mean and standard deviation of that
    column's scaling group. If standardize is False, then the scaling
    parameters are the min and max of that column's scaling group.
    """
    fets = dat.dtype.names
    assert fets is not None, \
        f"The provided array is not structured. dtype: {dat.dtype.descr}"
    assert len(scl_grps) == len(fets), \
        f"Invalid scaling groups ({scl_grps}) for dtype ({dat.dtype.descr})!"

    # Determine the unique scaling groups.
    scl_grps_unique = set(scl_grps)
    # Create an empty array to hold the min and max values (i.e.,
    # scaling parameters) for each scaling group.
    scl_grps_prms = np.empty((len(scl_grps_unique), 2), dtype="float64")
    # Function to reduce a structured array.
    rdc = (lambda fnc, arr:
           fnc(np.array(
               [fnc(arr[fet]) for fet in arr.dtype.names if fet != ""])))
    # Determine the min and the max of each scaling group.
    for scl_grp in scl_grps_unique:
        # Determine the features in this scaling group.
        scl_grp_fets = [fet for fet_idx, fet in enumerate(fets)
                        if scl_grps[fet_idx] == scl_grp]
        # Extract the columns corresponding to this scaling group.
        fet_values = dat[scl_grp_fets]
        # Record the min and max of these columns.
        scl_grps_prms[scl_grp] = [
            np.mean(utils.clean(fet_values))
            if standardize else rdc(np.min, fet_values),
            np.std(utils.clean(fet_values))
            if standardize else rdc(np.max, fet_values)
        ]

    # Create an empty array to hold the min and max values (i.e.,
    # scaling parameters) for each column (i.e., feature).
    scl_prms = np.empty((len(fets), 2), dtype="float64")
    # Create an empty array to hold the rescaled features.
    new = np.empty(dat.shape, dtype=dat.dtype)
    # Rescale each feature based on its scaling group's min and max.
    for fet_idx, fet in enumerate(fets):
        # Look up the parameters for this feature's scaling group.
        prm_1, prm_2 = scl_grps_prms[scl_grps[fet_idx]]
        # Store this min and max in the list of per-column scaling parameters.
        scl_prms[fet_idx] = np.array([prm_1, prm_2])
        fet_values = dat[fet]
        if standardize:
            # prm_1 is the mean and prm_2 is the standard deviation.
            scaled = (
                # Handle the rare case where the standard deviation is
                # 0 (meaning that all of the feature values are the
                # same), in which case return an array of zeros.
                np.zeros(
                    fet_values.shape, dtype=fet_values.dtype) if prm_2 == 0
                else (fet_values - prm_1) / prm_2)
        else:
            # prm_1 is the min and prm_2 is the max.
            scaled = (
                # Handle the rare case where the min and the max are
                # the same (meaning that all of the feature values are
                # the same.
                np.zeros(
                    fet_values.shape, dtype=fet_values.dtype) if prm_1 == prm_2
                else utils.scale(
                    fet_values, prm_1, prm_2, min_out=0, max_out=1))
        new[fet] = scaled

    return new, scl_prms


def create_dataloaders(args, trn, val, tst):
    """
    Creates dataloaders for the training, validation, and test data.

    args: Arguments.
    trn: Training data.
    val: Validation data.
    tst: Test data.

    trn, val, and test must be tuples of the form:
        (dat_in, dat_out, dat_extra)
    """
    fets = trn[0].dtype.names

    dat_trn_in, dat_trn_out = [utils.clean(dat) for dat in trn[:2]]
    dat_val_in, dat_val_out = [utils.clean(dat) for dat in val[:2]]
    dat_tst_in, dat_tst_out = [utils.clean(dat) for dat in tst[:2]]
    dat_trn_extra = trn[2]
    dat_val_extra = val[2]
    dat_tst_extra = tst[2]

    bch_trn = args["train_batch"]
    bch_tst = args["test_batch"]
    # Do not do bch_trn None check here (see comment below)!
    if bch_tst is None:
        bch_tst = dat_tst_in.shape[0]

    # Create the dataloaders.
    dataset_trn = utils.Dataset(fets, dat_trn_in, dat_trn_out, dat_trn_extra)
    return (
        # Train dataloader.
        torch.utils.data.DataLoader(
            dataset_trn,
            batch_sampler=utils.BalancedSampler(
                dataset_trn, bch_trn, drop_last=False,
                drop_popular=args["drop_popular"]))
        if args["balance"]
        else torch.utils.data.DataLoader(
            dataset_trn,
            # Do not calculate bch_trn above (similarly to bch_tst) because the
            # BalancedSampler has special handling for the case where bch_trn is
            # None.
            batch_size=dat_trn_in.shape[0] if bch_trn is None else bch_trn,
            # Ordinarily, shuffle should be True. But we shuffle the training
            # data in prepare_data.py, so we do not need to do so again here.
            shuffle=False, drop_last=False),
        # Validation dataloader.
        torch.utils.data.DataLoader(
            utils.Dataset(fets, dat_val_in, dat_val_out, dat_val_extra),
            batch_size=bch_tst, shuffle=False, drop_last=False),
        # Test dataloader.
        torch.utils.data.DataLoader(
            utils.Dataset(fets, dat_tst_in, dat_tst_out, dat_tst_extra),
            batch_size=bch_tst, shuffle=False, drop_last=False))
