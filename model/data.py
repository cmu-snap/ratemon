""" Creates training, validation, and test data. """

import numpy
from numpy.lib import recfunctions
import torch

import defaults
import models
import utils


def get_dataloaders(args, net, dat=None, save_data=True):
    """
    Builds training, validation, and test dataloaders.

    If args["sims"] and dat are both None, then bulk data splits are loaded. See
    get_data_from_sims() and get_bulk_data() for details about specific cases.

    net: The model.
    args: Dictionary of configuration info.
    dat: Manually-loaded data. A list where each entry is a Numpy array
        corresponding to one simulation.
    save_data: Boolean indicating whether to save the resulting processed data.
        Only applies if bulk data splits are *not* used.
    """
    ldrs, scl_prms = (
        get_bulk_data(args) if (args["sims"] is None and dat is None) else
        get_data_from_sims(args, net, dat))
    print(f"Number of input features: {len(ldrs[0].dataset.fets)}")


    # TODO: Save the data as well.

    # Save the processed data so that we do not need to process it again.
    if save_data:
        utils.save(dat_flp, dat_in, dat_out, dat_extra)

    # Save scaling parameters. We always need to save the scaling parameters,
    # even if save_data == False, because the trained model cannot be used
    # without them.
    utils.save_scl_prms(args["out_dir"], scl_prms)
    return ldrs


def get_bulk_data(args):
    """
    Loads bulk training, validation, and test data splits from disk. Returns
    a tuple of the form:
        ( (training dataloader, validation dataloader, test dataloader),
          scaling parameters )

    args: Dictionary of configuration info.
    """
    data_dir = args["data_dir"]
    trn, val, tst = [
        extract_fets(utils.load_split(data_dir, name), net)
        for name in ["train", "val", "test"]]
    # Select a fraction of the training data.
    num_pkts = math.ceil(num_pkts * args["train_prc"] / 100)]
    trn = (dat[:num_pkts] for dat in trn)

    # TODO: Scale input data.

    # TODO: Convert output data to classes.

    return (
        create_dataloaders(
            net, args["train_batch"], args["test_batch"], trn, val, tst),
        scl_prms)


def get_data_from_sims(args, net, dat=None, save_data=True):
    """
    Parses simulation files and converts them into training data. Returns
    a tuple of the form:
        ( (training dataloader, validation dataloader, test dataloader),
          scaling parameters )

    If the output directory already contains parsed data, then it is loaded.

    Otherwise, if dat is not None (i.e., contains manually-loaded data), then
    that data is used.

    Otherwise, if args["sims"] contains a list of paths to simulations, then
    those simulations are parsed.

    Otherwise, simulations from the configured data directory are parsed.
    """
    out_dir = args["out_dir"]
    dat_flp = path.join(out_dir, "data.npz")
    scl_prms_flp = utils.get_scl_prms_flp(out_dir)

    if (not args["regen_data"] and path.exists(dat_flp) and
            path.exists(scl_prms_flp)):
        # Parsed data already exsists, so load it.
        #
        # In the above if-statement, we check for the presence of both
        # the data and the scaling parameters because the resulting
        # model is useless without the proper scaling parameters.
        print("Found existing data!")
        dat_in, dat_out, dat_extra = utils.load_parsed_data(dat_flp)
    else:
        # The parsed data is not already stored on disk, so we need to
        # regenerate it.
        print("Regenerating data...")

        if dat is None:
            # Manually-loaded data was not provided, so we need to parse
            # simulations files.

            # Either use specific simulations, so load simulations
            # from the specified data directory.
            sims = args["sims"]
            if not sims:
                dat_dir = args["data_dir"]
                sims = [
                    path.join(dat_dir, sim) for sim in sorted(os.listdir(dat_dir))]
            if SHUFFLE:
                # Set the random seed so that multiple parallel instances of
                # this script see the same random order.
                utils.set_rand_seed()
                random.shuffle(sims)
            num_sims = args["num_sims"]
            if num_sims is not None:
                num_sims_actual = len(sims)
                assert num_sims_actual >= num_sims, \
                    (f"Insufficient simulations. Requested {num_sims}, but only "
                    f"{num_sims_actual} available.")
                sims = sims[:num_sims]

            dat_all, sims = load_sims(args, net, sims)

        else:
            # Use the manually-loaded data.
            dat_all, sims = dat

        dat_in, dat_out, dat_extra = validate_and_merge(dat_all)


        # Scale input features. Do this here instead of in load_sim()
        # because all of the features must be scaled using the same
        # parameters.
        dat_in, prms_in = scale_fets(dat_in, scl_grps, args["standardize"])

        # Save the processed data so that we do not need to process it again.
        if save_data:
            utils.save(dat_flp, dat_in, dat_out, dat_extra)

    # Split the data into training, validation, and test loaders.
    return split_data(
        net, dat_in, dat_out, dat_extra, args["train_batch"],
        args["test_batch"]), prms_in


def load_sims(args, net, sims):
    tot_sims = len(sims)
    print(f"Loading {tot_sims} simulations.")

    # Prepare temporary output directory. The output of parsing each
    # simulation it written to disk instead of being transfered
    # between processes because sometimes the data is too large for
    # Python to send between processes.
    tmp_dir = args["tmp_dir"]
    if tmp_dir is None:
        tmp_dir = args["out_dir"]
    if not path.isdir(tmp_dir):
        print(
            ("Temporary directory does not exist. Creating it: "
             f"{tmp_dir}"))
        os.makedirs(tmp_dir)

    # Parse simulations.
    sims_args = [
        (idx, tot_sims, net, sim, tmp_dir, args["warmup_percent"],
         args["keep_percent"])
        for idx, sim in enumerate(sims)]
    if defaults.SYNC or args["sync"]:
        dat_all = [load_sim(*sim_args) for sim_args in sims_args]
    else:
        with multiprocessing.Pool() as pol:
            # Each element of dat_all corresponds to a single simulation.
            dat_all = pol.starmap(load_sim, sims_args)
    # Throw away results from simulations that could not be parsed.
    dat_all = [dat for dat in dat_all if dat is not None]
    print(f"Discarded {tot_sims - len(dat_all)} simulations!")
    assert dat_all, "No valid simulations found!"
    dat_all, sims = zip(*dat_all)
    return [utils.load_tmp_file(flp) for flp in dat_all]


def load_sim(idx, total, net, sim_flp, tmp_dir, warmup_prc, keep_prc,
                sequential=False):
    """
    Loads and processes data from a single simulation.

    For logging purposes, "idx" is the index of this simulation amongst "total"
    simulations total. Uses "net" to determine the relevant input and output
    features. "sim_flp" is the path to the simulation file. The parsed results
    are stored in "tmp_dir". Drops the first "warmup_prc" percent of packets.
    Of the remaining packets, only "keep_prc" percent are kept. See
    utils.save_tmp_file() for the format of the results file.

    Returns the path to the results file and a descriptive utils.Sim object.
    """
    sim, dat = utils.load_sim(
        sim_flp, msg=f"{idx + 1:{f'0{len(str(total))}'}}/{total}")
    if dat is None:
        return None

    # Drop the first few packets so that we consider steady-state behavior only.
    dat = dat[math.floor(dat.shape[0] * warmup_prc / 100):]
    dat_in, dat_out, dat_extra = extract_fets(
        dat, net, num_flws=(sim.unfair_flws + sim.fair_flws))

    # If the results contains NaNs or Infs, then discard this
    # simulation.
    def has_non_finite(arr):
        for fet in arr.dtype.names:
            if not np.isfinite(arr[fet]).all():
                print(
                    f"    Simulation {sim_flp} has NaNs of Infs in feature "
                    f"{fet}")
                return True
        return False
    if has_non_finite(dat_in) or has_non_finite(dat_out):
        return None

    # Verify data.
    assert dat_in.shape[0] == dat_out.shape[0], \
        f"{sim_flp}: Input and output should have the same number of rows."
    # Find the uniques classes in the output features and make sure
    # that they are properly formed. Assumes that dat_out is a
    # structured numpy array containing a column named "class".
    for cls in set(dat_out["class"].tolist()):
        assert 0 <= cls < net.num_clss, f"Invalid class: {cls}"

    # Transform the data as required by this specific model.
    dat_in, dat_out, dat_extra, scl_grps = net.modify_data(
        sim, dat_in, dat_out, dat_extra, sequential=sequential)

    # Select a fraction of the data.
    if keep_prc != 100:
        num_rows = dat_in.shape[0]
        num_to_pick = math.ceil(num_rows * keep_prc / 100)
        idxs = np.random.random_integers(0, num_rows - 1, num_to_pick)
        dat_in = dat_in[idxs]
        dat_out = dat_out[idxs]
        dat_extra = dat_extra[idxs]

    # To avoid errors with sending large matrices between processes,
    # store the results in a temporary file.
    dat_flp = path.join(tmp_dir, f"{path.basename(sim_flp)[:-4]}_tmp.npz")
    utils.save_tmp_file(
        dat_flp, dat_in, dat_out, dat_extra, scl_grps)
    return dat_flp, sim


def validate_and_merge(dat_all):
    """
    Validates and provided list of data from many simulations, and merges it
    into a single array. The data is shuffled during this process.

    Returns a tuple of the form: (dat_in, dat_out, dat_extra)

    dat_all: A list where each element is a Numpy array corresponding to on
        simulation.
    """
    # Validate data.
    dim_in = None
    dtype_in = None
    dim_out = None
    dtype_out = None
    dim_extra = None
    dtype_extra = None
    scl_grps = None
    for dat_in, dat_out, dat_extra, scl_grps_cur in dat_all:
        dim_in_cur = len(dat_in.dtype.names)
        dim_out_cur = len(dat_out.dtype.names)
        dim_extra_cur = len(dat_extra.dtype.names)
        dtype_in_cur = dat_in.dtype
        dtype_out_cur = dat_out.dtype
        dtype_extra_cur = dat_extra.dtype
        if dim_in is None:
            dim_in = dim_in_cur
        if dim_out is None:
            dim_out = dim_out_cur
        if dim_extra is None:
            dim_extra = dim_extra_cur
        if dtype_in is None:
            dtype_in = dtype_in_cur
        if dtype_out is None:
            dtype_out = dtype_out_cur
        if dtype_extra is None:
            dtype_extra = dtype_extra_cur
        if scl_grps is None:
            scl_grps = scl_grps_cur
        assert dim_in_cur == dim_in, \
            f"Invalid input feature dim: {dim_in_cur} != {dim_in}"
        assert dim_out_cur == dim_out, \
            f"Invalid output feature dim: {dim_out_cur} != {dim_out}"
        assert dim_extra_cur == dim_extra, \
            f"Invalid extra data dim: {dim_extra_cur} != {dim_extra}"
        assert dtype_in_cur == dtype_in, \
            f"Invalud input dtype: {dtype_in_cur} != {dtype_in}"
        assert dtype_out_cur == dtype_out, \
            f"Invalid output dtype: {dtype_out_cur} != {dtype_out}"
        assert dtype_extra_cur == dtype_extra, \
            f"Invalid extra data dtype: {dtype_extra_cur} != {dtype_extra}"
        assert (scl_grps_cur == scl_grps).all(), \
            f"Invalid scaling groups: {scl_grps_cur} != {scl_grps}"
    assert dim_in is not None, "Unable to compute input feature dim!"
    assert dim_out is not None, "Unable to compute output feature dim!"
    assert dim_extra is not None, "Unable to compute extra data dim!"
    assert dtype_in is not None, "Unable to compute input dtype!"
    assert dtype_out is not None, "Unable to compute output dtype!"
    assert dtype_extra is not None, "Unable to compute extra data dtype!"
    assert scl_grps is not None, "Unable to compte scaling groups!"

    # Build combined feature lists.
    dat_in, dat_out, dat_extra, _ = zip(*dat)
    # Stack the arrays.
    dat_in = np.concatenate(dat_in, axis=0)
    dat_out = np.concatenate(dat_out, axis=0)
    dat_extra = np.concatenate(dat_extra, axis=0)

    # Convert all instances of -1 (feature value unknown) to the mean for
    # that feature.
    bad_fets = []
    for fet in dat_in.dtype.names:
        fet_values = dat_in[fet]
        # If all values for this feature are -1, then skip it.
        if (fet_values == -1).all():
            bad_fets.append(fet)
            continue
        dat_in[fet] = np.where(
            fet_values == -1,
            # Calculate the mean of the valid feature values.
            np.mean(fet_values[np.where(fet_values != -1)]),
            fet_values)
        # Make sure that there are no -1's remaining.
        assert (dat_in[fet] != -1).all(), \
            f"Still found \"-1\" in feature: {fet}"
    assert not bad_fets, f"Features contain only \"-1\": {bad_fets}"

    return dat_in, dat_out, dat_extra


def extract_fets(dat, net, num_flws=None):
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
    dat_in = recfunctions.repack_fields(dat[net.in_spc])
    dat_out = recfunctions.repack_fields(dat[net.out_spc])
    # Create a structured array to hold extra data that will not be
    # used as features but may be needed by the training/testing
    # process.
    raw_dtype = [typ for typ in dat.dtype.descr if typ[0] in net.out_spc][0][1]
    dtype = ([("raw", raw_dtype), ("num_flws", "int32")] +
             [typ for typ in dat.dtype.descr if typ[0] in defaults.EXTRA_FETS])
    dat_extra = np.empty(shape=dat.shape, dtype=dtype)
    dat_extra["raw"] = dat_out
    if "num_flws" in dat.dtype.names:
        dat_extra["num_flws"] = dat["num_flws"]
    else:

        dat_extra["num_flws"].fill(num_flws)
    # dat_extra = recfunctions.repack_fields(dat_extra)
    # Convert output features to class labels.
    dat_out = net.convert_to_class(sim, dat_out)
    return dat_in, dat_out, dat_extra


def scale_fets(dat, scl_grps, standardize=False):
    """
    Returns a copy of dat with the columns normalized. If standardize
    is True, then the scaling groups are normalized to a mean of 0 and
    a variance of 1. If standardize is False, then the scaling groups
    are normalized to the range [0, 1]. Also returns an array of shape
    (dat_all[0].shape[1], 2) where row i contains the scaling
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


def split_data(net, dat_in, dat_out, dat_extra, bch_trn, bch_tst,
               use_val=False):
    """
    Divides the input and output data into training, validation, and
    testing sets and constructs data loaders.
    """
    print("Creating train/val/test data...")

    fets = dat_in.dtype.names
    # Keep track of the dtype of dat_extra so that we can recreate it
    # as a structured array.
    extra_dtype = dat_extra.dtype.descr
    # Destroy columns names to make merging the matrices easier. I.e.,
    # convert from structured to regular numpy arrays.
    dat_in = utils.clean(dat_in)
    dat_out = utils.clean(dat_out)
    dat_extra = utils.clean(dat_extra)
    # Shuffle the data to ensure that the training, validation, and
    # test sets are uniformly sampled. To shuffle all of the arrays
    # together, we must first merge them into a combined matrix.
    num_cols_in = dat_in.shape[1]
    merged = np.concatenate(
        (dat_in, dat_out, dat_extra), axis=1)
    np.random.shuffle(merged)
    dat_in = merged[:, :num_cols_in]
    dat_out = merged[:, num_cols_in]
    # Rebuilding dat_extra is more complicated because we need it to
    # be a structed array (for ease of use).
    num_exps = dat_in.shape[0]
    dat_extra = np.empty((num_exps,), dtype=extra_dtype)
    num_cols = merged.shape[1]
    num_cols_extra = num_cols - (num_cols_in + 1)
    extra_names = dat_extra.dtype.names
    num_cols_extra_expected = len(extra_names)
    assert num_cols_extra == num_cols_extra_expected, \
        (f"Error while reassembling \"dat_extra\". {num_cols_extra} columns "
         f"does not match {num_cols_extra_expected} expected columns: "
         f"{extra_names}")
    for name, merged_idx in zip(extra_names, range(num_cols_in + 1, num_cols)):
        dat_extra[name] = merged[:, merged_idx]

    # 50% for training, 20% for validation, 30% for testing.
    num_val = int(round(num_exps * 0.2)) if use_val else 0
    num_tst = int(round(num_exps * 0.3))
    print((f"    Data - train: {num_exps - num_val - num_tst}, val: {num_val}, "
           f"test: {num_tst}"))
    # Validation.
    dat_val_in = dat_in[:num_val]
    dat_val_out = dat_out[:num_val]
    dat_val_extra = dat_extra[:num_val]
    # Testing.
    dat_tst_in = dat_in[num_val:num_val + num_tst]
    dat_tst_out = dat_out[num_val:num_val + num_tst]
    dat_tst_extra = dat_extra[num_val:num_val + num_tst]
    # Training.
    dat_trn_in = dat_in[num_val + num_tst:]
    dat_trn_out = dat_out[num_val + num_tst:]
    dat_trn_extra = dat_extra[num_val + num_tst:]

    return create_dataloaders(
        net, bch_trn, bch_tst,
        trn=(dat_trn_in, dat_trn_out, dat_trn_extra),
        val=(dat_val_in, dat_val_out, dat_val_extra),
        tst=(dat_tst_in, dat_tst_out, dat_tst_extra))


def create_dataloaders(net, fets, bch_trn, bch_tst, trn, val, tst):
    """
    Creates dataloaders for the training, validation, and test data.

    net: The model.
    bch_trn: The training batch size.
    bch_tst: The test bach size.
    trn: Training data.
    val: Validation data.
    tst: Test data.

    trn, val, and test must be tuples of the form:
        (dat_in, dat_out, dat_extra)
    """
    dat_trn_in, dat_trn_out, dat_trn_extra = trn
    dat_val_in, dat_val_out, dat_val_extra = val
    dat_tst_in, dat_tst_out, dat_tst_extra = tst

    # Create the dataloaders.
    dataset_trn = utils.Dataset(fets, dat_trn_in, dat_trn_out, dat_trn_extra)
    return (
        # Training dataloader.
        (torch.utils.data.DataLoader(
            dataset_trn, batch_size=bch_tst, shuffle=True, drop_last=False)
         if isinstance(net, models.SvmSklearnWrapper)
         else torch.utils.data.DataLoader(
             dataset_trn,
             batch_sampler=utils.BalancedSampler(
                 dataset_trn, bch_trn, drop_last=False))),
        # Validation dataloader.
        (torch.utils.data.DataLoader(
            utils.Dataset(fets, dat_val_in, dat_val_out, dat_val_extra),
            batch_size=bch_tst, shuffle=False, drop_last=False)
         if use_val else None),
        # Test dataloader.
        torch.utils.data.DataLoader(
            utils.Dataset(fets, dat_tst_in, dat_tst_out, dat_tst_extra),
            batch_size=bch_tst, shuffle=False, drop_last=False)
    )
