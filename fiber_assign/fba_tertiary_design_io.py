#!/usr/bin/env python

import os
from glob import glob
import sys
import tempfile
import yaml
import fitsio
import healpy as hp
import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits
from astropy.coordinates import SkyCoord, Longitude
from astropy.visualization.wcsaxes import SphericalCircle
from astropy import units
from desimodel.focalplane.geometry import get_tile_radius_deg
from desimodel.footprint import tiles2pix, is_point_in_desi
from desisurvey.tileqa import lb2uv
from desitarget.io import read_targets_in_hp
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask, scnd_mask
from desitarget.targets import encode_targetid
from desitarget.geomask import match
from desitarget.targetmask import obsconditions
from desispec.tile_qa_plot import get_quantz_cmap
from desiutil import dust
from desisurvey.utils import yesno
from desiutil.log import get_logger
from fiberassign.scripts.assign import parse_assign, run_assign_full
from fiberassign.fba_tertiary_io import (
    get_targfn,
    get_priofn,
    assert_tertiary_targ,
    assert_tertiary_prio,
)
import fiberassign
from fiberassign.utils import assert_isoformat_utc
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator

log = get_logger()

def sigmoid(y):
    return 1/(1+np.exp(-y))
    
def success_prob(magnitude, exptime, A=-1.1701, B=28.1211):
    mag_transformed = magnitude - 1.25*np.log10(exptime/6000)
    success = sigmoid(A*mag_transformed + B)
    return success
    

def get_environ_settings_ref():
    """ """
    return {
        "fiberassign": "5.6.0",
        "DESIMODEL": "/global/common/software/desi/perlmutter/desiconda/current/code/desimodel/main",
        "SKYHEALPIXS_DIR": os.path.join(
            os.getenv("DESI_ROOT"), "target", "skyhealpixs", "v1"
        ),
        "SKYBRICKS_DIR": os.path.join(
            os.getenv("DESI_ROOT"), "target", "skybricks", "v3"
        ),
    }


def get_environ_settings_actual():
    """ """
    return {
        "fiberassign": fiberassign.__version__,
        "DESIMODEL": os.getenv("DESIMODEL"),
        "SKYHEALPIXS_DIR": os.getenv("SKYHEALPIXS_DIR"),
        "SKYBRICKS_DIR": os.getenv("SKYBRICKS_DIR"),
    }


def warning_environ_settings():
    """
    Print warning message to ensure that the overall environment is correctly set.

    Notes:
        Use the print() function instead of log.info() for clarity on the prompt.
    """
    # AR expected settings
    ref_settings = get_environ_settings_ref()
    print("")
    print(
        "========================================================================================"
    )
    print("Verify that you are running with these settings:")
    print("")
    print("source /global/cfs/cdirs/desi/software/desi_environment.sh 22.2")
    print("module swap fiberassign/{}".format(ref_settings["fiberassign"]))
    for key in ["DESIMODEL", "SKYHEALPIXS_DIR", "SKYBRICKS_DIR"]:
        print("export {}={}".format(key, ref_settings[key]))
    print("")
    print(
        "========================================================================================"
    )
    print("")


def assert_environ_settings():
    """
    Verify that some of the environment settings are correctly set.
    If there are some differences, ask to proceed or not.

    Notes:
        Use the print() function instead of log.info() for clarity on the prompt.
    """
    # AR expected settings
    ref_settings = get_environ_settings_ref()
    # AR actual settings
    act_settings = get_environ_settings_actual()
    diffkeys = []
    for key in ref_settings:
        print("actual {}: {}".format(key, act_settings[key]))
        if act_settings[key] != ref_settings[key]:
            diffkeys.append(key)
    print("")
    if len(diffkeys) > 0:
        warning_environ_settings()
        qstr = "the following settings are not as expected: {} => continue?".format(
            ", ".join(diffkeys)
        )
        proceed = yesno(qstr)
        if not proceed:
            print("answer = n => exit")
            sys.exit(1)


def print_tertiary_settings_help():
    """
    Print guidelines on how to do the settings in tertiary-config-PROGNUMPAD.yaml.

    Notes:
        Use the print() function instead of log.info() for clarity on the prompt.
    """
    print("")
    print("Guidelines for the settings in the tertiary-config-PROGNUMPAD.yaml file.")
    print(" - prognum:")
    print("    for development: 9999")
    print(
        "    for final design: pick the next-in-line number (check $DESI_ROOT/survey/fiberassign/special/tertiary)"
    )
    print(" - targdir:")
    print("    for development: typically DESIROOT/users/username/fiberassign-xxx/vx")
    print(
        "    for final design: DESIROOT/survey/fiberassign/special/tertiary/PROGNUMPAD"
    )
    print("    note the 'DESIROOT', which will be converted into $DESI_ROOT")
    print(" - ntile: number of tiles")
    print(" - tileid_start:")
    print("    for development: 999999 - ntile + 1")
    print(
        "    for final design: pick the next-in-line number (check https://desi.lbl.gov/trac/browser/data/tiles/trunk/083)"
    )
    print(
        " - rundate: pick a roundish date just before the last line of the latest desi-state file"
    )
    print(
        "    e.g. as of May 2023: /global/common/software/desi/perlmutter/desiconda/current/code/desimodel/main/data/focalplane/desi-state_2022-09-21T17:00:39+00:00.ecsv"
    )
    print(" - obsconds: 'BACKUP', 'BRIGHT', or 'DARK'")
    print(" - sbprof: 'ELG', 'BGS', or 'PSF'")
    print(" - goaltime: per-tile goal effective time in seconds")
    print(" - std_dtver: main desitarget catalog version to pick the standard stars")
    print("    '1.1.1' if inside the ls-dr9 footprint")
    print("    '2.2.0' if outside the ls-dr9 footprint (to pick gaia stars)")
    print(" - np_rand_seed: numpy random seed, usually set to 1234")
    print("")


def assert_tertiary_settings(mydict):
    """
    Asserts that the settings in tertiary-config-PROGNUMPAD.yaml make sense.

    Args:
        mydict: dictionary with the expected keys
    """
    keys = [
         "prognum",
        "targdir",
        "rosette_rad_list",
        "rosette_ntile_list",
        "tileid_start",
        "rundate",
        "obsconds",
        "sbprof",
        "goaltime",
        "std_dtver",
        "np_rand_seed",
        "field_ra",
        "field_dec",
        "target_list_fn",
    ]
    keyname_maxlen = np.max([len(key) for key in keys])
    # AR verify that at least those keys are there + print values
    mykeys = list(mydict.keys())
    for key in keys:
        assert key in mykeys
        log.info("{}\t= {}".format(key.ljust(keyname_maxlen), mydict[key]))
    # AR verify some values
    assert_isoformat_utc(mydict["rundate"])
    assert mydict["obsconds"] in ["BACKUP", "BRIGHT", "DARK"]
    assert mydict["sbprof"] in ["ELG", "BGS", "PSF"]
    # AR for PROGNUM=33, std_dtver was by mistake set to 1.1.0 instead of 1.1.1
    if mydict["prognum"] != 33:
        assert mydict["std_dtver"] in ["1.1.1", "2.2.0"]


def get_fn(prognum, case, targdir):
    """
    Returns the fiducial file name for the yaml, tiles, priorities, and targets files.

    Args:
        prognum: prognum (int)
        case: "yaml", "tiles", "priorities", "targets", or "log" (str)
        targdir: output dir (str)

    Returns:
        fn: output file name (str)
    """
    assert np.in1d(case, ["yaml", "tiles", "priorities", "targets", "log"])
    if case == "yaml":
        fn = os.path.join(targdir, "tertiary-config-{:04d}.yaml".format(prognum))
    if case == "tiles":
        fn = os.path.join(targdir, "tertiary-tiles-{:04d}.ecsv".format(prognum))
    elif case == "priorities":
        fn = get_priofn(prognum, targdir=targdir)
    elif case == "targets":
        fn = get_targfn(prognum, targdir=targdir)
    elif case == "log":
        fn = os.path.join(targdir, "desi_fba_tertiary_{:04d}.log".format(prognum))
    return fn


def get_tile_centers_rosette(field_ra, field_dec, npt=11, rad=0.12):
    """
    Returns the tile centers for a rosette.

    Args:
        field_ra: R.A. of the rosette center (float)
        field_dec: Dec. of the rosette center (float)
        npt (optional, defaults to 11): number of tiles (i.e. distributed over angles of 360/npt deg) (int)
        rad (optional, defaults to 0.12): radius (projected distance, in degree) where the tile centers are distributed over (float)

    Returns:
        ras: R.A. of the tile centers (np.array() of npt floats)
        decs: Dec. of the tile centers (np.array() of npt floats)

    Notes:
        The default npt and rad values mimick the DESI/SV3 rosette pattern.
    """
    # AR sanity check, in case...
    if (field_dec - rad <= -90) | (field_dec + rad >= 90):
        msg = "field_dec={}, rad={}: field_dec and rad should verify: -90 < field_dec - rad and field_dec + rad < 90".format(
            field_dec, rad
        )
        log.error(msg)
        raise ValueError(msg)
    # AR first get the projected offsets
    angs = np.pi / 2 + np.linspace(0, 2 * np.pi, npt + 1)[:-1]
    dras = rad * np.cos(angs)
    ddecs = rad * np.sin(angs)
    # AR correct for dec term
    dras /= np.cos(np.radians(field_dec + ddecs))
    # AR tile centers
    ras = field_ra + dras
    decs = field_dec + ddecs
    # AR wrap ra in 0, 360
    ras = Longitude(ras * units.deg).value
    return ras, decs


def get_tile_centers_grid(field_ra, field_dec, npt=5, rad=0.048):
    """
    Returns tile centers on a grid.

    Args:
        field_ra: R.A. of the rosette center (float)
        field_dec: Dec. of the rosette center (float)
        npt (optional, defaults to 5): number of tiles (i.e. distributed over angles of 360/npt deg) (int)
        rad (optional, defaults to 0.048): gridsize (projected distance, in degree) where the tile centers are distributed (float)

    Returns:
        ras: R.A. of the tile centers (np.array() of npt floats)
        decs: Dec. of the tile centers (np.array() of npt floats)

    Notes:
        The default rad value is approx. the DESI fiber patrol diameter (12[mm] * 70.4[um/arcsec]).
    """
    # AR handle npt=0...
    if npt == 0:
        return np.zeros(0), np.zeros(0)
    # AR pick a grid large enough
    n = np.ceil((np.sqrt(npt) - 1) / 2)
    vals = rad * np.arange(-2 * n, 2 * n + 1)
    assert vals.size**2 >= npt
    # AR built the (ra, dec) positions for this large grid
    ras, decs = [], []
    for i in range(len(vals)):
        ras += (field_ra + vals[i] / np.cos(np.radians(field_dec + vals))).tolist()
        decs += (field_dec + vals).tolist()
    ras, decs = np.array(ras), np.array(decs)
    # AR a couple of simple sanity checks
    msg = None
    if (decs.min() <= -90) | (decs.max() >= 90):
        msg = "npt={} or rad={} too large (min(decs)={:.1f}, max(decs)={:.1f})".format(
            npt, rad, decs.min(), decs.max()
        )
    if ras.max() - ras.min() >= 360:
        msg = "npt={} or rad={} too large (max(ras) - min(ras) = {:.1f})".format(
            npt, rad, ras.max() - ras.min()
        )
    if msg is not None:
        log.error(msg)
        raise ValueError(msg)
    # AR cut on npt points, with increasing distance from the field center
    field_c = SkyCoord(field_ra * units.degree, field_dec * units.degree, frame="icrs")
    cs = SkyCoord(ras * units.degree, decs * units.degree, frame="icrs")
    ii = cs.separation(field_c).value.argsort()
    ras, decs = ras[ii], decs[ii]
    ras, decs = ras[:npt], decs[:npt]
    # AR wrap ra in 0, 360
    ras = Longitude(ras * units.deg).value
    return ras, decs


def get_tile_ebv_meds(tile_ras, tile_decs, fprad=1.605, round=3):
    """
    For set of tiles, returns the median EBV over each tile.

    Args:
        tile_ras: R.A.'s of the tiles (float, single value or np.array())
        tile_decs: Dec.'s of the tiles (float, single value or np.array())
        fprad (optional, defaults to 1.605): considered tile radius (float)
        round (optional, defaults to 3): precision of output values (int)

    Returns:
        ebv_meds: median EBV value over each tile (float, single value or np.array())

    Notes:
        This is a simple editing of desisurvey.tileqa.add_info_fields().
        If round=None, no rounding is done.
    """
    # AR check single value vs. array
    singlevalue = False
    if isinstance(tile_ras, float) | isinstance(tile_decs, int):
        singlevalue = True
        tile_ras, tile_decs = np.array([tile_ras]), np.array([tile_decs])
    #
    nside = 512
    thetas, phis = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    ls, bs = np.degrees(phis), 90 - np.degrees(thetas)
    ebvs = dust.ebv(
        ls, bs, frame="galactic", mapdir=os.getenv("DUST_DIR") + "/maps", scaling=1
    )
    cs = SkyCoord(ra=tile_ras * units.deg, dec=tile_decs * units.deg, frame="icrs")
    ls, bs = cs.galactic.l.value, cs.galactic.b.value
    uvs = lb2uv(ls, bs)
    ebv_meds = np.zeros(len(tile_ras))
    for i in range(len(tile_ras)):
        ii = hp.query_disc(nside, uvs[i], np.radians(fprad))
        ebv_meds[i] = np.median(ebvs[ii])
    # AR rounding?
    if round is not None:
        ebv_meds = ebv_meds.round(round)
    # AR if single value, return single value
    if singlevalue:
        ebv_meds = ebv_meds[0]
    return ebv_meds


def read_yaml(fn):
    """
    Reads a .yaml file.

    Args:
        fn: yaml file name (string)

    Returns:
        mydict: the yaml file content

    Notes:
        Modifies "DESIROOT" to os.getenv("DESI_ROOT") in params["settings"]["targidir"]
    """
    with open(fn, "r") as file:
        mydict = yaml.safe_load(file)
    if "settings" in mydict:
        if "targdir" in mydict["settings"]:
            mydict["settings"]["targdir"] = mydict["settings"]["targdir"].replace(
                "DESIROOT", os.getenv("DESI_ROOT")
            )
    return mydict


def match_coord(ra1, dec1, ra2, dec2, search_radius=1.0, nthneighbor=1, verbose=True):
    """
    Match objects in (ra2, dec2) to (ra1, dec1).

    Args:
        ra1: R.A. of catalog 1 (np.array() of floats)
        dec1: Dec. of catalog 1 (np.array() of floats)
        ra2: R.A. of catalog 2 (np.array() of floats)
        dec2: Dec. of catalog 2 (np.array() of floats)
        search_radius (optional, defaults to 1.0.): search radius in arcsec (float)
        nthneighbor (optional, defaults to 1): find the n-th closest neighbor; 1 being the closest (int)
        one object in t2 is match to the same object in t1 (i.e. double match), only the closest pair
        is kept.
        verbose (optional, defaults to True): verbose (boolean)

    Returns:
        idx1: catalog 1 indices of matched objects (np.array() of floats)
        idx2: catalog 2 indices of matched objects (np.array() of floats)
        d2d: matched-pairs distances (in arcsec) (np.array() of floats)
        d_ra: matched-pairs projected angular separation in R.A. (in arcsec) (np.array() of floats)
        d_dec: matched-pairs projected angular separation in R.A. (in arcsec) (np.array() of floats)

    Notes:
        Slightly edited from https://desi.lbl.gov/svn/docs/technotes/targeting/target-truth/trunk/python/match_coord.py.
        Removed the keep_all_pairs option.
        If more than one object in catalog 2 is matched to the same object in catalog 1, keep only the closest match.
    """
    t1 = Table()
    t2 = Table()
    # protect the global variables from being changed by np.sort
    ra1, dec1, ra2, dec2 = list(map(np.copy, [ra1, dec1, ra2, dec2]))
    t1["ra"] = ra1
    t2["ra"] = ra2
    t1["dec"] = dec1
    t2["dec"] = dec2
    t1["id"] = np.arange(len(t1))
    t2["id"] = np.arange(len(t2))
    # Matching catalogs
    sky1 = SkyCoord(ra1 * units.degree, dec1 * units.degree, frame="icrs")
    sky2 = SkyCoord(ra2 * units.degree, dec2 * units.degree, frame="icrs")
    idx, d2d, d3d = sky2.match_to_catalog_sky(sky1, nthneighbor=nthneighbor)
    # This finds a match for each object in t2. Not all objects in t1 catalog are included in the result.

    # convert distances to numpy array in arcsec
    d2d = np.array(d2d.to(units.arcsec))
    matchlist = d2d < search_radius
    if np.sum(matchlist) == 0:
        if verbose:
            log.info("0 matches")
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([]),
            np.array([]),
            np.array([]),
        )
    t2["idx"] = idx
    t2["d2d"] = d2d
    t2 = t2[matchlist]
    init_count = np.sum(matchlist)
    # --------------------------------removing doubly matched objects--------------------------------
    # if more than one object in t2 is matched to the same object in t1, keep only the closest match
    t2.sort("idx")
    i = 0
    while i <= len(t2) - 2:
        if t2["idx"][i] >= 0 and t2["idx"][i] == t2["idx"][i + 1]:
            end = i + 1
            while end + 1 <= len(t2) - 1 and t2["idx"][i] == t2["idx"][end + 1]:
                end = end + 1
            findmin = np.argmin(t2["d2d"][i : end + 1])
            for j in range(i, end + 1):
                if j != i + findmin:
                    t2["idx"][j] = -99
            i = end + 1
        else:
            i = i + 1

    mask_match = t2["idx"] >= 0
    t2 = t2[mask_match]
    t2.sort("id")
    if verbose:
        log.info(("Doubly matched objects = %d" % (init_count - len(t2))))
    # -----------------------------------------------------------------------------------------
    if verbose:
        log.info(("Final matched objects = %d" % len(t2)))
    # This rearranges t1 to match t2 by index.
    t1 = t1[t2["idx"]]
    d_ra = (t2["ra"] - t1["ra"]) * 3600.0  # in arcsec
    d_dec = (t2["dec"] - t1["dec"]) * 3600.0  # in arcsec
    ##### Convert d_ra to actual arcsecs #####
    mask = d_ra > 180 * 3600
    d_ra[mask] = d_ra[mask] - 360.0 * 3600
    mask = d_ra < -180 * 3600
    d_ra[mask] = d_ra[mask] + 360.0 * 3600
    d_ra = d_ra * np.cos(t1["dec"] / 180 * np.pi)
    ##########################################
    return (
        np.array(t1["id"]),
        np.array(t2["id"]),
        np.array(t2["d2d"]),
        np.array(d_ra),
        np.array(d_dec),
    )


def print_samples_overlap(d, samples):
    """
    Prints the overlap matrix for samples inside a tertiary program.

    Args:
        d: Table() with boolean columns for each sample (Table() array)
        samples: sample names (list or np.array) of str)
    """
    # AR sanity check
    for sample in samples:
        assert sample in d.colnames
    #
    log.info("\t".join(["\t"] + samples.tolist()))
    for i1, sample1 in enumerate(samples):
        xs = [sample1]
        for i2, sample2 in enumerate(samples):
            if i2 <= i1:
                xs.append(str(((d[sample1]) & (d[sample2])).sum()))
        log.info("\t".join(xs))
    log.info("")


def create_tiles_table(
    tileids,
    tileras,
    tiledecs,
    program,
    in_desis=None,
    obsconditionss=None,
    radec_round=3,
):
    """
    Creates a tiles table formatted as input for fiberassign.

    Args:
        tileids: tileids (int)
        tileras: R.A. of the tile centers (float)
        tiledecs: Dec. of the tile centers (float)
        program: BACKUP, BRIGHT, or DARK (str)
        in_desis (optional, defaults to IN_DESI=1 for all tiles): IN_DESI 0 or 1 value for each tile (int)
        radec_round (optional, defaults to 3): rounding of the (ra, dec); set to None for no rounding (int)

    Returns:
        d: Table() array formatted as input for fiberassign

    Notes:
        tileids, tileras, tiledecs, in_desis, obsconditions should be lists or 1d-arrays.
        Default IN_DESI=1 should be regularly used.
        Note that setting IN_DESI=0 will result in fiberassign discarding the tile.
    """
    # AR sanity check
    assert program in ["BACKUP", "BRIGHT", "DARK"]
    # AR defaults
    if in_desis is None:
        in_desis = np.ones(len(tileids), dtype=int)
    # AR round ra, dec
    if radec_round is not None:
        log.info("round (tileras, tiledecs) to {} digits".format(radec_round))
        tileras, tiledecs = np.round(tileras, radec_round), np.round(
            tiledecs, radec_round
        )
    # AR build table
    d = Table()
    d["TILEID"] = tileids
    d["RA"], d["DEC"] = tileras, tiledecs
    d["PROGRAM"] = program
    d["IN_DESI"] = in_desis
    d["EBV_MED"] = get_tile_ebv_meds(tileras, tiledecs)
    d["DESIGNHA"] = 0
    return d


def create_empty_priority_dict():
    """
    Creates a dictionary with the requested keys for the tertiary-priorities-PROGNUM.ecsv file.

    Args:
        None

    Returns:
        myd: empty dictionary with the following keys:
            TERTIARY_TARGET, NUMOBS_DONE_MIN, NUMOBS_DONE_MAX, PRIORITY
    """
    myd = {
        key: []
        for key in ["TERTIARY_TARGET", "NUMOBS_DONE_MIN", "NUMOBS_DONE_MAX", "PRIORITY"]
    }
    return myd


def creates_priority_table(yamlfn):
    """
    Creates a priority table which can write to a tertiary-PROGNUM.ecsv,
        with the "usual" scheme that the priority is boosted by +1
        each time it is assigned.

    Args:
        yamlfn: .yamlfn with the requested infos (PRIORITY_INIT, NGOAL, CHECKER, FN) for each sample (str)

    Returns:
        d: Table() array with TERTIARY_TARGET, NUMOBS_DONE_MIN, NUMOBS_DONE_MAX, PRIORITY (Table() array)

    Notes:
        See e.g. $DESI_ROOT/survey/fiberassign/special/tertiary/0026/tertiary-config-0026.yaml for an example of yamlfn.
        This "usual" scheme is used for ~all tertiary programs so far.
        Note that this function only requires yamlfn to have (PRIORITY_INIT and NGOAL); CHECKER and FN are expected to be there for other functions.
    """
    # AR read the requested settings
    mydict = read_yaml(yamlfn)["samples"]
    # AR initiate a dictionary
    myd = create_empty_priority_dict()
    # AR loop over target samples
    for target in mydict:
        prio_init, ngoal = mydict[target]["PRIORITY_INIT"], mydict[target]["NGOAL"]
        for nmin in range(ngoal + 1):
            if nmin == ngoal:
                nmax, prio = 99, 1
            else:
                nmax, prio = nmin, prio_init + nmin
            myd["TERTIARY_TARGET"].append(target)
            myd["NUMOBS_DONE_MIN"].append(nmin)
            myd["NUMOBS_DONE_MAX"].append(nmax)
            myd["PRIORITY"].append(prio)
    # AR convert to Table()
    d = Table()
    for key in myd:
        d[key] = myd[key]
    return d


def subsample_targets_avail(d, prognum, targdir, rundate, ignore_samples=""):
    """
    Subsample the targets based on their expected best-chance to have NASSIGN>=NGOAL.

    Args:
        d: Table array with the targets
        prognum: tertiary PROGNUM (int)
        targdir: folder where the files are (str)
        rundate: yyyy-mm-ddThh:mm:ss+00:00 rundate for focalplane with UTC timezone formatting (str)
        ignore_samples (optional, defaults to ''): comma-separated list of samples
            (i.e. TERTIARY_TARGET) to ignore in this process (str)

    Returns:
        d: modified Table array, subsampled and also with possibly different row-ordering

    Notes:
        Credits to David s suggestion ([desi-survey 4055], 3/16/23, 11:28 AM).
        We suggest to not apply this process to the lowest-priority sample
            (i.e. set it in ignore_samples), as otherwise, the un-used fibers
            are then re-distributed to higher-priority targets which mostly
            already have NASSIGN=NGOAL; thus this creates a tail of
            NASSIGN>NGOAL targets for those higher-priority samples,
            which may not be desirable in general, as the goal often is
            to have a uniform sampling in EFFTIME for each sample.
        We sort the targets by samples of decreasing priorities
            and then just loop through the targets using the row index.
        For each sample, first shuffle rows to break any dependence
            of possible subsamples in the catalog construction.
            See emails with Arjun from 04-04-2023.
    """
    # AR samples, prio_inits, by decreasing prio_inits
    fn = get_fn(prognum, "yaml", targdir)
    mydict = read_yaml(fn)["samples"]
    samples = np.array(list(mydict.keys()))
    prio_inits = np.array([mydict[sample]["PRIORITY_INIT"] for sample in samples])
    ii = prio_inits.argsort()[::-1]
    samples, prio_inits = samples[ii], prio_inits[ii]

    # AR PRIORITY_INIT, NGOAL columns
    # AR    (we expect those not be here,
    # AR    but try to handle the case where they are here)
    # AR if they are not here, we temporarily add them
    # AR    and remove them at the end of the function
    temporary_keys = []
    for key in ["PRIORITY_INIT", "NGOAL"]:
        if key in d.colnames:
            for sample in samples:
                ii = np.where(d["TERTIARY_TARGET"] == sample)[0]
                assert np.all(d[key][ii] == mydict[sample][key])
        else:
            temporary_keys.append(key)
            d[key] = 0
            for sample in samples:
                ii = np.where(d["TERTIARY_TARGET"] == sample)[0]
                d[key][ii] = mydict[sample][key]

    # AR first sort by decreasing priority
    ii = np.zeros(0, dtype=int)
    for sample in samples:
        jj = np.where(d["TERTIARY_TARGET"] == sample)[0]
        ii = np.append(ii, jj)
    assert ii.size == len(d)
    d = d[ii]
    assert np.all(np.diff(d["PRIORITY_INIT"]) <= 0)
    log.info("input table re-ordered with samples by decreasing PRIORITY_INIT")

    # AR compute navail
    fn = get_fn(prognum, "tiles", targdir)
    tiles = Table.read(fn)
    d["FAKE_TARGETID"] = np.arange(len(d), dtype=int)
    temporary_keys.append("FAKE_TARGETID")
    d["AVAIL"], favails = get_avail(
        d["RA"], d["DEC"], tiles, rundate, tids=d["FAKE_TARGETID"], return_favails=True
    )
    d["NAVAIL"] = d["AVAIL"].sum(axis=1)
    log.info("NAVAIL computed")

    # AR loop through each sample (and deal with ignore_samples)
    log.info("loop through all samples: {}".format(samples))
    keep = np.zeros(len(d), dtype=bool)
    ii_checked = []
    for sample in samples:
        ii = np.where(d["TERTIARY_TARGET"] == sample)[0]
        log.info("found {} rows for TERTIARY_TARGET={}".format(ii.size, sample))
        if sample in ignore_samples.split(","):
            keep[ii] = True
            log.info(
                "keeping all rows for TERTIARY_TARGET={} as it is in ignore_samples={}".format(
                    sample, ignore_samples
                )
            )
        else:
            # AR randomize the rows to avoid any correlation between our approach
            # AR    and some possible way the catalog has been created
            ii = np.random.permutation(ii)
            for i in ii:
                tid, ngoal = d["FAKE_TARGETID"][i], d["NGOAL"][i]
                ntile = 0
                fibers = {}
                for tileid in tiles["TILEID"]:
                    sel = favails[tileid]["TARGETID"] == tid
                    ist = sel.sum() > 0
                    ntile += int(ist)
                    fibers[tileid] = favails[tileid]["FIBER"][sel]
                assert ntile <= d["NAVAIL"][i]
                if ntile >= ngoal:
                    keep[i] = True
                    nassign = 0
                    for tileid in tiles["TILEID"]:
                        if (len(fibers[tileid]) > 0) & (nassign < ngoal):
                            sel = favails[tileid]["FIBER"] != fibers[tileid][0]
                            favails[tileid] = favails[tileid][sel]
                            nassign += 1
        n_keep = np.in1d(ii, np.arange(len(d))[keep]).sum()
        log.info(
            "{}:\t{}/{}={:.0f}% kept after cutting on favails".format(
                sample, n_keep, ii.size, 100 * n_keep / ii.size
            )
        )
        ii_checked += ii.tolist()
        # log.info(str(np.sum([np.unique(favails[tileid]["FIBER"]).size for tileid in tiles["TILEID"]])))
    assert len(ii_checked) == np.unique(ii_checked).size
    assert len(ii_checked) == len(d)
    # AR cut
    d = d[keep]
    d.remove_columns(temporary_keys)
    return d


def get_avail(
    ras,
    decs,
    tiles,
    rundate,
    return_favails=False,
    tids=None,
    has=None,
    margin_poss=None,
    margin_petals=None,
    margin_gfas=None,
    workdir=None,
):
    """
    Args:
        ras: targets R.A. (np.array() of floats)
        decs: tagets Dec. (np.array() of floats)
        tiles: tiles table formatted for fiberassign (Table() array)
        rundate: yyyy-mm-ddThh:mm:ss+00:00 rundate for focalplane with UTC timezone formatting (str)
        return_favails (optional, defaults to False): return a dictionary with the fiberassign FAVAIL tables? (boolean)
        tids (optional, defaults to None, i.e. will create fake TARGETIDs): TARGETID values (np.array() of ints)
        has (optional, defaults to 0): design for the given Hour Angle in degrees (np.array() of floats)
        margin_poss (optional, defaults to 0.05): add margin (in mm) around positioner keep-out polygons (np.array() of floats)
        margin_petals (optional, defaults to 0.4): add margin (in mm) around petal-boundary keep-out polygons (np.array() of floats)
        margin_gfas (optional, defaults to 0.4): add margin (in mm) around GFA keep-out polygons (np.array() of floats)
        workdir (optional, defaults to tempfile.mkdtemp()): folder where the fiberassign work is done (str)

    Returns:
        isavail: (ntarg, ntile) array (booleans)
        favails (if requested): dictionary with the fiberassign FAVAIL tables

    Notes:
        See e.g. create_tiles_table() for an example of formatted tiles table.
        Will overwrite files in workdir, be careful if providing workdir.
        If tiles only has one element, isavail is array(ntarg), otherwise array(ntarg, ntile).
    """
    ntarg, ntile = len(ras), len(tiles)
    # AR defaults
    if tids is None:
        tids = np.arange(ntarg, dtype=int)
    if has is None:
        has = np.zeros(ntarg, dtype=float)
    if margin_poss is None:
        margin_poss = np.full(ntarg, 0.05, dtype=float)
    if margin_petals is None:
        margin_petals = np.full(ntarg, 0.4, dtype=float)
    if margin_gfas is None:
        margin_gfas = np.full(ntarg, 0.4, dtype=float)
    if workdir is None:
        workdir = tempfile.mkdtemp()
    # AR formatted target table (set OBSCONDITIONS and DESI_TARGET with dummy values)
    d = Table()
    d["TARGETID"] = tids
    d["RA"], d["DEC"] = ras, decs
    d["SUBPRIORITY"] = np.random.uniform(ntarg)
    d["OBSCONDITIONS"] = 1
    d["DESI_TARGET"] = 1
    d.write(os.path.join(workdir, "targets.fits"), overwrite=True)
    isavail = np.zeros((len(ras), len(tiles)), dtype=bool)
    # AR file name used to store individual tiles files
    tiles_fn = os.path.join(workdir, "tiles.fits")
    #
    favails = {}
    if len(tiles) == 1:
        isavail = np.zeros(ntarg, dtype=bool)
    else:
        isavail = np.zeros((ntarg, ntile), dtype=bool)
    #
    for i, tileid in enumerate(tiles["TILEID"]):
        # AR output fba file (clean beforehand, safe)
        fbafn = os.path.join(workdir, "fba-{:06d}.fits".format(tileid))
        if os.path.isfile(fbafn):
            os.remove(fbafn)
        # AR individual tile file
        tmptiles = create_tiles_table(
            [tileid], [tiles["RA"][i]], [tiles["DEC"][i]], "DARK"
        )
        tmptiles["OBSCONDITIONS"] = obsconditions.mask("DARK")
        tmptiles.write(tiles_fn, overwrite=True)
        # AR run fiberassign
        opts = [
            "--targets",
            os.path.join(workdir, "targets.fits"),
            "--overwrite",
            "--write_all_targets",
            "--dir",
            workdir,
            "--footprint",
            tiles_fn,
            "--rundate",
            rundate,
            "--ha",
            str(has[i]),
            "--margin-pos",
            str(margin_poss[i]),
            "--margin-petal",
            str(margin_petals[i]),
            "--margin-gfa",
            str(margin_gfas[i]),
        ]
        log.info(" ; ".join(opts))
        ag = parse_assign(opts)
        run_assign_full(ag)
        # AR read result
        favail = Table.read(fbafn, "FAVAIL")
        sel_avail = np.in1d(d["TARGETID"], favail["TARGETID"])
        if len(tiles) == 1:
            isavail[sel_avail] = True
        else:
            isavail[sel_avail, i] = True
        favails[tileid] = favail
    #
    if return_favails:
        return isavail, favails
    else:
        return isavail


def format_pmradec_refepoch(d):
    """
    Ensures some correct formatting of PMRA, PMDEC, REF_EPOCH.

    Args:
        d: Table array with PMRA, PMDEC, REF_EPOCH columns.

    Returns:
        d: same Table array, with PMRA, PMDEC, REF_EPOCH possibly reformatted/modified.
    """
    # AR first force dtype
    for key in ["PMRA", "PMDEC", "REF_EPOCH"]:
        d[key] = d[key].astype(np.float32)
    # AR then chase non-valid cases..
    sel = (d["PMRA"] == 0) & (d["PMDEC"] == 0)
    sel |= (~np.isfinite(d["PMRA"])) | (~np.isfinite(d["PMDEC"]))
    sel |= (d["PMRA"] == -999.0) | (d["PMDEC"] == -999.0)
    d["PMRA"][sel], d["PMDEC"][sel], d["REF_EPOCH"][sel] = 0.0, 0.0, 2015.5
    log.info(
        "force PMRA=PMDEC=0 and REF_EPOCH=2015.5 for {} rows with either PMRA=PMDEC=0, or non-finite values of PMRA,PMDEC, or PMRA,PMDEC=-999".format(
            sel.sum()
        )
    )
    return d


def finalize_target_table(d, yamlfn):
    """
    Executes the usual common final steps for the tertiary-targets-PROGNUM.fits table:
        - proper formatting of PMRA, PMDEC, REF_EPOCH
        - define TARGETID based on PROGNUM
        - define SUBPRIORITY
        - add CHECKER column
        - add header keywords (EXTNAME, FAPRGRM, OBSCONDS, SBPROF, GOALTIME)

    Args:
        d: Table array with the targets
        yamlfn: path to the tertiary-config-PROGNUM.yaml file

    Returns:
        d: same Table array with modifications listed above.
    """
    #
    mydict = read_yaml(yamlfn)["settings"]
    prognum = mydict["prognum"]
    obsconds, sbprof, goaltime = (
        mydict["obsconds"],
        mydict["sbprof"],
        mydict["goaltime"],
    )
    # AR sanity checks
    assert obsconds in ["DARK", "BRIGHT", "BACKUP"]
    assert sbprof in ["ELG", "BGS", "PSF"]
    # AR pmra, pmdec, ref_epoch
    d = format_pmradec_refepoch(d)
    # AR targetid and subpriority
    d["TARGETID"] = encode_targetid(
        release=8888, brickid=prognum, objid=np.arange(len(d))
    )
    d["SUBPRIORITY"] = np.random.uniform(size=len(d))
    # AR checker
    d["CHECKER"] = np.zeros(len(d), dtype="object")
    mydict = read_yaml(yamlfn)["samples"]
    for sample in mydict:
        sel = d["TERTIARY_TARGET"] == sample
        d["CHECKER"][sel] = mydict[sample]["CHECKER"]
    d["CHECKER"] = d["CHECKER"].astype(str)
    # AR header keywords
    d.meta["EXTNAME"] = "TARGETS"
    d.meta["FAPRGRM"] = "tertiary{}".format(prognum)
    d.meta["OBSCONDS"] = obsconds
    d.meta["SBPROF"] = sbprof
    d.meta["GOALTIME"] = goaltime
    return d


def assert_files(prognum, targdir):
    """
    Asserts that the tertiary-targets-PROGNUM.fits and tertiary-priorities-PROGNUM.ecsv
        are correctly formatted.

    Args:
        prognum: tertiary PROGNUM (int)
        targdir: folder where the files are (str)
    """
    targfn = get_fn(prognum, "targets", targdir)
    priofn = get_fn(prognum, "priorities", targdir)
    assert_tertiary_targ(prognum, targfn)
    assert_tertiary_prio(prognum, priofn, targfn)


def create_targets_assign(prognum, targdir):
    """
    Creates the tertiary-targets-PROGNUM-assign.fits file.

    Args:
        prognum: tertiary PROGNUM (int)
        targdir: folder where the files are (str)
    """
    # AR output file
    outfn = get_fn(prognum, "targets", targdir).replace(".fits", "-assign.fits")

    # AR yaml
    fn = get_fn(prognum, "yaml", targdir)
    mydict = read_yaml(fn)["samples"]
    goaltime = read_yaml(fn)["settings"]["goaltime"]

    # AR tiles
    fn = get_fn(prognum, "tiles", targdir)
    tiles = Table.read(fn)
    tileids = tiles["TILEID"]
    ntile = tileids.size

    # AR all targets
    fn = get_fn(prognum, "targets", targdir)
    d = Table.read(fn)
    # AR cut on the ones falling in the tiling
    fns = [
        os.path.join(targdir, "ToO-{:04d}-{:06d}.ecsv".format(prognum, tileid))
        for tileid in tileids
    ]
    tids = np.unique(
        vstack([Table.read(fn) for fn in fns], metadata_conflicts="silent")["TARGETID"]
    )
    sel = np.in1d(d["TARGETID"], tids)
    log.info(
        "cutting on {} / {} targets falling in the {} tiles".format(
            sel.sum(), len(d), ntile
        )
    )
    d = d[sel]

    # AR ngoal
    if "NGOAL" not in d.colnames:
        d["NGOAL"] = np.zeros(len(d), dtype=int)
        for sample in np.unique(d["TERTIARY_TARGET"]):
            sel = d["TERTIARY_TARGET"] == sample
            d["NGOAL"][sel] = mydict[sample]["NGOAL"]

    # AR now store avail/assign/fibers
    d["AVAIL"] = np.zeros((len(d), ntile), dtype=bool)
    d["FIBERS"] = np.array(["-" for x in range(len(d) * ntile)], dtype=object).reshape(
        (len(d), ntile)
    )
    d["ASSIGN"] = np.zeros((len(d), ntile), dtype=bool)
    for i in range(ntile):
        # AR filename
        tileidpad = "{:06d}".format(tileids[i])
        fn = os.path.join(
            targdir, tileidpad[:3], "fiberassign-{}.fits.gz".format(tileidpad)
        )
        # AR assign
        f = Table.read(fn, "FIBERASSIGN")
        assert np.unique(f["TARGETID"]).size == len(f)
        iid, iif = match(d["TARGETID"], f["TARGETID"])
        d["ASSIGN"][iid, i] = True
        d["FIBERS"][iid, i] = f["FIBER"][iif].astype(str)
        # AR avail
        tids = Table.read(fn, "TARGETS")["TARGETID"]
        d["AVAIL"][np.in1d(d["TARGETID"], tids), i] = True
    #
    d["NAVAIL"] = d["AVAIL"].sum(axis=1)
    d["NASSIGN"] = d["ASSIGN"].sum(axis=1)
    d["FIBERS"] = d["FIBERS"].astype(str)
    #
    d["EXP_TIME"] = d["NASSIGN"]*goaltime
    d["SUCCESS_PROB"] = success_prob(d["I_MAG"],d["EXP_TIME"])
    d.write(outfn)


def get_radeclim(tileras, tiledecs, rad=1.65):
    """
    Get the (R.A., Dec.) limits for plotting the tiles.

    Args:
        tileras: R.A. centers of the tiles (array of floats)
        tiledecs: Dec. centers of the tiles (array of floats)

    Returns:
        ralim: 2-elements tuple for the R.A. range.
        declim: 2-elements tuple for the Dec. range.

    Notes:
        TODO: handle the case where crossing R.A.=0.
    """
    ralim, declim = np.array([1e3, -1e3]), np.array([1e3, -1e3])
    for ra, dec in zip(tileras, tiledecs):
        ramin = ra - rad / np.cos(np.radians(dec))
        ramax = ra + rad / np.cos(np.radians(dec))
        decmin, decmax = dec - rad, dec + rad
        if ramin < ralim[0]:
            ralim[0] = ramin
        if ramax > ralim[1]:
            ralim[1] = ramax
        if decmin < declim[0]:
            declim[0] = decmin
        if decmax > declim[1]:
            declim[1] = decmax
    ralim, declim = ralim.round(2), declim.round(2)
    ralim = ralim[::-1]
    return ralim, declim



def plot_targets_assign(prognum, targdir):
    """
    Creates the tertiary-targets-PROGNUM-assign.fits file.

    Args:
        prognum: tertiary PROGNUM (int)
        targdir: folder where the files are (str)
    """

    # AR output file
    outfn = get_fn(prognum, "targets", targdir).replace(".fits", "-assign.pdf")

    # AR samples, priority_inits
    yamlfn = get_fn(prognum, "yaml", targdir)
    mydict = read_yaml(yamlfn)["samples"]
    samples = np.array(list(mydict.keys()))

    d = Table.read(outfn.replace(".pdf", ".fits"))

    # AR various infos...
    targfn = get_fn(prognum, "targets", targdir)
    hdr = fits.getheader(targfn, 1)
    tilesfn = get_fn(prognum, "tiles", targdir)
    tiles = Table.read(tilesfn)
    ntile = len(tiles)
    ralim, declim = get_radeclim(tiles["RA"], tiles["DEC"])

    # AR some settings
    clim = (0, ntile)
    cmap = get_quantz_cmap("jet", clim[1] - clim[0] + 1, 0, 1)
    bins = -0.5 + np.arange(clim[0], clim[1] + 2)
    cens = 0.5 * (bins[1:] + bins[:-1])

    # AR for the overall stats
    avail_nfiber = 4100 * ntile  # AR assume 4100 science fibers per tile
    used_nfiber = d["NASSIGN"].sum()
    sel = (d["NASSIGN"] > 0) & (d["NASSIGN"] < d["NGOAL"])
    short_nfiber = d["NASSIGN"][sel].sum()
    sel = d["NASSIGN"] >= d["NGOAL"]
    goal_nfiber = d["NGOAL"][sel].sum()
    sel = d["NASSIGN"] > d["NGOAL"]
    over_nfiber = d["NASSIGN"][sel].sum() - d["NGOAL"][sel].sum()
    myd = Table()
    myd.meta[
        "FIBAVAIL"
    ] = "# NFIBER_AVAIL = approx. {} (4100 per tile, {} tiles)".format(
        avail_nfiber, ntile
    )
    myd.meta["FIBUSED"] = "NFIBERS_USED = {} ({:.0f}%)".format(
        used_nfiber, 100 * used_nfiber / avail_nfiber
    )
    myd.meta["FIBSHORT"] = "NFIBERS_SHORTUSED = {} ({:.0f}%)".format(
        short_nfiber, 100 * short_nfiber / avail_nfiber
    )
    myd.meta["FIBGOAL"] = "NFIBERS_GOALUSED = {} ({:.0f}%)".format(
        goal_nfiber, 100 * goal_nfiber / avail_nfiber
    )
    myd.meta["FIBOVER"] = "NFIBERS_OVERUSED = {} ({:.0f}%)".format(
        over_nfiber, 100 * over_nfiber / avail_nfiber
    )
    for key in [
        "TARGET",
        "PRIORITY_INIT",
        "NGOAL",
        "NTARG",
        "NFIBERS",
        "FRAC_FIBERS_USED",
        "NOK",
        "FRACOK",
    ]:
        if key in ["TARGET"]:
            dtype = "object"
        if key in ["PRIORITY_INIT", "NGOAL", "NTARG", "NFIBERS", "NOK"]:
            dtype = int
        if key in ["FRAC_FIBERS_USED", "FRACOK"]:
            dtype = float
        myd[key] = np.zeros(len(samples) + 1, dtype=dtype)

    # AR now loop on all the samples
    with PdfPages(outfn) as pdf:
        #plot the tiles as circles and centers on the field
        targets = Table(fitsio.read(read_yaml(yamlfn)["settings"]["target_list_fn"]))
        target_ra = targets["RA"]
        target_dec = targets["DEC"]
        
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot()
        ax.scatter(target_ra,target_dec,marker=".",alpha=0.1,rasterized=True)
        ax.scatter(tiles["RA"], tiles["DEC"], marker=".",rasterized=False)
        
        tile_rad = get_tile_radius_deg()
        for t_ra, t_dec in zip(tiles["RA"],tiles["DEC"]):
            circle = SphericalCircle((t_ra*units.degree,t_dec*units.degree),tile_rad*units.degree)
            ax.plot(circle.get_xy()[:,0],circle.get_xy()[:,1],c="C1")
        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")
        ax.set_aspect('equal')
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    
        for it, sample in enumerate(["ALL"] + samples.tolist()):

            fig = plt.figure(figsize=(10,8))
            # gs = gridspec.GridSpec(1, 2, wspace=0.25, hspace=0.25)
            if sample == "ALL":
                istarg = np.ones(len(d), dtype=bool)
            else:
                istarg = d["TERTIARY_TARGET"] == sample
            nok = (d["NASSIGN"][istarg] >= d["NGOAL"][istarg]).sum()
            title = "{} ({}) : {} targets\nNtarg: {}, Nfiber: {} ({:.0f}%), Nassign>=Ngoal: {} ({:.0f}%)".format(
                hdr["FAPRGRM"],
                hdr["OBSCONDS"],
                sample,
                istarg.sum(),
                d["NASSIGN"][istarg].sum(),
                100 * d["NASSIGN"][istarg].sum() / d["NASSIGN"].sum(),
                nok,
                100 * nok / istarg.sum(),
            )
            log.info(title)
            s = 10
            if istarg.sum() > 1000:
                s = 5
            if istarg.sum() > 10000:
                s = 1

            # AR plot (ra, dec)
            # ax = fig.add_subplot(gs[0, 0])
            ax = fig.add_subplot()
            for sel, zorder in zip(
                [(istarg) & (d["NASSIGN"] == 0), (istarg) & (d["NASSIGN"] > 0)],
                [0, 1],
            ):
                sc = ax.scatter(
                    d["RA"][sel],
                    d["DEC"][sel],
                    c=d["NASSIGN"][sel],
                    s=s,
                    alpha=0.5,
                    zorder=zorder,
                    cmap=cmap,
                    vmin=clim[0],
                    vmax=clim[1],
                    rasterized=True,
                )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("R.A [deg]")
            ax.set_ylabel("Dec. [deg]")
            ax.set_xlim(ralim)
            ax.set_ylim(declim)
            ax.grid()
            cbar = plt.colorbar(sc)
            cbar.set_label("N assign")
            cbar.mappable.set_clim(clim)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()
            # AR plot NASSIGN histograms
            fig = plt.figure(figsize=(20, 6))
            # ax = fig.add_subplot(gs[0, 1])
            ax = fig.add_subplot()
            _ = ax.hist(
                d["NASSIGN"][istarg], bins=bins, density=True, histtype="stepfilled"
            )
            ns, counts = np.unique(d["NASSIGN"][istarg], return_counts=True)
            for n, count in zip(ns, counts):
                ax.text(
                    n,
                    0.95,
                    "Ntarg={}".format(count),
                    zorder=1,
                    rotation=90,
                    ha="center",
                    va="top",
                )
            if sample == "ALL":
                ngoal = -99
                prio_init = -99
            else:
                ngoal = mydict[sample]["NGOAL"]
                prio_init = mydict[sample]["PRIORITY_INIT"]
            if ngoal != -99:
                ax.plot([ngoal, ngoal], [0, 0.5], color="k", ls="--")
                ax.text(
                    ngoal,
                    0.52,
                    "Goal",
                    color="k",
                    ha="center",
                    va="bottom",
                    rotation="vertical",
                )

            # AR filling infos for the overall stats
            myd["TARGET"][it] = sample
            myd["PRIORITY_INIT"][it] = prio_init
            myd["NGOAL"][it] = ngoal
            myd["NTARG"][it] = istarg.sum()
            myd["NFIBERS"][it] = d["NASSIGN"][istarg].sum()
            myd["FRAC_FIBERS_USED"][it] = np.round(
                d["NASSIGN"][istarg].sum() / d["NASSIGN"].sum(), 2
            )
            myd["NOK"][it] = nok
            myd["FRACOK"][it] = np.round(nok / istarg.sum(), 2)

            # AR plot stuff
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Nassign")
            ax.set_ylabel("Norm. counts")
            ax.set_xlim(bins[0], bins[-1])
            ax.set_ylim(0, 1)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.grid()
            ax.set_axisbelow(True)
            axt = ax.twinx()
            fracs = [(d["NASSIGN"][istarg] >= n).mean() for n in cens]
            axt.plot(cens, fracs, "-o", color="r")
            axt.set_ylabel("Fraction with >= Nassign", color="r")
            axt.set_ylim(0, 1)
            axt.xaxis.set_major_locator(MultipleLocator(1))
            axt.yaxis.set_major_locator(MultipleLocator(0.1))
            axt.yaxis.label.set_color("r")
            axt.tick_params(axis="y", colors="r")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

    # AR
    myd["TARGET"] = myd["TARGET"].astype(str)
    myd.pprint_all()
    myd.write(outfn.replace(".pdf", "-summary.ecsv"),overwrite=True)


def get_main_primary_priorities(program):
    """
    Retrieve the simplified list of target classes and associated PRIORITY_INIT values
        from the DESI Main primary targets.

    Args:
        program: "DARK" or "BRIGHT" or "BACKUP" (str)


    Returns:
        names: (tertiary-adapted) names of the target classes (np.array() of str)
        initprios: PRIORITY_INIT values for names (np.array() of int)
        calib_or_nonstds: is the target class from calibration and other non-standard targets, like sky (np.array() of bool)

    Notes:
        The approach is to define a TERTIARY_TARGET class for each DESI Main
            primary target class.
        We parse the following masks: desi_mask, mws_mask, bgs_mask, scnd_mask.
        The names are built as "{prefix}_{target_class}", where prefix is desi, mws, bgs, scnd
        The observation scheme for the calib_or_nonstds=True is non-standard,
            so one may want to treat them in a custom way in some tertiary designs
            (note that the standard stars are independently picked in fba_launch anyway).
        The list of calib_or_nonstds=True DARK targets is:
            DESI_SKY,DESI_STD_FAINT,DESI_STD_WD,DESI_SUPP_SKY,DESI_NO_TARGET,DESI_NEAR_BRIGHT_OBJECT,DESI_SCND_ANY
            MWS_GAIA_STD_FAINT,MWS_GAIA_STD_WD
        The list of calib_or_nonstds=True BRIGHT targets is:
            DESI_SKY,DESI_STD_WD,DESI_STD_BRIGHT,DESI_SUPP_SKY,DESI_NO_TARGET,DESI_NEAR_BRIGHT_OBJECT,DESI_SCND_ANY
            MWS_GAIA_STD_WD,MWS_GAIA_STD_BRIGHT
        20250212: "BACKUP" enabled (used for for non-calibration tiles)
    """

    assert program in ["DARK", "BRIGHT", "BACKUP"]

    # AR we discard some names
    black_names = {}
    for prefix, mask in zip(
        ["DESI", "MWS", "BGS", "SCND"],
        [desi_mask, mws_mask, bgs_mask, scnd_mask],
    ):
        black_names[prefix] = [
            key for key in mask.names() if key[-5:] in ["NORTH", "SOUTH"]
        ]

    # AR loop on masks
    names, initprios, calib_or_nonstds = [], [], []
    for prefix, mask in zip(
        ["DESI", "MWS", "BGS", "SCND"],
        [desi_mask, mws_mask, bgs_mask, scnd_mask],
    ):
        mask_names = [name for name in mask.names() if name not in black_names[prefix]]
        for name in mask_names:
            if program in mask[name].obsconditions:
                names.append("{}_{}".format(prefix, name))
                initprios.append(
                    mask[name].priorities["UNOBS"]
                    if "UNOBS" in mask[name].priorities
                    else None
                )
                calib_or_nonstds.append("UNOBS" not in mask[name].priorities)

    names = np.array(names)
    initprios = np.array(initprios)
    calib_or_nonstds = np.array(calib_or_nonstds)

    return names, initprios, calib_or_nonstds


def get_main_primary_targets(
    program,
    field_ras,
    field_decs,
    radius=None,
    dtver=None,
    remove_stds=False,
):
    """
    Read the DESI Main primary targets inside a set of field positions.

    Args:
        program: "DARK" or "BRIGHT" or "BACKUP" (str)
        field_ras: R.A. center of the calibration field (float or np.array() of floats)
        field_decs: Dec. center of the calibration field (float or np.array() of floats)
        radius (optional, defaults to the DESI tile radius): radius in deg. to query around
            the field centers (float)
        dtver (optional, defaults to 2.2.0 for BACKUP, 1.1.1 for BRIGHT,DARK): main desitarget catalog version (str)
        remove_stds (optional, defaults to False): remove STD_{BRIGHT,FAINT} targets (bool)

    Returns:
        d: a Table() structure with the regular desitarget.io functions formatting.

    Notes:
        There is no high-level desitarget.io routines doing that, because we want to allow
            the possibility to query several tiles with a custom radius.
        The remove_stds argument is in case one wants to remove standard stars,
            as those can be independently picked up by fba_launch.
        20250212: BACKUP enabled (used for for non-calibration tiles)
    """

    assert program in ["DARK", "BRIGHT", "BACKUP"]

    if program == "BACKUP":
        photcat = "gaiadr2"
        std_names = ["GAIA_STD_BRIGHT", "GAIA_STD_FAINT", "GAIA_STD_WD"]
        std_key, std_mask = "MWS_TARGET", mws_mask
        if dtver is None:
            dtver = "2.2.0"
    else:
        photcat = "dr9"
        std_names = ["STD_BRIGHT", "STD_FAINT", "STD_WD"]
        std_key, std_mask = "DESI_TARGET", desi_mask
        if dtver is None:
            dtver = "1.1.1"

    # AR default to desi tile radius
    if radius is None:
        radius = get_tile_radius_deg()

    # AR desitarget folder
    hpdir = os.path.join(
        os.getenv("DESI_TARGET"),
        "catalogs",
        photcat,
        dtver,
        "targets",
        "main",
        "resolve",
        program.lower(),
    )
    log.info("reading targets from {}".format(hpdir))

    # AR get the file nside
    fn = sorted(
        glob(os.path.join(hpdir, "targets-{}-hp-*fits".format(program.lower())))
    )[0]
    nside = fitsio.read_header(fn, 1)["FILENSID"]

    # AR get the list of pixels overlapping the tiles
    tiles = Table()
    if isinstance(field_ras, float):
        tiles["RA"], tiles["DEC"] = [field_ras], [field_decs]
    else:
        tiles["RA"], tiles["DEC"] = field_ras, field_decs
    pixs = tiles2pix(nside, tiles=tiles, radius=radius)

    # AR read the targets in these healpix pixels
    d = Table(read_targets_in_hp(hpdir, nside, pixs, quick=True))
    d.meta["TARG"] = hpdir

    # AR cut on actual radius
    sel = is_point_in_desi(tiles, d["RA"], d["DEC"], radius=radius)
    d = d[sel]

    # AR remove STD_BRIGHT,STD_FAINT?
    # AR    as those will also be included in the fba_launch
    # AR    call with the --targ_std_only argument
    if remove_stds:
        reject = np.zeros(len(d), dtype=bool)
        for name in std_names:
            reject |= (d[std_key] & std_mask[name]) > 0
        log.info("removing {} names={} targets".format(reject.sum(), ",".join(std_names)))
        d = d[~reject]

    return d


def get_main_primary_targets_names(
    targ,
    program,
    tertiary_targets=None,
    initprios=None,
    do_ignore_gcb=False,
    keep_calib_or_nonstds=False,
):
    """
    Get the TERTIARY_TARGET values for the DESI Main primary targets.

    Args:
        targ: Table() array with the DESI Main primary targets;
            typically output of get_main_primary_targets() (Table() array)
        program: "DARK" or "BRIGHT" (str)
        tertiary_targets (optional, defaults to get_main_primary_priorities()):
            (tertiary-adapted) names of the target classes (np.array() of str)
        initprios (optional, defaults to get_main_primary_priorities()): PRIORITY_INIT values for names (np.array() of int)
        do_ignore_gcb (optional, defaults to False): ignore the GC_BRIGHT targets for the sanity check; this
            is used for PROGNUM=8 (bool)
        keep_calib_or_nonstds (optional, defaults to False): keep calib_or_nonstds targets (bool)

    Returns:
        names: the TERTIARY_TARGET values (np.array() of str)
    """

    assert program in ["DARK", "BRIGHT", "BACKUP"]

    # AR tertiary names + priorities
    if tertiary_targets is not None:
        assert initprios is not None
    else:
        assert initprios is None
        tertiary_targets, initprios, calib_or_nonstds = get_main_primary_priorities(
            program
        )
    if not keep_calib_or_nonstds:
        sel = np.array([_ is not None for _ in initprios])
        tertiary_targets, initprios = tertiary_targets[sel], initprios[sel]
        log.info("keep_calib_or_nonstds=False -> restrict to tertiary_targets={}".format(tertiary_targets))

    # AR TERTIARY_TARGET
    # AR loop on np.unique(tertiary_targets), for backwards-reproducibility
    dtype = "|S{}".format(np.max([len(x) for x in tertiary_targets]))
    names = np.array(["-" for i in range(len(targ))], dtype=dtype)
    myprios = -99 + np.zeros(len(targ))
    ii = tertiary_targets.argsort()
    for tertiary_target, initprio in zip(tertiary_targets[ii], initprios[ii]):
        if tertiary_target[:5] == "DESI_":
            name, dtkey, mask = tertiary_target[5:], "DESI_TARGET", desi_mask
        if tertiary_target[:4] == "BGS_":
            name, dtkey, mask = tertiary_target[4:], "BGS_TARGET", bgs_mask
        if tertiary_target[:4] == "MWS_":
            name, dtkey, mask = tertiary_target[4:], "MWS_TARGET", mws_mask
        if tertiary_target[:5] == "SCND_":
            name, dtkey, mask = tertiary_target[5:], "SCND_TARGET", scnd_mask
        sel = ((targ[dtkey] & mask[name]) > 0) & (myprios < initprio)
        # AR assuming all secondaries have OVERRIDE=False
        # AR i.e. a primary target will keep its properties if it
        # AR also is a secondary
        if dtkey == "SCND_TARGET":
            sel &= targ["DESI_TARGET"] == desi_mask["SCND_ANY"]
        myprios[sel] = initprio
        names[sel] = tertiary_target

    # AR verify that all rows got assigned a TERTIARY_TARGET
    sel = names.astype(str) == "-"
    assert (names.astype(str) == "-").sum() == 0

    # AR verify that we set for PRIORITY_INIT the same values as in desitarget
    # AR except for STD_BRIGHT,STD_FAINT
    # AR except for WD_BINARIES_BRIGHT, WD_BINARIES_DARK...
    # AR            where the above assumption is false.. (their scnd priority
    # AR            1998 is higher than their primary_priority, 1400-1500
    # AR            from MWS_BROAD,MWS_MAIN_BLUE..
    # AR except for (6) GC_BRIGHT in COSMOS/BRIGHT (same reason as above)
    myprios = -99 + np.zeros(len(targ))
    for t, p in zip(tertiary_targets, initprios):
        myprios[names.astype(str) == t] = p
    ignore_std = np.in1d(names.astype(str), ["DESI_STD_BRIGHT", "DESI_STD_FAINT"])
    log.info(
        "priority_check: ignoring {} DESI_STD_BRIGHT|DESI_STD_FAINT".format(
            ignore_std.sum()
        )
    )
    ignore_wd = (targ["SCND_TARGET"] & scnd_mask["WD_BINARIES_BRIGHT"]) > 0
    ignore_wd |= (targ["SCND_TARGET"] & scnd_mask["WD_BINARIES_DARK"]) > 0
    log.info(
        "priority check: ignoring {} WD_BINARIES_BRIGHT|WD_BINARIES_DARK".format(
            ignore_wd.sum()
        )
    )
    ignore = (ignore_std) | (ignore_wd)
    if do_ignore_gcb:
        ignore_gcb = (targ["SCND_TARGET"] & scnd_mask["GC_BRIGHT"]) > 0
        log.info("priority check: ignoring {} GC_BRIGHT".format(ignore_gcb.sum()))
        ignore |= ignore_gcb
    sel = (myprios != targ["PRIORITY_INIT"]) & (~ignore)

    # AR sanity check
    # AR do not raise an error, as according to the usage it could be ok
    # AR    (because this as been tested only on calibration fields)
    if sel.sum() > 0:
        msg = "the set PRIORITY_INIT differs from the desitarget one for {}/{} rows; please check!".format(
            sel.sum(), len(targ)
        )
        log.warning(msg)

    return names
