#!/usr/bin/env python

import os
from pathlib import Path
import fitsio
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from desiutil.log import get_logger
from desiutil.redirect import stdouterr_redirected
from desisurveyops.fba_tertiary_design_io import (
    assert_environ_settings,
    get_fn,
    read_yaml,
    assert_tertiary_settings,
    get_tile_centers_grid,
    create_tiles_table,
    creates_priority_table,
    finalize_target_table,
    assert_files,
    create_targets_assign,
    plot_targets_assign,
)
from desitarget.targetmask import desi_mask, bgs_mask
from argparse import ArgumentParser

# AR message to ensure that some key settings in the environment are correctly set
# AR put it at the very top, so that it appears first, and is not redirected
# AR    (=burried) in the log file
assert_environ_settings()

log = get_logger()


def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--yamlfn",
        help="path to the tertiary-config-PROGNUMPAD.yaml file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--steps",
        help="comma-separated list of steps to execute (default='tiles,priorities,targets,run,diagnosis')",
        type=str,
        default="tiles,priorities,targets,run,diagnosis",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="for the 'run' step, only print the commands on the prompt",
    )
    parser.add_argument(
        "--log-stdout",
        "--log_stdout",
        action="store_true",
        help="log to stdout instead of redirecting to a file",
    )
    args = parser.parse_args()
    for kwargs in args._get_kwargs():
        log.info("{} = {}".format(kwargs[0], kwargs[1]))
    return args


# AR mimicking the sv3 rosette pattern
# AR https://desi.lbl.gov/trac/wiki/SurveyOps/OnePercent#Rosettefootprints
# Creates ntile offsets arranged in a circle of radius rad
def get_offsets(ntile, rad=0.12):
    angs = np.pi / 2 + np.linspace(0, 2 * np.pi, ntile + 1)[:-1]
    dras = rad * np.cos(angs)
    ddecs = rad * np.sin(angs)
    return dras, ddecs


# AR 4 tiles
# AR see email from Noah W. from 2024-07-10 2:50PM
def create_tiles(tileid_start, ntile, field_ra, field_dec, obsconds, outfn):
    dras, ddecs = get_offsets(ntile, rad=0.12)
    dras /= np.cos(np.radians(field_dec + ddecs))  # AR correct for dec term
    tile_ras = field_ra + dras
    tile_decs = field_dec + ddecs
    # AR round to 3 digits
    tile_ras, tile_decs = tile_ras.round(3), tile_decs.round(3)
    tileids = np.arange(tileid_start, tileid_start + ntile, dtype=int)
    d = create_tiles_table(tileids, tile_ras, tile_decs, obsconds)
    d.write(outfn)


# AR usual priority scheme (+1 when observed)
def create_priorities(yamlfn, outfn):
    d = creates_priority_table(yamlfn)
    # AR print
    d.pprint_all()
    for sample in np.unique(d["TERTIARY_TARGET"]):
        sel = d["TERTIARY_TARGET"] == sample
        log.info("priorites for {}: {}".format(sample, d["PRIORITY"][sel].tolist()))
    d.write(outfn)


def create_targets(yamlfn, outfn):
    mydict = read_yaml(yamlfn)["samples"]
    tertiary_target = str(*mydict.keys())
    d = Table(fitsio.read(mydict[tertiary_target]["FN"]))
    # d["TERTIARY_TARGET"] = np.zeros(len(d), dtype=object)
    d["TERTIARY_TARGET"] = tertiary_target
    # AR finalize
    d = finalize_target_table(d, yamlfn)
    d.meta["RANDSEED"] = read_yaml(yamlfn)["settings"]["np_rand_seed"]
    d.write(outfn)


def main():

    # AR read + assert settings
    mydict = read_yaml(args.yamlfn)["settings"]
    assert_tertiary_settings(mydict)
    prognum, targdir = mydict["prognum"], mydict["targdir"]

    # AR set random seed (for SUBPRIORITY reproducibility)
    np.random.seed(mydict["np_rand_seed"])

    # AR tiles file
    if "tiles" in args.steps.split(","):
        tilesfn = get_fn(prognum, "tiles", targdir)
        log.info("run create_tiles() to generate {}".format(tilesfn))
        create_tiles(
            mydict["tileid_start"],
            mydict["ntile"],
            mydict["field_ra"],
            mydict["field_dec"],
            mydict["obsconds"],
            tilesfn,
        )

    # AR priorities file
    if "priorities" in args.steps.split(","):
        priofn = get_fn(prognum, "priorities", targdir)
        log.info("run create_priorities() to generate {}".format(priofn))
        create_priorities(args.yamlfn, priofn)

    # AR targets file
    if "targets" in args.steps.split(","):
        targfn = get_fn(prognum, "targets", targdir)
        log.info("run create_targets() to generate {}".format(targfn))
        create_targets(args.yamlfn, targfn)

    # AR sanity checks + run
    if "run" in args.steps.split(","):
        assert_files(prognum, targdir)
        cmd = "desi_fba_tertiary_wrapper --prognum {} --targdir {} --rundate {} --std_dtver {}".format(
            prognum, targdir, mydict["rundate"], mydict["std_dtver"]
        )
        cmd = "{} --custom_too_development".format(cmd)
        if args.dry_run:
            cmd = "{} --dry_run".format(cmd)
        log.info(cmd)
        os.system(cmd)

    # AR diagnosis
    if "diagnosis" in args.steps.split(","):
        create_targets_assign(prognum, targdir)
        plot_targets_assign(prognum, targdir)


if __name__ == "__main__":

    args = parse()

    if args.log_stdout:
        main()
    else:
        _ = read_yaml(args.yamlfn)["settings"]
        logfn = get_fn(_["prognum"], "log", _["targdir"])
        with stdouterr_redirected(to=logfn):
            main()
