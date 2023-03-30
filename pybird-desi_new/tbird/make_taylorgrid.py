import numpy as np
import os
import sys
import copy
from configobj import ConfigObj

sys.path.append("../")
from pybird_dev import pybird
from tbird.Grid import grid_properties, run_camb, run_class
from fitting_codes.fitting_utils import format_pardict, FittingData

if __name__ == "__main__":

    # Read in the config file, job number and total number of jobs
    configfile = sys.argv[1]
    job_no = int(sys.argv[2])
    njobs = int(sys.argv[3])
    pardict = format_pardict(ConfigObj(configfile))

    # Compute some stuff for the grid based on the config file
    valueref, delta, flattenedgrid, _ = grid_properties(pardict)
    lenrun = int(len(flattenedgrid) / njobs)
    start = job_no * lenrun
    final = min((job_no + 1) * lenrun, len(flattenedgrid))
    arrayred = flattenedgrid[start:final]

    # Read in some properties of the data
    winnames = list(np.loadtxt(pardict["winfile"], dtype=str))
    fittingdata = FittingData(pardict)
    xdata = [max(x, key=len) for x in fittingdata.data["x_data"]]

    # Get some cosmological values at the grid centre
    if pardict["code"] == "CAMB":
        kin, Pin, Om, Da_fid, Hz_fid, DN_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12, r_d = run_camb(pardict)
    else:
        kin, Pin, Om, Da_fid, Hz_fid, DN_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12, r_d = run_class(pardict)

    # Set up pybird. Can't use the "skycut" option for multiple redshift bins if we want to accurately account
    # for non-linear growth due to e.g., neutrinos, so if not "scale_independent" we'll set up each redshift bin as a separate correlator.
    Nl = 3

    if pardict["scale_independent"]:

        correlator = pybird.Correlator()
        correlatorcf = pybird.Correlator()

        correlator.set(
            {
                "output": "bPk",
                "multipole": Nl,
                "skycut": len(pardict["z_pk"]),
                "z": pardict["z_pk"],
                "optiresum": False,
                "with_bias": False,
                "kmax": 0.35,
                "xdata": xdata,
                "with_AP": True,
                "DA_AP": Da_fid,
                "H_AP": Hz_fid,
                "with_fibercol": True,
                "with_window": True,
                "windowPk": winnames,
                "windowCf": [winname + ".dat" for winname in winnames],
            }
        )
        """correlatorcf.set(
            {
                "output": "bCf",
                "multipole": Nl,
                "skycut": len(pardict["z_pk"]),
                "z": pardict["z_pk"],
                "optiresum": True,
                "with_bias": False,
                "with_nlo_bias": True,
                "xdata": xdata,
                "with_AP": True,
                "DA_AP": Da_fid,
                "H_AP": Hz_fid,
                "with_window": True,
                "windowPk": winnames,
                "windowCf": [winname + ".dat" for winname in winnames],
            }
        )"""

    else:

        # Assumes 2 sky patches per redshift bin
        correlator = [pybird.Correlator() for _ in range(len(pardict["z_pk"][::2]))]
        # correlatorcf = [pybird.Correlator() for _ in range(len(pardict["z_pk"][::2]))]

        for i, z in enumerate(pardict["z_pk"][::2]):

            correlator[i].set(
                {
                    "output": "bPk",
                    "multipole": Nl,
                    "skycut": 2,
                    "z": pardict["z_pk"][2 * i : 2 * i + 2],
                    "optiresum": False,
                    "with_bias": False,
                    "kmax": 0.35,
                    "xdata": xdata[2 * i : 2 * i + 2],
                    "with_AP": True,
                    "DA_AP": Da_fid[2 * i : 2 * i + 2],
                    "H_AP": Hz_fid[2 * i : 2 * i + 2],
                    "with_fibercol": True,
                    "with_window": False,
                    "windowPk": winnames[2 * i : 2 * i + 2],
                    "windowCf": [winname + ".dat" for winname in winnames[2 * i : 2 * i + 2]],
                }
            )
            """correlatorcf[i].set(
                {
                    "output": "bCf",
                    "multipole": Nl,
                    "skycut": 2,
                    "z": z,
                    "optiresum": True,
                    "with_bias": False,
                    "with_nlo_bias": True,
                    "xdata": xdata,
                    "with_AP": True,
                    "DA_AP": Da_fid,
                    "H_AP": Hz_fid,
                    "with_window": True,
                    "windowPk": winnames[2 * i : 2 * i + 1],
                    "windowCf": [winname + ".dat" for winname in winnames[2 * i : 2 * i + 1]],
                }
            )"""

    # Now loop over all grid cells and compute the EFT model
    allPlin = [np.empty((len(arrayred), Nl * len(x), 4)) for x in xdata]
    allPloop = [np.empty((len(arrayred), Nl * len(x), 19)) for x in xdata]
    # allClin = []
    # allCloop = []
    allParams = np.empty((len(pardict["z_pk"]), len(arrayred), 9))
    allPin = np.empty((len(pardict["z_pk"]), len(arrayred), len(kin)))
    for i, theta in enumerate(arrayred):
        parameters = copy.deepcopy(pardict)
        truetheta = valueref + theta * delta
        idx = i
        print("i on tot", i, len(arrayred))

        if (i == 0) or ((i + 1) % 10 == 0):
            print("theta check: ", arrayred[idx], theta, truetheta)

        for k, var in enumerate(pardict["freepar"]):
            parameters[var] = truetheta[k]
        if parameters["code"] == "CAMB":
            kin, Pin, Om, Da, Hz, DN, fN, sigma8, sigma8_0, sigma12, r_d = run_camb(parameters)
        else:
            kin, Pin, Om, Da, Hz, DN, fN, sigma8, sigma8_0, sigma12, r_d = run_class(parameters)

        allPin[:, i, :] = Pin
        allParams[:, i, :] = np.array(
            [
                np.repeat(Om, len(pardict["z_pk"])),
                Da,
                Hz,
                DN,
                fN,
                sigma8,
                np.repeat(sigma8_0, len(pardict["z_pk"])),
                sigma12,
                np.repeat(r_d, len(pardict["z_pk"])),
            ]
        ).T

        if pardict["scale_independent"]:

            # Get non-linear power spectra from pybird for all z-bins at once. This only needs the 'first' Pin value,
            # corresponding to the 'first' DN. Everything then gets rescaled by DN/DN[0] ** n as appropriate. However
            # pybird reorders the skycuts so the middle redshift is actually the 'first', so we need to make sure
            # to use the middle redshift too!
            first = len(pardict["z_pk"]) // 2
            correlator.compute({"k11": kin, "P11": Pin[first], "Omega0_m": Om, "D": DN, "f": fN, "DA": Da, "H": Hz})

        else:

            # Get non-linear power spectra from pybird for each z-bin one at a time
            for i, z in enumerate(pardict["z_pk"][::2]):
                correlator[i].compute(
                    {
                        "k11": kin,
                        "P11": Pin[2 * i],
                        "Omega0_m": Om,
                        "D": DN[2 * i : 2 * i + 2],
                        "f": fN[2 * i : 2 * i + 2],
                        "DA": Da[2 * i : 2 * i + 2],
                        "H": Hz[2 * i : 2 * i + 2],
                    }
                )

        for j in range(len(pardict["z_pk"][::2])):
            corr = correlator if pardict["scale_independent"] else correlator[j]
            # corrcf = correlatorcf if pardict["scale_independent"] else correlatorcf[j]
            for cut in range(2):
                ri = 2 * j + cut
                bi = ri if pardict["scale_independent"] else cut

                Plin, Ploop = corr.birds[bi].formatTaylorPs(kdata=xdata[ri])
                allPlin[ri][i], allPloop[ri][i] = Plin, Ploop

    for j in range(len(pardict["z_pk"])):
        if pardict["code"] == "CAMB":
            np.save(
                os.path.join(pardict["outpk"], "redindex%d" % (j), "CAMB_run%s.npy" % (str(job_no))),
                np.array(allPin[j]),
            )
        else:
            np.save(
                os.path.join(pardict["outpk"], "redindex%d" % (j), "CLASS_run%s.npy" % (str(job_no))),
                np.array(allPin[j]),
            )
        np.save(
            os.path.join(pardict["outpk"], "redindex%d" % (j), "Plin_run%s.npy" % (str(job_no))),
            np.array(allPlin[j]),
        )
        np.save(
            os.path.join(pardict["outpk"], "redindex%d" % (j), "Ploop_run%s.npy" % (str(job_no))),
            np.array(allPloop[j]),
        )
        """np.save(
            os.path.join(pardict["outpk"], "redindex%d" % (ri), "Clin_run%s.npy" % (str(job_no))),
            np.array(allClin),
        )
        np.save(
            os.path.join(pardict["outpk"], "redindex%d" % (ri), "Cloop_run%s.npy" % (str(job_no))),
            np.array(allCloop),
        )"""
        np.save(
            os.path.join(pardict["outpk"], "redindex%d" % (j), "Params_run%s.npy" % (str(job_no))),
            np.array(allParams[j]),
        )
