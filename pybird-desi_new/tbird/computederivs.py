import numpy as np
import os
import time
import sys
from itertools import combinations
import findiff
from configobj import ConfigObj

sys.path.append("../")
from tbird.Grid import run_camb, run_class, grid_properties


def get_grids(parref, outgrid, name, nmult=3, nout=3, pad=True, cf=False):
    # order_i is the number of points away from the origin for parameter i
    # The len(freepar) sub-arrays are the outputs of a meshgrid, which I feed to findiff
    # outgrid = parref["outgrid"]
    # name = parref["code"].lower() + "-" + parref["gridname"]

    # Coordinates have shape (len(freepar), 2 * order_1 + 1, ..., 2 * order_n + 1)
    shapecrd = np.concatenate([[len(parref["freepar"])], np.full(len(parref["freepar"]), 2 * int(parref["order"]) + 1)])
    padshape = [(1, 1)] * (len(shapecrd) - 1)

    # grids need to be reshaped and padded at both ends along the freepar directions
    params = np.load(os.path.join(outgrid, "TableParams_%s.npy" % name))
    params = params.reshape((*shapecrd[1:], params.shape[-1]))
    if pad:
        params = np.pad(params, padshape + [(0, 0)], "constant", constant_values=0)

    if cf:
        plin = np.load(os.path.join(outgrid, "TableClin_%s.npy" % name))
    else:
        plin = np.load(os.path.join(outgrid, "TablePlin_%s.npy" % name))
    plin = plin.reshape((*shapecrd[1:], nmult, plin.shape[-2] // nmult, plin.shape[-1]))
    if pad:
        plin = np.pad(plin, padshape + [(0, 0)] * 3, "constant", constant_values=0)

    if cf:
        ploop = np.load(os.path.join(outgrid, "TableCloop_%s.npy" % name))
    else:
        ploop = np.load(os.path.join(outgrid, "TablePloop_%s.npy" % name))
    ploop = ploop.reshape((*shapecrd[1:], nmult, ploop.shape[-2] // nmult, ploop.shape[-1]))
    if pad:
        ploop = np.pad(ploop, padshape + [(0, 0)] * 3, "constant", constant_values=0)

    """if cf:
        plin_noAP = np.load(os.path.join(outgrid, "TableClin_%s.npy" % name))
    else:
        plin_noAP = np.load(os.path.join(outgrid, "TablePlin_%s.npy" % name))
    plin_noAP = plin_noAP.reshape((*shapecrd[1:], nmult, plin_noAP.shape[-2] // nmult, plin_noAP.shape[-1]))
    if pad:
        plin_noAP = np.pad(plin_noAP, padshape + [(0, 0)] * 3, "constant", constant_values=0)

    if cf:
        ploop_noAP = np.load(os.path.join(outgrid, "TableCloop_%s.npy" % name))
    else:
        ploop_noAP = np.load(os.path.join(outgrid, "TablePloop_%s.npy" % name))
    ploop_noAP = ploop_noAP.reshape((*shapecrd[1:], nmult, ploop_noAP.shape[-2] // nmult, ploop_noAP.shape[-1]))
    if pad:
        ploop_noAP = np.pad(ploop_noAP, padshape + [(0, 0)] * 3, "constant", constant_values=0)"""

    # The output is not concatenated for multipoles
    return (
        params,
        plin[..., :nout, :, :],
        ploop[..., :nout, :, :],
        # plin_noAP[..., :nout, :, :],
        # ploop_noAP[..., :nout, :, :],
    )


def get_pder_lin(parref, pi, dx, filename, template=False):
    """Calculates the derivative aroud the Grid.valueref points. Do this only once.
    gridshape is 2 * order + 1, times the number of free parameters
    pi is of shape gridshape, n multipoles, k length, P columns (zeroth being k's)"""
    # Findiff syntax is Findiff((axis, delta of uniform grid along the axis, order of derivative, accuracy))
    t0 = time.time()

    if template:
        lenpar = 4
        idx = int(parref["template_order"]) + 1
    else:
        lenpar = len(parref["freepar"])
        idx = int(parref["order"]) + 1

    p0 = pi[(idx,) * lenpar]
    t1 = time.time()
    print("Done p0 in %s sec" % str(t1 - t0))

    dpdx = np.array([findiff.FinDiff((i, dx[i], 1), acc=4)(pi)[(idx,) * lenpar] for i in range(lenpar)])
    t0 = time.time()
    print("Done dpdx in %s sec" % str(t0 - t1))

    # Second derivatives
    d2pdx2 = np.array([findiff.FinDiff((i, dx[i], 2), acc=2)(pi)[(idx,) * lenpar] for i in range(lenpar)])
    t1 = time.time()
    print("Done d2pdx2 in %s sec" % str(t1 - t0))

    d2pdxdy = np.array(
        [
            [i, j, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), acc=2)(pi)[(idx,) * lenpar]]
            for (i, j) in combinations(range(lenpar), 2)
        ]
    )
    t0 = time.time()
    print("Done d2pdxdy in %s sec" % str(t0 - t1))

    # Third derivatives: we only need it for A_s, so I do this by hand
    d3pdx3 = np.array([findiff.FinDiff((i, dx[i], 3))(pi)[(idx,) * lenpar] for i in range(lenpar)])
    t1 = time.time()
    print("Done d3pdx3 in %s sec" % str(t1 - t0))

    d3pdx2dy = np.array(
        [
            [i, j, findiff.FinDiff((i, dx[i], 2), (j, dx[j], 1))(pi)[(idx,) * lenpar]]
            for (i, j) in combinations(range(lenpar), 2)
        ]
    )
    t0 = time.time()
    print("Done d3pdx2dy in %s sec" % str(t0 - t1))

    d3pdxdy2 = np.array(
        [
            [i, j, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 2))(pi)[(idx,) * lenpar]]
            for (i, j) in combinations(range(lenpar), 2)
        ]
    )
    t1 = time.time()
    print("Done d3pdxdy2 in %s sec" % str(t1 - t0))

    d3pdxdydz = np.array(
        [
            [i, j, k, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), (k, dx[k], 1))(pi)[(idx,) * lenpar]]
            for (i, j, k) in combinations(range(lenpar), 3)
        ]
    )
    t0 = time.time()
    print("Done d3pdxdydz in %s sec" % str(t0 - t1))

    d4pdx4 = np.array([findiff.FinDiff((i, dx[i], 4))(pi)[(idx,) * lenpar] for i in range(lenpar)])
    t1 = time.time()
    print("Done d4pdx4 in %s sec" % str(t1 - t0))

    d4pdx3dy = np.array(
        [
            [i, j, findiff.FinDiff((i, dx[i], 3), (j, dx[j], 1))(pi)[(idx,) * lenpar]]
            for (i, j) in combinations(range(lenpar), 2)
        ]
    )
    t0 = time.time()
    print("Done d4pdx3dy in %s sec" % str(t0 - t1))

    d4pdxdy3 = np.array(
        [
            [i, j, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 3))(pi)[(idx,) * lenpar]]
            for (i, j) in combinations(range(lenpar), 2)
        ]
    )
    t1 = time.time()
    print("Done d4pdxdy3 in %s sec" % str(t1 - t0))

    d4pdx2dydz = np.array(
        [
            [i, j, k, findiff.FinDiff((i, dx[i], 2), (j, dx[j], 1), (k, dx[k], 1))(pi)[(idx,) * lenpar]]
            for (i, j, k) in combinations(range(lenpar), 3)
        ]
    )
    t0 = time.time()
    print("Done d4pdx2dydz in %s sec" % str(t0 - t1))

    d4pdxdy2dz = np.array(
        [
            [i, j, k, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 2), (k, dx[k], 1))(pi)[(idx,) * lenpar]]
            for (i, j, k) in combinations(range(lenpar), 3)
        ]
    )
    t1 = time.time()
    print("Done d4pdxdy2dz in %s sec" % str(t1 - t0))

    d4pdxdydz2 = np.array(
        [
            [i, j, k, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), (k, dx[k], 2))(pi)[(idx,) * lenpar]]
            for (i, j, k) in combinations(range(lenpar), 3)
        ]
    )
    t0 = time.time()
    print("Done d4pdxdydz2 in %s sec" % str(t0 - t1))

    d4pdxdydzdzm = np.array(
        [
            [
                i,
                j,
                k,
                m,
                findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), (k, dx[k], 1), (m, dx[m], 1))(pi)[(idx,) * lenpar],
            ]
            for (i, j, k, m) in combinations(range(lenpar), 4)
        ]
    )
    t1 = time.time()
    print("Done d4pdxdydzdm in %s sec" % str(t1 - t0))

    allder = (
        p0,
        dpdx,
        d2pdx2,
        d2pdxdy,
        d3pdx3,
        d3pdx2dy,
        d3pdxdy2,
        d3pdxdydz,
        d4pdx4,
        d4pdx3dy,
        d4pdxdy3,
        d4pdx2dydz,
        d4pdxdy2dz,
        d4pdxdydz2,
        d4pdxdydzdzm,
    )
    np.save(filename, allder)
    return allder


def get_PSTaylor(dtheta, derivatives, taylor_order):
    # Shape of dtheta: number of free parameters
    # Shape of derivatives: tuple up to third derivative where each element has shape (num free par, multipoles, lenk, columns)
    t1 = np.einsum("pd,pmkb->dmkb", dtheta, derivatives[1])
    t2diag = np.einsum("pd,pmkb->dmkb", dtheta ** 2, derivatives[2])
    t2nondiag = np.sum([np.multiply.outer(dtheta[d[0]] * dtheta[d[1]], d[2]) for d in derivatives[3]], axis=0)
    t3diag = np.einsum("pd,pmkb->dmkb", dtheta ** 3, derivatives[4])
    t3semidiagx = np.sum([np.multiply.outer(dtheta[d[0]] ** 2 * dtheta[d[1]], d[2]) for d in derivatives[5]], axis=0)
    t3semidiagy = np.sum([np.multiply.outer(dtheta[d[0]] * dtheta[d[1]] ** 2, d[2]) for d in derivatives[6]], axis=0)
    t3nondiag = np.sum(
        [np.multiply.outer(dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]], d[3]) for d in derivatives[7]], axis=0
    )
    t4diag = np.einsum("pd,pmkb->dmkb", dtheta ** 4, derivatives[8])
    t4semidiagx = np.sum([np.multiply.outer(dtheta[d[0]] ** 3 * dtheta[d[1]], d[2]) for d in derivatives[9]], axis=0)
    t4semidiagy = np.sum([np.multiply.outer(dtheta[d[0]] * dtheta[d[1]] ** 3, d[2]) for d in derivatives[10]], axis=0)
    t4semidiagx2 = np.sum(
        [np.multiply.outer(dtheta[d[0]] ** 2 * dtheta[d[1]] * dtheta[d[2]], d[3]) for d in derivatives[11]], axis=0
    )
    t4semidiagy2 = np.sum(
        [np.multiply.outer(dtheta[d[0]] * dtheta[d[1]] ** 2 * dtheta[d[2]], d[3]) for d in derivatives[12]], axis=0
    )
    t4semidiagz2 = np.sum(
        [np.multiply.outer(dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] ** 2, d[3]) for d in derivatives[13]], axis=0
    )
    t4nondiag = np.sum(
        [np.multiply.outer(dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] * dtheta[d[3]], d[4]) for d in derivatives[14]],
        axis=0,
    )
    allPS = derivatives[0] + t1
    if taylor_order > 1:
        allPS += 0.5 * t2diag + t2nondiag
        if taylor_order > 2:
            allPS += t3diag / 6.0 + t3semidiagx / 2.0 + t3semidiagy / 2.0 + t3nondiag
            if taylor_order > 3:
                allPS += (
                    t4diag / 24.0
                    + t4semidiagx / 6.0
                    + t4semidiagy / 6.0
                    + t4semidiagx2 / 2.0
                    + t4semidiagy2 / 2.0
                    + t4semidiagz2 / 2.0
                    + t4nondiag
                )

    return allPS


def get_ParamsTaylor(dtheta, derivatives, taylor_order):
    # Shape of dtheta: number of free parameters
    # Shape of derivatives: tuple up to third derivative where each element has shape (num free par, multipoles, lenk, columns)
    t1 = np.einsum("p,pm->m", dtheta, derivatives[1])
    t2diag = np.einsum("p,pm->m", dtheta ** 2, derivatives[2])
    t2nondiag = np.sum([dtheta[d[0]] * dtheta[d[1]] * d[2] for d in derivatives[3]], axis=0)
    t3diag = np.einsum("p,pm->m", dtheta ** 3, derivatives[4])
    t3semidiagx = np.sum([dtheta[d[0]] ** 2 * dtheta[d[1]] * d[2] for d in derivatives[5]], axis=0)
    t3semidiagy = np.sum([dtheta[d[0]] * dtheta[d[1]] ** 2 * d[2] for d in derivatives[6]], axis=0)
    t3nondiag = np.sum([dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] * d[3] for d in derivatives[7]], axis=0)
    t4diag = np.einsum("p,pm->m", dtheta ** 4, derivatives[8])
    t4semidiagx = np.sum([dtheta[d[0]] ** 3 * dtheta[d[1]] * d[2] for d in derivatives[9]], axis=0)
    t4semidiagy = np.sum([dtheta[d[0]] * dtheta[d[1]] ** 3 * d[2] for d in derivatives[10]], axis=0)
    t4semidiagx2 = np.sum([dtheta[d[0]] ** 2 * dtheta[d[1]] * dtheta[d[2]] * d[3] for d in derivatives[11]], axis=0)
    t4semidiagy2 = np.sum([dtheta[d[0]] * dtheta[d[1]] ** 2 * dtheta[d[2]] * d[3] for d in derivatives[12]], axis=0)
    t4semidiagz2 = np.sum([dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] ** 2 * d[3] for d in derivatives[13]], axis=0)
    t4nondiag = np.sum(
        [dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] * dtheta[d[3]] * d[4] for d in derivatives[14]], axis=0
    )
    allPS = derivatives[0] + t1
    if taylor_order > 1:
        allPS += 0.5 * t2diag + t2nondiag
        if taylor_order > 2:
            allPS += t3diag / 6.0 + t3semidiagx / 2.0 + t3semidiagy / 2.0 + t3nondiag
            if taylor_order > 3:
                allPS += (
                    t4diag / 24.0
                    + t4semidiagx / 6.0
                    + t4semidiagy / 6.0
                    + t4semidiagx2 / 2.0
                    + t4semidiagy2 / 2.0
                    + t4semidiagz2 / 2.0
                    + t4nondiag
                )

    return allPS


if __name__ == "__main__":

    # Read in the config file and total number of jobs
    configfile = sys.argv[1]
    redindex = int(sys.argv[2])
    pardict = ConfigObj(configfile)
    gridnames = np.loadtxt(pardict["gridname"], dtype=str)
    outgrids = np.loadtxt(pardict["outgrid"], dtype=str)
    name = pardict["code"].lower() + "-" + gridnames[redindex]
    print(name)

    # Get the grid properties
    """
    if template:
        if pardict["code"] == "CAMB":
            kin, Pin, Om, Da, Hz, fN, sigma8, sigma8_0, sigma12, r_d = run_camb(pardict)
        else:
            kin, Pin, Om, Da, Hz, fN, sigma8, sigma8_0, sigma12, r_d = run_class(pardict)
        valueref, delta, flattenedgrid, truecrd = grid_properties_template(pardict, fN, sigma8)

        print("Let's start!")
        t0 = time.time()
        plingrid, ploopgrid = get_template_grids(pardict)
        print("Got grids in %s seconds" % str(time.time() - t0))
        print("Calculate derivatives of linear PS")
        get_pder_lin(
            pardict,
            plingrid,
            delta,
            os.path.join(pardict["outgrid"], "DerPlin_%s_template.npy" % name),
            template=True,
        )
        print("Calculate derivatives of loop PS")
        get_pder_lin(
            pardict,
            ploopgrid,
            delta,
            os.path.join(pardict["outgrid"], "DerPloop_%s_template.npy" % name),
            template=True,
        )

        plingrid, ploopgrid = get_template_grids(pardict, cf=True)
        print("Got grids in %s seconds" % str(time.time() - t0))
        print("Calculate derivatives of linear CF")
        get_pder_lin(
            pardict,
            plingrid,
            delta,
            os.path.join(pardict["outgrid"], "DerClin_%s_template.npy" % name),
            template=True,
        )
        print("Calculate derivatives of loop CF")
        get_pder_lin(
            pardict,
            ploopgrid,
            delta,
            os.path.join(pardict["outgrid"], "DerCloop_%s_template.npy" % name),
            template=True,
        )

    else:
    """

    valueref, delta, flattenedgrid, truecrd = grid_properties(pardict)

    print("Let's start!")
    t0 = time.time()
    paramsgrid, plingrid, ploopgrid = get_grids(pardict, outgrids[redindex], name)
    print("Got PS grids in %s seconds" % str(time.time() - t0))
    print("Calculate derivatives of params")
    get_pder_lin(pardict, paramsgrid, delta, os.path.join(outgrids[redindex], "DerParams_%s.npy" % name))
    print("Calculate derivatives of linear PS")
    get_pder_lin(pardict, plingrid, delta, os.path.join(outgrids[redindex], "DerPlin_%s.npy" % name))
    print("Calculate derivatives of loop PS")
    get_pder_lin(pardict, ploopgrid, delta, os.path.join(outgrids[redindex], "DerPloop_%s.npy" % name))
    # print("Calculate derivatives of linear PS without AP effect")
    # get_pder_lin(pardict, plingrid_noAP, delta, os.path.join(pardict["outgrid"], "DerPlin_%s_noAP.npy" % name))
    # print("Calculate derivatives of loop PS without AP effect")
    # get_pder_lin(pardict, ploopgrid_noAP, delta, os.path.join(pardict["outgrid"], "DerPloop_%s_noAP.npy" % name))

    """paramsgrid, clingrid, cloopgrid, clingrid_noAP, cloopgrid_noAP = get_grids(pardict, cf=True)
    print("Got CF grids in %s seconds" % str(time.time() - t0))
    print("Calculate derivatives of linear CF")
    get_pder_lin(pardict, clingrid, delta, os.path.join(pardict["outgrid"], "DerClin_%s.npy" % name))
    print("Calculate derivatives of loop CF")
    get_pder_lin(pardict, cloopgrid, delta, os.path.join(pardict["outgrid"], "DerCloop_%s.npy" % name))
    print("Calculate derivatives of linear CF without AP effect")
    get_pder_lin(pardict, clingrid_noAP, delta, os.path.join(pardict["outgrid"], "DerClin_%s_noAP.npy" % name))
    print("Calculate derivatives of loop CF without AP effect")
    get_pder_lin(pardict, cloopgrid_noAP, delta, os.path.join(pardict["outgrid"], "DerCloop_%s_noAP.npy" % name))"""
