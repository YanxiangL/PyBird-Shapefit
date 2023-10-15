from __future__ import print_function
import os
import numpy as np
import scipy.constants as conts
from cobaya.likelihood import Likelihood
from cobaya.conventions import _input_params_prefix, _output_params_prefix

try:
    from . import pybird as pb
except ImportError:
    raise Exception("Cannot find pybird library")


class Likelihood_eft(Likelihood):
    def initialize(self):
        """
        Prepare any computation, importing any necessary code, files, etc.
        """
        # read values of k (in h/Mpc)
        k3, PSdata = self.__load_data()
        self.k = k3.reshape(3, -1)[0]
        self.ps = PSdata.reshape(3, -1)
        self.Nk = len(self.k)
        try:
            self.kmax0
            self.kmax2
        except:
            self.kmax0 = self.kmax
            self.kmax2 = self.kmax
        kmask0 = np.argwhere((self.k <= self.kmax0) & (self.k >= self.kmin))[:, 0]
        kmask2 = np.argwhere((self.k <= self.kmax2) & (self.k >= self.kmin))[:, 0] + len(self.k)
        self.kmask = np.concatenate((kmask0, kmask2))
        self.xdata = self.k[kmask0]
        self.ydata = PSdata[self.kmask]

        # BAO
        try:
            if self.baoH is not 0 and self.baoD is not 0:
                self.ydata = np.concatenate((self.ydata, [self.baoH, self.baoD]))
                self.kmask = np.concatenate((self.kmask, [-2, -1]))
                self.with_bao = True
            else:
                self.with_bao = False
        except:
            self.with_bao = False

        # read covariance matrices
        try:
            self.covmat_file
            self.use_covmat = True
        except Exception:
            print("You should declare a covariance matrix!")

        if self.use_covmat:
            cov = np.loadtxt(os.path.join(self.data_directory, self.covmat_file))
            covred = cov[self.kmask.reshape((len(self.kmask), 1)), self.kmask]
            self.invcov = np.linalg.inv(covred)

        self.chi2data = np.dot(self.ydata, np.dot(self.invcov, self.ydata))
        self.invcovdata = np.dot(self.ydata, self.invcov)

        # self.kin = np.logspace(-5, 0, 200)

        try:
            # GDA Explain something here?
            if self.use_prior and self.priors is not None:
                self.priors = np.array(self.priors)
                if self.model == 1:
                    b3, cct, cr1, ce2, sn = self.priors
                    print(
                        "EFT priors: b3: %s, cct: %s, cr1(+cr2): %s, ce2: %s, shotnoise: %s" % (b3, cct, cr1, ce2, sn)
                    )
                elif self.model == 2:
                    b3, cct, cr1, ce2 = self.priors
                    print("EFT priors: b3: %s, cct: %s, cr1(+cr2): %s, ce2: %s" % (b3, cct, cr1, ce2))
            else:
                print("EFT priors: none")
                self.use_prior = True
                if self.model == 1:
                    self.priors = np.array([10.0, 10.0, 16.0, 10.0, 2.0])
                elif self.model == 2:
                    self.priors = np.array([10.0, 10.0, 16.0, 10.0])
                elif self.model == 3:
                    self.priors = np.array([10.0, 10.0, 16.0, 10.0, 10.0])
        except:
            self.use_prior = True
            if self.model == 1:
                self.priors = np.array([2.0, 2.0, 8.0, 2.0, 2.0])
                b3, cct, cr1, ce2, sn = self.priors
                print(
                    "EFT priors: b3: %s, cct: %s, cr1(+cr2): %s, ce2: %s, shotnoise: %s (default)"
                    % (b3, cct, cr1, ce2, sn)
                )
            elif self.model == 2:
                self.priors = np.array([2.0, 2.0, 8.0, 2.0])
                b3, cct, cr1, ce2 = self.priors
                print("EFT priors: b3: %s, cct: %s, cr1(+cr2): %s, ce2: %s (default)" % (b3, cct, cr1, ce2))
            elif self.model == 3:
                self.priors = np.array([10.0, 4.0, 8.0, 4.0, 2.0])
                b3, cct, cr1, ce2, ce1 = self.priors
                print(
                    "EFT priors: b3: %s, cct: %s, cr1(+cr2): %s, ce2: %s, ce1: %s (default)" % (b3, cct, cr1, ce2, ce1)
                )

        self.priormat = np.diagflat(1.0 / self.priors ** 2)

        self.use_BBNprior = False
        try:
            if self.omega_b_BBNsigma is not 0:
                self.use_BBNprior = True
                print("BBN prior on omega_b: on")
            else:
                print("BBN prior on omega_b: none")
        except:
            print("BBN prior on omega_b: none")

    def __load_data(self):
        """
        Helper function to read in the full data vector.
        """
        print("Loading data...")
        fname = os.path.join(self.data_directory, self.data_file)
        kPS, PSdata, _ = np.loadtxt(fname, unpack=True)
        return kPS, PSdata


class challengeA(Likelihood_eft):
    def initialize(self):

        Likelihood_eft.initialize(self)

        print("-- bird settings --")

        try:
            if self.birdlkl is "full":
                print("bird lkl: full")
            elif self.birdlkl is "marg":
                print("bird lkl: marg")
            elif self.birdlkl is "fastfull":
                print("bird lkl: fast full")
            elif self.birdlkl is "fastmarg":
                print("bird lkl: fast marg")
            else:
                self.birdlkl = "fastmarg"
                print("bird lkl: fast marg")
        except:
            self.birdlkl = "fastmarg"
            print("bird lkl: fast marg (default)")

        try:
            if self.optiresum is True:
                print("resummation: optimized")
            else:
                self.optiresum = False
                print("resummation: full")
        except:
            self.optiresum = False
            print("resummation: full (default)")

        try:
            if self.zAP != self.z:
                print("Effective redshift: %s, AP redshift: %s" % (self.z, self.zAP))
            else:
                self.zAP = self.z
                print("Effective redshift: %s" % (self.z))
        except:
            self.zAP = self.z
            print("Effective redshift: %s" % (self.z))

        try:
            self.path_to_window = os.path.join(self.data_directory, self.path_to_window)
            self.window_configspace_file = os.path.join(self.path_to_window, self.window_configspace_file)
            test = np.loadtxt(self.window_configspace_file)
            self.use_window = True
            print("Mask: on")
        except:
            print("Mask: none")
            self.window_fourier_name = None
            self.path_to_window = None
            self.window_configspace_file = None
            self.use_window = False

        try:
            if self.fibcol_window:
                print("fiber collision window: on")
            else:
                print("fiber collision window: none")
        except:
            self.fibcol_window = False
            print("fiber collision window: none")

        try:
            if self.binning:
                print("k-binning: on")
            else:
                print("k-binning: none")
        except:
            self.binning = False
            print("k-binning: none")

        self.co = pb.Common(optiresum=self.optiresum)
        self.nonlinear = pb.NonLinear(load=True, save=True, co=self.co)
        self.resum = pb.Resum(co=self.co)
        self.projection = pb.Projection(
            self.k,
            self.Om_AP,
            self.zAP,
            window_fourier_name=self.window_fourier_name,
            path_to_window=self.path_to_window,
            window_configspace_file=self.window_configspace_file,
            binning=self.binning,
            fibcol=self.fibcol_window,
            co=self.co,
        )

        self.bird = None
        print("-- bird loaded --")

    def get_requirements(self):
        """
        returns a dictionary specifying quantities calculated by a theory code are needed
        """
        needs = {
            "H0": None,
            "omegam": None,
            "omega_b": None,
            "Pk_grid": {"z": self.z, "k_max": 1.1, "nonlinear": False, "vars_pairs": [("delta_tot", "delta_tot")]},
            "angular_diameter_distance": {"z": [0.0, self.z]},
            "Hubble": {"z": [0.0, self.z], "units": "1/Mpc"},
        }
        # "fsigma8": {"z": [0., self.z]}}
        return needs

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        """
        # Prepare the vector of sampled parameter values
        # print("p dict: ", params_values)
        # Fill the derived parameters
        b1 = params_values["b1"]
        c2 = params_values["c2"]
        b3 = params_values["b3"]
        c4 = params_values["c4"]
        b5 = params_values["b5"]
        b6 = params_values["b6"]
        b8 = params_values["b8"]
        b9 = params_values["b9"]
        b10 = params_values["b10"]

        b2 = (c2 + c4) / np.sqrt(2.0)
        b4 = (c2 - c4) / np.sqrt(2.0)
        bs = [b1, b2, b3, b4, b5 / self.knl ** 2, b6 / self.km ** 2, 0.0]

        if self.birdlkl is "fastmarg" or self.birdlkl is "fastfull":
            # This is a tuple (k, z, PK), where k and z are arrays, and PK[i,j] is the value at z[i], k[j]
            self.kin, dummy, plin = self.theory.get_Pk_grid(("delta_tot", "delta_tot"))
            hpar = self.theory.get_param("H0") / 100.0
            # print(dummy) # This is weird, it is now an array, first entry is 1+z...
            # print(plin.shape)
            plin = plin[0] * hpar ** 3  # Change units to h^3/Mpc^3
            Da = (self.theory.get_angular_diameter_distance(self.z) * self.theory.get_Hubble(0.0, units="1/Mpc"))[0]
            H = (self.theory.get_Hubble(self.z) / self.theory.get_Hubble(0.0))[0]
            # f = self.theory.get_fsigma8(self.z)[0] / self.theory.get_sigma8(self.z)[0]
            # Stupid approximation for f... Not really ideal
            f = (self.theory.get_param("omegam") * (1 + self.z) ** 3 / H ** 2) ** (0.55)
            # print("Did I get Da? ", Da)
            # print("Did I get H? ", H)
            # print("Did I get f? ", f)
            # print("This is Plin: ", plin)
            self.bird = pb.Bird(self.kin, plin, f, Da, H, self.z, which="all", co=self.co)
            self.nonlinear.PsCf(self.bird)
            self.bird.setPsCfl()
            self.resum.Ps(self.bird)
            self.projection.AP(self.bird)
            if self.use_window is True:
                self.projection.Window(self.bird)
            if self.fibcol_window:
                self.projection.fibcolWindow(self.bird)
            if self.binning:
                self.projection.kbinning(self.bird)
            else:
                self.projection.kdata(self.bird)

            self.bird.setreducePslb(bs)

            if self.birdlkl is "fastmarg":
                self.bird.Pb3 = self.bird.Ploopl[:, 3] + b1 * self.bird.Ploopl[:, 7]

            if self.birdlkl is "fastfull":
                self.bird.fullPs[0] += b8 / self.nd + b9 / self.nd / self.km ** 2 * self.k ** 2
                self.bird.fullPs[1] += b10 / self.nd / self.km ** 2 * self.k ** 2

        modelX = self.bird.fullPs.reshape(-1)

        if self.with_bao:  # BAO
            DM_at_z = self.theory.get_angular_diameter_distance(self.zbao) * (1.0 + self.zbao)
            H_at_z = self.theory.get_H(self.zbao) * conts.c / 1000.0
            rd = self.theory.get_param("rdrag") * self.rs_rescale

            theo_DM_rdfid_by_rd_in_Mpc = DM_at_z / rd * self.rd_fid_in_Mpc
            theo_H_rd_by_rdfid = H_at_z * rd / self.rd_fid_in_Mpc

            modelX = np.concatenate((modelX, [theo_H_rd_by_rdfid, theo_DM_rdfid_by_rd_in_Mpc]))

        modelX = modelX[self.kmask]

        if "marg" in self.birdlkl:
            Pi = self.__get_Pi_for_marg(self.bird.Pctl, self.bird.Pb3, b1, self.bird.f, model=self.model)
            Covbi = np.dot(Pi, np.dot(self.invcov, Pi.T)) + self.priormat
            Cinvbi = np.linalg.inv(Covbi)
            vectorbi = np.dot(modelX, np.dot(self.invcov, Pi.T)) - np.dot(self.invcovdata, Pi.T)
            chi2nomar = (
                np.dot(modelX, np.dot(self.invcov, modelX)) - 2.0 * np.dot(self.invcovdata, modelX) + self.chi2data
            )
            chi2mar = -np.dot(vectorbi, np.dot(Cinvbi, vectorbi)) + np.log(np.linalg.det(Covbi))
            chi2 = chi2mar + chi2nomar - self.priors.shape[0] * np.log(2.0 * np.pi)

        elif "full" in self.birdlkl:
            chi2 = np.dot(modelX - self.ydata, np.dot(self.invcov, modelX - self.ydata))

        if self.use_prior:
            prior = -0.5 * (
                (c2 / 10.0) ** 2
                + (c4 / 2.0) ** 2  # c2
                + (b3 / self.priors[0]) ** 2  # c4
                + (b5 / self.knl ** 2 / self.priors[1]) ** 2  # b3
                + (b6 / self.km ** 2 / self.priors[2]) ** 2  # cct
                + (b10 / self.nd / self.km ** 2 / self.priors[3]) ** 2  # cr1(+cr2)  # ce,l2
            )
            if self.model == 1:
                prior += -0.5 * ((b8 / self.nd / self.priors[4]) ** 2)  # ce0
            if self.model == 3:
                prior += -0.5 * ((b9 / self.nd / self.km ** 2 / self.priors[4]) ** 2)  # ce,l0

        if self.use_BBNprior:
            prior += -0.5 * ((self.theory.get_param("omega_b") - self.omega_b_BBNcenter) / self.omega_b_BBNsigma) ** 2

        lkl = -0.5 * chi2 + prior

        return lkl

    def __get_Pi_for_marg(self, Pct, Pb3, b1, f, model=2):

        kl2 = np.array([np.zeros(self.Nk), self.k])  # k^2 quad

        Pi = np.array(
            [
                Pb3,  # *b3
                (2 * f * Pct[:, 0 + 3] + 2 * b1 * Pct[:, 0]) / self.knl ** 2,  # *cct
                (2 * f * Pct[:, 1 + 3] + 2 * b1 * Pct[:, 1]) / self.km ** 2,  # *cr1
                # (2*f*Pct[:,2+3]+2*b1*Pct[:,2]) / self.km**2 , # *cr2
                kl2 ** 2 / self.nd / self.km ** 2,  # *ce,l2
            ]
        )

        if model == 1:
            Onel0 = np.array([np.array([np.ones(self.Nk), np.zeros(self.Nk)])])  # shot-noise mono
            Pi = np.concatenate((Pi, Onel0 / self.nd))
        elif model == 3:
            kl0 = np.array([np.array([self.k, np.zeros(self.Nk)])])  # k^2 mono
            Pi = np.concatenate((Pi, kl0 ** 2 / self.nd / self.km ** 2))

        Pi = Pi.reshape((Pi.shape[0], -1))

        if self.with_bao:  # BAO
            newPi = np.zeros(shape=(Pi.shape[0], Pi.shape[1] + 2))
            newPi[: Pi.shape[0], : Pi.shape[1]] = Pi
            Pi = 1.0 * newPi

        Pi = Pi[:, self.kmask]

        return Pi
