import os
import sys
from itertools import combinations
from pathlib import Path
from pprint import pformat
from random import seed as r_seed

import numpy as np
import pandas as pd
import pyvinecopulib as pvc
from dotenv import load_dotenv
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri

load_dotenv()
DIR_WORK = Path("C:/Users/saiki/OneDrive/Documents/GitHub/WxDerivs/script/wxderivs")
sys.path.append(str(DIR_WORK / "UniBM"))

import unibm
from unibm import benchmark

dct_name_file = {}
for idx, file in enumerate([*(DIR_WORK / "data").glob("*.csv")]):
    station = file.stem.replace("_DAILY_CLIMATE_BEST_Jan14May24", "").strip()
    print(idx, station)
    dct_name_file[station] = file
SET_EUROPE = {"ESSEN", "LONDON HEATHROW", "AMSTERDAM SCHIPHOL", "PARIS ORLY"}
SET_ASIA = {"TOKYO"}
SET_USA = {_ for _ in dct_name_file if _ not in SET_EUROPE and _ not in SET_ASIA}
len(SET_USA)
# ! Nov to Mar, Dec to Feb
SET_HDD = SET_USA | SET_EUROPE
# ! May to Sep, Jul to Aug
SET_CDD = SET_USA
# ! May to Sep, Jul to Aug for Europe; Nov to Mar, May to Sep, Dec to Feb, Jul to Aug for Tokyo
SET_CAT = SET_EUROPE | SET_ASIA
#
MONTH_CDD = {5, 6, 7, 8, 9}
MONTH_HDD = {11, 12, 1, 2, 3}
#
# NUM_SIM = 1000
NUM_SIM = 50021


def fit_mean_std(srs: pd.Series) -> dict:
    vec_x = np.arange(0, len(srs))
    vec_t = srs.index.day_of_year.to_numpy() / (srs.index.is_leap_year + 365)
    vec_y = srs.values

    # ! mean
    # ! trend: intercept, slope, quadratic,
    # ! seasonality: amplitude 1, phase 1, amplitude 2, phase 2
    def SSE(par):
        intercept, slope, quadratic, amplitude_1, phase_1, amplitude_2, phase_2 = par
        return np.square(
            vec_y
            - (
                intercept
                + slope * vec_x
                + quadratic * vec_x**2
                + amplitude_1 * np.sin(2 * np.pi * (vec_t + phase_1))
                + amplitude_2 * np.cos(2 * np.pi * (vec_t + phase_2))
            )
        ).sum()

    fit_mean, fun_mean = None, np.inf
    for mtd in (
        "Nelder-Mead",
        "L-BFGS-B",
        "Powell",
        "TNC",
        "COBYLA",
        "SLSQP",
        "trust-constr",
    ):
        fit = minimize(
            SSE,
            x0=np.zeros(7),
            # x0=np.random.rand(7) / 2 + 0.5,
            method=mtd,
            bounds=[(None, None)] * 4 + [(0, 1)] + [(None, None)] + [(0, 1)],
            tol=1e-10,
        )
        if fit.fun < fun_mean:
            fit_mean, fun_mean = fit, fit.fun
    intercept, slope, quadratic, amplitude_1, phase_1, amplitude_2, phase_2 = fit.x
    vec_y_pred = pd.Series(
        (
            intercept
            + slope * vec_x
            + quadratic * vec_x**2
            + amplitude_1 * np.sin(2 * np.pi * (vec_t + phase_1))
            + amplitude_2 * np.cos(2 * np.pi * (vec_t + phase_2))
        ),
        index=srs.index,
    )

    # ! std.dev: Seasonal GARCH
    # ! amplitude_3, phase_3, amplitude_4, phase_4, omega, alpha, beta
    vec_res = vec_y - vec_y_pred
    var_max = (vec_res**2).max() * 10

    def get_sgarch_variance(vec_res: np.array, par: np.array) -> np.array:
        amplitude_3, phase_3, amplitude_4, phase_4, omega, alpha, beta = par
        vec_variance = np.empty_like(vec_res)
        # * initial value as sample variance
        vec_variance[0] = vec_res.var()
        # vec_variance[0] = omega / (1 - alpha - beta)
        # ! evolution of variance as defined in SGARCH, where parameters are engaged
        for i in range(1, len(vec_res)):
            vec_variance[i] = (
                amplitude_3 * np.sin(2 * np.pi * (vec_t[i] + phase_3))
                + amplitude_4 * np.cos(2 * np.pi * (vec_t[i] + phase_4))
                + omega
                + alpha * vec_res[i - 1] ** 2
                + beta * vec_variance[i - 1] ** 2
            )
        return vec_variance

    def MLE_sgarch(par: np.array) -> float:
        vec_variance = get_sgarch_variance(vec_res=vec_res, par=par).clip(1e-8, var_max)
        # ! neg log-likelihood function assuming Gaussian residuals
        return np.nansum(np.log(vec_variance) / 2 + np.square(vec_res) / vec_variance)
        # return -np.nansum(
        #     norm.logpdf(
        #         x=vec_res / np.sqrt(get_sgarch_variance(vec_res, par).clip(1e-8, var_max)),
        #     )
        # )

    mtd = "SLSQP"
    fit_sgarch, fun_sgarch = None, np.inf
    # * multiple starting points
    for idx in range(1, 100, 10):
        par_0 = np.ones(7) / idx
        par_0[4] = 1
        fit = minimize(
            MLE_sgarch,
            x0=par_0,
            method=mtd,
            bounds=[(-var_max, var_max), (0, 1)] * 2
            + [(1e-10, var_max)]
            + [(1e-10, 1)]
            + [(1e-10, 1)],
            constraints=[
                {"type": "ineq", "fun": lambda par: 1 - par[5] - par[6]},
                {
                    "type": "ineq",
                    "fun": lambda par: par[4] ** 2 - par[0] ** 2 - par[2] ** 2,
                },
                {
                    "type": "ineq",
                    "fun": lambda par: get_sgarch_variance(vec_res, par).min(),
                },
            ],
            tol=1e-10,
        )
        if fit.fun < fun_sgarch:
            fit_sgarch, fun_sgarch = fit, fit.fun
    vec_std = np.sqrt(get_sgarch_variance(vec_res=vec_res, par=fit_sgarch.x))

    return {
        "vec_x": vec_x,
        "vec_t": vec_t,
        "vec_y": vec_y,
        "vec_y_pred": vec_y_pred,
        "vec_std": vec_std,
        "vec_res": vec_res,
        "vec_res_scaled": vec_res / vec_std,
        "fit_mean": fit_mean,
        "fit_sgarch": fit_sgarch,
        "vec_index": srs.index,
    }


def make_cdf_ppf(
    vec_obs: np.array,
    mtd_margin: str,
    seed: int = 0,
    lst_distribution: list = [
        stats.norm,
        stats.t,
        stats.nct,
        stats.laplace,
        stats.laplace_asymmetric,
        stats.johnsonsu,
        stats.genpareto,
        stats.genextreme,
        # stats.tukeylambda,
    ],
) -> pd.Series:
    """fit marginal univariate distributions, return fitted cdf/ppf functions
    # ! band_width for np fit from:
    Dhaker, H., Deme, E. H., & Ciss, Y. (2021).
    β-Divergence loss for the kernel density estimation with bias reduced.
    Statistical Theory and Related Fields, 5(3), 221-231.
    """
    r_seed(seed)
    np.random.seed(seed=seed)
    vec_clean = np.sort(vec_obs[np.isfinite(vec_obs)])
    if mtd_margin == "np":
        # The constant has beta = 1.7320508075688772 from
        # (4 * beta**4 / (9 * beta**4 - 36 * beta**3 + 90 * beta**2 + 270 * beta + 105) * sqrt(2 / pi))**(1 / 9)
        band_width = vec_clean.std() * 0.6973425390765554 * (len(vec_clean)) ** (-1 / 9)

        # empirical cdf, semi parametric approach, from mixture
        @np.vectorize
        def res_cdf(quantile: np.array):
            # ±∞ return to (0,1) cdf
            return ndtr((quantile - vec_clean) / band_width).mean()

        vec_F, idx = np.unique(res_cdf(vec_clean), return_index=True)
        func_spline = CubicSpline(x=vec_F, y=vec_clean[idx])
        eps = 1 / (len(vec_clean) + 1)

        def res_ppf(percentage: np.array):
            # (0,1) cdf to ±∞ return
            return func_spline(np.clip(a=percentage, a_min=eps, a_max=1 - eps))

    elif mtd_margin == "p":
        dist, par, bic2, lnn2 = None, None, np.inf, np.log(len(vec_clean)) / 2
        for iter_dist in lst_distribution:
            iter_par = iter_dist.fit(vec_clean)
            iter_bic2 = (
                len(iter_par) * lnn2 - iter_dist.logpdf(vec_clean, *iter_par).sum()
            )
            if iter_bic2 < bic2:
                dist, par, bic2 = iter_dist, iter_par, iter_bic2
        res_cdf, res_ppf = (
            lambda vec: dist.cdf(vec, *par),
            lambda vec: dist.ppf(vec, *par),
        )

    return pd.Series({"cdf": res_cdf, "ppf": res_ppf})


def get_gen_idx_cv(idx_all: pd.DatetimeIndex, num_splits: int = 5, seed: int = 0):
    """generator for random cross validation"""
    np.random.seed(seed=seed)
    r_seed(seed)
    num_obs_fold = len(idx_all) // num_splits
    idx_all = np.random.permutation(idx_all)
    for i in range(num_splits):
        if i == num_splits - 1:
            idx_test = idx_all[i * num_obs_fold :]
        else:
            idx_test = idx_all[i * num_obs_fold : (i + 1) * num_obs_fold]
        idx_train = np.setdiff1d(idx_all, idx_test)
        yield idx_train, idx_test


class EllipticalCop:
    def __init__(self, family: str) -> None:
        super().__init__()
        self.__family = family

    def __repr__(self) -> str:
        return pformat(self.__dict__)

    @property
    def family(self):
        return self.__family

    @property
    def negloglik(self):
        return self.__negloglik

    @property
    def dim(self):
        return self.__rho.shape[1]

    @property
    def AIC(self):
        # rho
        num_par = sum(range(1, self.dim + 1))
        if self.__family == "studentt":
            # nu
            num_par += 1
        return 2 * (self.__negloglik + num_par)

    @property
    def nu(self) -> float:
        return self.__nu if (self.__nu is not None) else None

    @property
    def rho(self) -> np.array:
        return self.__rho

    def __str__(self) -> str:
        return pformat(
            object={
                "family": self.family,
                "dim": self.dim,
                "negloglik": round(self.negloglik, 4),
                "AIC": round(self.AIC, 4),
                # "nu": self.nu,
                # "rho": self.rho,
            },
            compact=True,
            sort_dicts=False,
            underscore_numbers=True,
        )

    @nu.setter
    def nu(self, val: float) -> None:
        self.__nu = np.clip(val, a_min=2.0, a_max=None)

    @rho.setter
    def rho(self, val: np.array) -> None:
        # post-treatment, positive definite
        # ! give rho to facilitate direct sim
        self.__rho = np.clip(val, a_min=-0.9999, a_max=0.9999)

    def fit(self, arr_V: np.array) -> None:
        arr_V = np.clip(a=arr_V, a_min=1e-10, a_max=1 - 1e-10)
        # ! rho by inverse of Kendall's tau
        _dim = arr_V.shape[1]
        _rho = np.ones(shape=(_dim, _dim))
        for i, j in combinations(range(_dim), 2):
            _rho[i, j] = _rho[j, i] = np.sin(
                stats.kendalltau(x=arr_V[:, i], y=arr_V[:, j]).correlation
                * 1.5707963267948966
            )
        self.rho = _rho
        if self.__family == "gaussian":
            # * it's known centered
            arr_R_center = ndtri(arr_V)
            self.__negloglik = (
                -stats.multivariate_normal.logpdf(
                    x=arr_R_center, cov=self.__rho, allow_singular=True
                ).sum()
                + stats.norm.logpdf(x=arr_R_center).sum()
            )
        elif self.__family == "studentt":
            # ! nu by mle
            # * it's known centered
            # ! rho: use plug-in gaussian est when rho by inverse of tau is not pos def
            if np.linalg.eigvals(self.__rho).min() <= 0:
                self.rho = np.corrcoef(ndtri(arr_V), rowvar=False)

            # * nu by MLE
            def objfun(nu):
                # ~ Negative log-likelihood func, for mv studentt COP (margin & cop)!
                arr_R_center = stats.t.ppf(q=arr_V, df=nu)
                return (
                    -stats.multivariate_t.logpdf(
                        x=arr_R_center, df=nu, shape=self.__rho
                    ).sum()
                    + stats.t.logpdf(x=arr_R_center, df=nu).sum()
                )

            res = minimize(
                fun=objfun, x0=(30,), bounds=((2.00001, 50),), method="Nelder-Mead"
            )
            self.__nu = res.x.item()
            self.__negloglik = res.fun

    def sim(self, num_sim: int, seed: int = 0) -> np.array:
        # ! https://stats.stackexchange.com/questions/318264/given-a-multivariate-normal-distribution-how-can-we-simulate-uniform-random-var
        # ! https://stats.stackexchange.com/questions/68476/drawing-from-the-multivariate-students-t-distribution
        np.random.seed(seed=seed)
        if self.__family == "gaussian":
            return ndtr(
                np.random.multivariate_normal(
                    mean=np.zeros(self.dim), cov=self.__rho, size=num_sim
                )
            )
        elif self.__family == "studentt":
            return stats.t.cdf(
                stats.multivariate_t.rvs(
                    loc=np.zeros(self.dim),
                    shape=self.__rho,
                    df=self.__nu,
                    size=num_sim,
                    random_state=seed,
                ),
                df=self.__nu,
            )
        else:
            raise NotImplementedError


def get_sim(
    df_train: pd.DataFrame,
    num_sim: int = NUM_SIM,
    seed: int = 0,
    mtd_margin: str = "np",
    lst_mdl: list = ["gaussian", "studentt", "vine"],
    # * https://vinecopulib.github.io/pyvinecopulib/_generate/pyvinecopulib.FitControlsVinecop.__init__.html
    pvc_ctrl: pvc.FitControlsVinecop = pvc.FitControlsVinecop(
        family_set=[
            pvc.BicopFamily.indep,
            pvc.BicopFamily.gaussian,
            pvc.BicopFamily.student,
            pvc.BicopFamily.clayton,
            pvc.BicopFamily.gumbel,
            pvc.BicopFamily.frank,
            pvc.BicopFamily.joe,
            pvc.BicopFamily.bb1,
            pvc.BicopFamily.bb6,
            pvc.BicopFamily.bb7,
            pvc.BicopFamily.bb8,
            pvc.BicopFamily.tawn,
            # pvc.BicopFamily.tll,
        ],
        selection_criterion="aic",
        num_threads=4,
    ),
) -> dict:
    """
    fit and simulate copula models, based on scaled residuals (training data)
    """
    np.random.seed(seed=seed)
    r_seed(seed)
    dct_sim = {}
    # * fit marginal distribution
    df_cdf_ppf = df_train.apply(
        lambda srs: make_cdf_ppf(
            vec_obs=srs.values,
            mtd_margin=mtd_margin,
            seed=seed,
        ),
        axis=0,
    )
    df_cop = df_train.apply(df_cdf_ppf.loc["cdf"], axis=0)
    for mdl in lst_mdl:
        # * Gaussian, Student's t cop
        if mdl in ["gaussian", "studentt"]:
            mdl_cop = EllipticalCop(family=mdl)
            mdl_cop.fit(arr_V=df_cop.values)
            dct_sim[mdl] = pd.DataFrame(
                data=mdl_cop.sim(num_sim=num_sim, seed=seed), columns=df_cop.columns
            ).apply(
                lambda srs: df_cdf_ppf.loc["ppf", srs.name](srs.values),
                axis=0,
            )
        # * vine cop
        elif mdl == "vine":
            # * vine copula
            mdl_cop = pvc.Vinecop.from_data(data=df_cop.values, controls=pvc_ctrl)
            dct_sim[mdl] = pd.DataFrame(
                data=mdl_cop.simulate(n=num_sim, num_threads=4), columns=df_cop.columns
            ).apply(
                lambda srs: df_cdf_ppf.loc["ppf", srs.name](srs.values),
                axis=0,
            )
    return dct_sim


def get_VaR_ES_portfolio(
    dct_sim: dict,
    df_y_test: pd.DataFrame,
    df_std: pd.DataFrame,
    df_y_pred: pd.DataFrame,
    alpha: float = 0.05,
    is_cdd: bool = True,
) -> dict:
    """
    get VaR and ES for portfolio, based on simulated data;
    * dct_sim : dict of simulated data, from various copula models
    * df_y : observed (test) data
    * df_std : fitted std.dev
    * df_y_pred : fitted mean
    * alpha : significance level
    * is_cdd : True for CDD, False for HDD; assuming equal weight long holder
    """
    srs_idx = df_y_test.index
    df_y_pred = df_y_pred.loc[srs_idx]
    df_std = df_std.loc[srs_idx]
    # * benchmark portfolio, the smooth mean
    srs_benchmark_portfolio = (
        ((df_y_pred - 65) if is_cdd else (65 - df_y_pred))
        .clip(lower=0, upper=None)
        .mean(axis=1)
    )
    # * realized portfolio
    srs_test_portfolio = (
        ((df_y_test - 65) if is_cdd else (65 - df_y_test))
        .clip(lower=0, upper=None)
        .mean(axis=1)
    )
    # ! return wrt to benchmark
    srs_test_portfolio = srs_test_portfolio / srs_benchmark_portfolio - 1
    # ! handle nan inf values
    srs_test_portfolio = pd.Series(
        data=np.nan_to_num(srs_test_portfolio.values, nan=0, posinf=0, neginf=0),
        index=srs_test_portfolio.index,
    )
    dct_VaR_ES = {_: dict() for _ in ["srs_VaR", "srs_ES"]}
    for mdl in dct_sim:
        srs_sim_portfolio = pd.Series(
            {
                idx: (
                    (
                        df_y_pred.loc[idx].values
                        + dct_sim[mdl].values * df_std.loc[idx].values
                        - 65
                    )
                    if is_cdd
                    else (
                        65
                        - df_y_pred.loc[idx].values
                        - dct_sim[mdl].values * df_std.loc[idx].values
                    )
                )
                .clip(0, None)
                .mean(axis=1)
                for idx in srs_idx
            }
        ).sort_index()
        # * VaR and ES
        srs_VaR_portfolio = srs_sim_portfolio.apply(np.quantile, q=alpha)
        srs_ES_portfolio = pd.Series(
            {
                idx: sim[sim <= srs_VaR_portfolio[idx]].mean()
                for idx, sim in srs_sim_portfolio.items()
            }
        )
        srs_VaR_portfolio = srs_VaR_portfolio / srs_benchmark_portfolio - 1
        srs_VaR_portfolio = pd.Series(
            data=np.nan_to_num(srs_VaR_portfolio.values, nan=0, posinf=0, neginf=0),
            index=srs_VaR_portfolio.index,
        )
        srs_ES_portfolio = srs_ES_portfolio / srs_benchmark_portfolio - 1
        srs_ES_portfolio = pd.Series(
            data=np.nan_to_num(srs_ES_portfolio.values, nan=0, posinf=0, neginf=0),
            index=srs_ES_portfolio.index,
        )

        dct_VaR_ES["srs_VaR"][mdl] = srs_VaR_portfolio
        dct_VaR_ES["srs_ES"][mdl] = srs_ES_portfolio
    dct_VaR_ES["srs_test"] = srs_test_portfolio
    return dct_VaR_ES


def get_joint_test_stat_cv(dct_VaR_ES_cv: dict, alpha: float = 0.05) -> dict:
    """ "
    test statistic for joint VaR and ES;
    Equation 22-23 in:
    Deng, K. and Qiu, J., 2021. Backtesting expected shortfall and beyond.
    Quantitative Finance, 21(7), pp.1109-1125.
    """
    dct_test_stat = {}
    dct_test_stat["srs_test"] = pd.concat(
        [dct_VaR_ES_cv[_]["srs_test"] for _ in dct_VaR_ES_cv]
    ).sort_index()
    dct_test_stat["srs_VaR"] = {
        mdl: pd.concat(
            [dct_VaR_ES_cv[_]["srs_VaR"][mdl] for _ in dct_VaR_ES_cv]
        ).sort_index()
        for mdl in ["gaussian", "studentt", "vine"]
    }
    dct_test_stat["srs_ES"] = {
        mdl: pd.concat(
            [dct_VaR_ES_cv[_]["srs_ES"][mdl] for _ in dct_VaR_ES_cv]
        ).sort_index()
        for mdl in ["gaussian", "studentt", "vine"]
    }
    # ! VaR and ES are positive in the paper, but negative in our case
    # * Eq. 22
    dct_test_stat["srs_test_stat"] = {
        mdl: alpha * (dct_test_stat["srs_VaR"][mdl] - dct_test_stat["srs_ES"][mdl])
        + (dct_test_stat["srs_test"] - dct_test_stat["srs_VaR"][mdl]).clip(None, 0)
        for mdl in dct_test_stat["srs_VaR"]
    }
    # * Eq. 23
    dct_test_stat["test_stat"] = {
        mdl: (
            (dct_test_stat["srs_test"] - dct_test_stat["srs_VaR"][mdl]).clip(None, 0)
            / (dct_test_stat["srs_VaR"][mdl] - dct_test_stat["srs_ES"][mdl])
        )
        .mean()
        .item()
        / alpha
        + 1
        for mdl in dct_test_stat["srs_VaR"]
    }
    return dct_test_stat
