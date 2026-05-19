import numpy as np
import matplotlib.pyplot as plt
from .binary import BinaryCrowdingModel
from .ternary import TernaryCrowdingModel


def _is_valid_data(x, y):
    if x is None or y is None:
        return False
    x_arr = np.atleast_1d(x)
    y_arr = np.atleast_1d(y)
    
    if len(x_arr) == 1:
        val = x_arr[0]
        if val is None or (isinstance(val, (int, float, np.number)) and np.isnan(val)):
            return False
    if len(y_arr) == 1:
        val = y_arr[0]
        if val is None or (isinstance(val, (int, float, np.number)) and np.isnan(val)):
            return False
            
    return len(x_arr) == len(y_arr) and len(x_arr) > 0


def _clean_yerr(yerr, y_len):
    if yerr is None:
        return None
    yerr_arr = np.atleast_1d(yerr)
    if len(yerr_arr) == 1:
        val = yerr_arr[0]
        if val is None or (isinstance(val, (int, float, np.number)) and np.isnan(val)):
            return None
        return val
    if len(yerr_arr) != y_len:
        return None
    if np.all([v is None or (isinstance(v, (int, float, np.number)) and np.isnan(v)) for v in yerr_arr]):
        return None
    return yerr_arr


class BinaryPlotter:
    def __init__(self, model: BinaryCrowdingModel):
        self.model = model

    def plot_results(self, concentration_type='phi', exp_conc=np.nan, exp_ddG=np.nan, err_ddG=np.nan,
                    exp_concT=np.nan, exp_ddH=np.nan, exp_TddS=np.nan, err_ddH=np.nan, err_TddS=np.nan,
                    folding=True):
        ''' 
        Plot model results 

        Arg:
            concentration_type: type of concetration. str - 'phi', 'molar', or 'molal'
            exp_conc: experimental concetration
            exp_ddG: experimental free energy
            err_ddG: experimental error in free energy
            exp_concT: experimental concetration for the enthalpy entropy data set
            exp_ddH: experimental enthalpy
            exp_TddS: experimental entropy
            err_ddH: experimental error in enthalpy
            err_TddS: experimental error in entropy
            folding: plot folding data in kJ (True), plot unfolding data in kcal.
        '''
        
        assert self.model.flag, 'Run solve_equil first'
        if not folding:
            if exp_ddG is not None and not (isinstance(exp_ddG, float) and np.isnan(exp_ddG)):
                exp_ddG = np.array(exp_ddG) / 4.184
            if err_ddG is not None and not (isinstance(err_ddG, float) and np.isnan(err_ddG)):
                err_ddG = np.array(err_ddG) / 4.184
            if exp_ddH is not None and not (isinstance(exp_ddH, float) and np.isnan(exp_ddH)):
                exp_ddH = np.array(exp_ddH) / 4.184
            if err_ddH is not None and not (isinstance(err_ddH, float) and np.isnan(err_ddH)):
                err_ddH = np.array(err_ddH) / 4.184
            if exp_TddS is not None and not (isinstance(exp_TddS, float) and np.isnan(exp_TddS)):
                exp_TddS = np.array(exp_TddS) / 4.184
            if err_TddS is not None and not (isinstance(err_TddS, float) and np.isnan(err_TddS)):
                err_TddS = np.array(err_TddS) / 4.184

        if concentration_type == 'phi':
            conc = self.model.phiC
            str_conc = r'$\phi_C$'
        elif concentration_type=='molar':
            conc = self.model.molar
            str_conc = 'molar'
        elif concentration_type=='molal':
            conc = self.model.molal
            str_conc = 'molal'

        if folding:
            ddA, ddA_nu, ddA_chi, ddA_eps = self.model.ddA_kj, self.model.ddA_nu_kj, self.model.ddA_chi_kj, self.model.ddA_eps_kj
            ddE, ddE_chi, ddE_eps = self.model.ddE_kj, self.model.ddE_chi_kj, self.model.ddE_eps_kj
            TddS, TddS_nu, TddS_chi, TddS_eps = self.model.TddS_kj, self.model.TddS_nu_kj, self.model.TddS_chi_kj, self.model.TddS_eps_kj
            units = '[kJ]'
        else:
            ddA, ddA_nu, ddA_chi, ddA_eps = self.model.ddA_kcal, self.model.ddA_nu_kcal, self.model.ddA_chi_kcal, self.model.ddA_eps_kcal
            ddE, ddE_chi, ddE_eps = self.model.ddE_kcal, self.model.ddE_chi_kcal, self.model.ddE_eps_kcal
            TddS, TddS_nu, TddS_chi, TddS_eps = self.model.TddS_kcal, self.model.TddS_nu_kcal, self.model.TddS_chi_kcal, self.model.TddS_eps_kcal
            units = '[kcal]'
            
        fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(8, 8), layout="constrained")
        axes[0,0].plot(conc, self.model.gamma)
        axes[0,0].set_xlabel(str_conc)
        axes[0,0].set_ylabel(r'$\Delta\Gamma_S$')

        axes[0,1].plot(conc, self.model.osm)
        axes[0,1].set_xlabel(str_conc)
        axes[0,1].set_ylabel(r'$\Pi (Osmolal)$')

        axes[0,2].plot(conc, self.model.phiCsurf)
        axes[0,2].set_xlabel(str_conc)
        axes[0,2].set_ylabel(r'$\phi_C^{surf}$')

        axes[1,0].plot(conc, ddA)
        axes[1,0].plot(conc, ddA_nu)
        axes[1,0].plot(conc, ddA_chi)
        axes[1,0].plot(conc, ddA_eps)
        if _is_valid_data(exp_conc, exp_ddG):
            axes[1,0].errorbar(exp_conc, exp_ddG, yerr=_clean_yerr(err_ddG, len(exp_ddG)), marker='o', ls='', capsize=10, label='_nolegend_')
        axes[1,0].set_xlabel(str_conc)
        axes[1,0].set_ylabel(r'$\Delta\Delta G_i^{0}$ '+units)
        axes[1,0].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])
        
        axes[1,1].plot(conc, ddE)
        axes[1,1].plot(conc, ddE_chi)
        axes[1,1].plot(conc, ddE_eps)
        if _is_valid_data(exp_concT, exp_ddH):
            axes[1,1].errorbar(exp_concT, exp_ddH, yerr=_clean_yerr(err_ddH, len(exp_ddH)), marker='o', ls='', capsize=10, label='_nolegend_')
        axes[1,1].set_xlabel(str_conc)
        axes[1,1].set_ylabel(r'$\Delta\Delta H_i^{0}$ '+units)
        axes[1,1].legend(['tot',r'$\chi$',r'$\varepsilon$'])

        axes[1,2].plot(conc, TddS)
        axes[1,2].plot(conc, TddS_nu)
        axes[1,2].plot(conc, TddS_chi)
        axes[1,2].plot(conc, TddS_eps)
        if _is_valid_data(exp_concT, exp_TddS):
            axes[1,2].errorbar(exp_concT, exp_TddS, yerr=_clean_yerr(err_TddS, len(exp_TddS)), marker='o', ls='', capsize=10, label='_nolegend_')
        axes[1,2].set_xlabel(str_conc)
        axes[1,2].set_ylabel(r'$T\Delta\Delta S_i^{0}$ '+units)
        axes[1,2].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])

        axes[2,0].plot(self.model.osm, ddA)
        axes[2,0].plot(self.model.osm, ddA_nu)
        axes[2,0].plot(self.model.osm, ddA_chi)
        axes[2,0].plot(self.model.osm, ddA_eps)
        axes[2,0].set_xlabel(r'$\Pi (Osmolal)$')
        axes[2,0].set_ylabel(r'$\Delta\Delta G_i^{0}$ '+units)
        axes[2,0].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])

        axes[2,1].plot([-max(abs(ddE)),max(abs(ddE))], [-max(abs(ddE)),max(abs(ddE))], color="darkgrey",label='_nolegend_') 
        axes[2,1].plot([-max(abs(ddE)),max(abs(ddE))], [max(abs(ddE)),-max(abs(ddE))], color="darkgrey",label='_nolegend_')
        axes[2,1].plot(ddE, TddS)
        axes[2,1].plot(np.zeros(TddS_nu.shape), TddS_nu)
        axes[2,1].plot(ddE_chi, TddS_chi)
        axes[2,1].plot(ddE_eps, TddS_eps)
        if _is_valid_data(exp_ddH, exp_TddS):
            axes[2,1].plot(exp_ddH, exp_TddS, 'o', label='_nolegend_')   
        axes[2,1].set_xlabel(r'$\Delta\Delta H_i^{0}$ '+units)
        axes[2,1].set_ylabel(r'$T\Delta\Delta S_i^{0}$ '+units)
        axes[2,1].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])


        if max(abs(ddE_chi)) != 0:
            axes[2,2].set_xlim([-max(abs(ddE_chi)),max(abs(ddE_chi))])
        else:
            axes[2,2].set_xlim([-max(abs(TddS_chi)),max(abs(TddS_chi))])
        axes[2,2].set_ylim([-max(abs(TddS_chi)),max(abs(TddS_chi))])
        axes[2,2].plot([-max(abs(ddE)),max(abs(ddE))], [-max(abs(ddE)),max(abs(ddE))], color="darkgrey",label='_nolegend_') 
        axes[2,2].plot([-max(abs(ddE)),max(abs(ddE))], [max(abs(ddE)),-max(abs(ddE))], color="darkgrey",label='_nolegend_')
        axes[2,2].plot(ddE, TddS)
        axes[2,2].plot(np.zeros(TddS_nu.shape), TddS_nu)
        axes[2,2].plot(ddE_chi, TddS_chi)
        axes[2,2].plot(ddE_eps, TddS_eps)
        axes[2,2].set_xlabel(r'$\Delta\Delta H_i^{0}$ '+units)
        axes[2,2].set_ylabel(r'$T\Delta\Delta S_i^{0}$ '+units)
        axes[2,2].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])
        axes[2,2].locator_params(axis='both', nbins=3)
        return fig


class TernaryPlotter:
    def __init__(self, model: TernaryCrowdingModel):
        self.model = model

    def plot_phiS(self):
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(10,3), layout="constrained")

        cp0=axes[0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.phi1s,
                        levels=20)
        axes[0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0].set_xlabel(r'$\phi_2$')
        axes[0].set_ylabel(r'$\phi_3$')
        axes[0].set_title(r'$\phi_1^{s}$')
        fig.colorbar(cp0)

        cp1=axes[1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.phi2s,
                        levels=20)
        axes[1].contour(cp1, colors='k', linestyles='solid', linewidths=0.5)
        axes[1].set_xlabel(r'$\phi_2$')
        axes[1].set_ylabel(r'$\phi_3$')
        axes[1].set_title(r'$\phi_2^{s}$')
        fig.colorbar(cp1)

        cp2=axes[2].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.phi3s,
                        levels=20)
        axes[2].contour(cp2, colors='k', linestyles='solid', linewidths=0.5)
        axes[2].set_xlabel(r'$\phi_2$')
        axes[2].set_ylabel(r'$\phi_3$')
        axes[2].set_title(r'$\phi_3^{s}$')
        fig.colorbar(cp2)
        return fig
        
    def plot_Ms(self):
        fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(12,4), layout="constrained")

        cp0=axes[0,0].contourf(self.model.phi2s, 
                        self.model.phi3s, 
                        1-self.model.Ms2_Ms-self.model.Ms3_Ms,
                        levels=20)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\phi_2^s$')
        axes[0,0].set_ylabel(r'$\phi_3^s$')
        axes[0,0].set_title(r'$Ms1/Ms$')
        fig.colorbar(cp0)
        axes[0,0].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[0,0].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)

        cp0=axes[1,0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        1-self.model.Ms2_Ms-self.model.Ms3_Ms,
                        levels=20)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\phi_2$')
        axes[1,0].set_ylabel(r'$\phi_3$')
        axes[1,0].set_title(r'$Ms1/Ms$')
        fig.colorbar(cp0)
        axes[1,0].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[1,0].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)


        cp0=axes[0,1].contourf(self.model.phi2s, 
                        self.model.phi3s, 
                        self.model.Ms2_Ms,
                        levels=20)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\phi_2^s$')
        axes[0,1].set_ylabel(r'$\phi_3^s$')
        axes[0,1].set_title(r'$Ms2/Ms$')
        fig.colorbar(cp0)
        axes[0,1].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[0,1].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)

        cp0=axes[1,1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.Ms2_Ms,
                        levels=20)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\phi_2$')
        axes[1,1].set_ylabel(r'$\phi_3$')
        axes[1,1].set_title(r'$Ms2/Ms$')
        fig.colorbar(cp0)
        axes[1,1].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[1,1].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)


        cp0=axes[0,2].contourf(self.model.phi2s, 
                        self.model.phi3s, 
                        self.model.Ms3_Ms,
                        levels=20)
        axes[0,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,2].set_xlabel(r'$\phi_2^s$')
        axes[0,2].set_ylabel(r'$\phi_3^s$')
        axes[0,2].set_title(r'$Ms3/Ms$')
        fig.colorbar(cp0)
        axes[0,2].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[0,2].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)


        cp0=axes[1,2].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.Ms3_Ms,
                        levels=20)
        axes[1,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,2].set_xlabel(r'$\phi_2$')
        axes[1,2].set_ylabel(r'$\phi_3$')
        axes[1,2].set_title(r'$Ms3/Ms$')
        fig.colorbar(cp0)
        axes[1,2].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[1,2].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)

        cp0=axes[0,3].contourf(self.model.phi2s, 
                        self.model.phi3s, 
                        self.model.Ms3_Ms+self.model.Ms2_Ms,
                        levels=20)
        axes[0,3].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,3].set_xlabel(r'$\phi_2^s$')
        axes[0,3].set_ylabel(r'$\phi_3^s$')
        axes[0,3].set_title(r'$Ms2/Ms+Ms3/Ms$')
        fig.colorbar(cp0)
        axes[0,3].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[0,3].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)

        cp0=axes[1,3].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.Ms3_Ms+self.model.Ms2_Ms,
                        levels=20)
        axes[1,3].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,3].set_xlabel(r'$\phi_2$')
        axes[1,3].set_ylabel(r'$\phi_3$')
        axes[1,3].set_title(r'$Ms2/Ms+Ms3/Ms$')
        fig.colorbar(cp0)
        axes[1,3].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[1,3].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)

        return fig
        
    def plot_mus2(self):
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3), layout="constrained")

        cp0=axes[0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.mu1s2,
                        levels=5)
        axes[0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0].set_xlabel(r'$\phi_2$')
        axes[0].set_ylabel(r'$\phi_3$')
        axes[0].set_title(r'$\mu_1^{s2}$')
        fig.colorbar(cp0)
        axes[0].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[0].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)

        cp0=axes[1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.mu2s2,
                        levels=5)
        axes[1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1].set_xlabel(r'$\phi_2$')
        axes[1].set_ylabel(r'$\phi_3$')
        axes[1].set_title(r'$\mu_2^{s2}$')
        fig.colorbar(cp0)
        axes[1].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[1].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)
        return fig

        
    def plot_mus3(self):
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,3), layout="constrained")

        cp0=axes[0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.mu1s3,
                        levels=5)
        axes[0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0].set_xlabel(r'$\phi_2$')
        axes[0].set_ylabel(r'$\phi_3$')
        axes[0].set_title(r'$\mu_1^{s3}$')
        fig.colorbar(cp0)
        axes[0].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[0].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)

        cp0=axes[1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.mu2s3,
                        levels=5)
        axes[1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1].set_xlabel(r'$\phi_2$')
        axes[1].set_ylabel(r'$\phi_3$')
        axes[1].set_title(r'$\mu_2^{s3}$')
        fig.colorbar(cp0)
        axes[1].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[1].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)

        cp0=axes[2].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.mu3s3,
                        levels=5)
        axes[2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[2].set_xlabel(r'$\phi_2$')
        axes[2].set_ylabel(r'$\phi_3$')
        axes[2].set_title(r'$\mu_3^{s3}$')
        fig.colorbar(cp0)
        axes[2].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[2].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)
        return fig
        
    def plot_equil_cond(self):
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15,4), layout="constrained")

        cp0=axes[0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.mu2s2-self.model.mu2 + 
                            self.model.nu2*
                            (self.model.mu1-self.model.mu1s2),
                        levels=500)
        axes[0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0].set_xlabel(r'$\phi_2$')
        axes[0].set_ylabel(r'$\phi_3$')
        axes[0].set_title(r'$(\mu_2^{s2}-\mu_2)+\nu_2(\mu_1-\mu_1^{s2})$')
        fig.colorbar(cp0)
        axes[0].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[0].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)

        cp0=axes[1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.mu2s3-self.model.mu2 + 
                            self.model.nu2*
                            (self.model.mu1-self.model.mu1s3),
                        levels=5)
        axes[1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1].set_xlabel(r'$\phi_2$')
        axes[1].set_ylabel(r'$\phi_3$')
        axes[1].set_title(r'$(\mu_2^{s3}-\mu_2)+\nu_3(\mu_1-\mu_1^{s3})$')
        fig.colorbar(cp0)
        axes[1].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[1].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)

        cp0=axes[2].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.mu3s3-self.model.mu3 + 
                            self.model.nu3*
                            (self.model.mu1-self.model.mu1s3),
                        levels=5)
        axes[2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[2].set_xlabel(r'$\phi_2$')
        axes[2].set_ylabel(r'$\phi_3$')
        axes[2].set_title(r'$(\mu_3^{s3}-\mu_3)+\nu_3(\mu_1-\mu_1^{s3})$')
        fig.colorbar(cp0)
        axes[2].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[2].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)
        return fig
        
    def plot_TdS_mix(self):
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12,6), layout="constrained")

        cp0=axes[0,0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.TdS_mix_s2,
                        levels=10)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\phi_2$')
        axes[0,0].set_ylabel(r'$\phi_3$')
        axes[0,0].set_title(r'$T\Delta S_{mix}/(k \cdot Ms_2)$')
        fig.colorbar(cp0)

        cp0=axes[0,1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.TdS_mix_s3,
                        levels=10)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\phi_2$')
        axes[0,1].set_ylabel(r'$\phi_3$')
        axes[0,1].set_title(r'$T\Delta S_{mix}/(k \cdot Ms_3)$')
        fig.colorbar(cp0)

        cp0=axes[0,2].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.TdS_mix_s_tot,
                        levels=10)
        axes[0,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,2].set_xlabel(r'$\phi_2$')
        axes[0,2].set_ylabel(r'$\phi_3$')
        axes[0,2].set_title(r'$T\Delta S_{mix}/(k \cdot SASA)$')
        fig.colorbar(cp0)

        cp0=axes[1,0].contourf(self.model.phi2s, 
                        self.model.phi3s, 
                        self.model.TdS_mix_s2,
                        levels=10)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\phi_2^s$')
        axes[1,0].set_ylabel(r'$\phi_3^s$')
        axes[1,0].set_title(r'$T\Delta S_{mix}/(k \cdot Ms_2)$')
        fig.colorbar(cp0)
        axes[1,0].set_xlim(right=np.nanmax(self.model.phi2s), left=0)
        axes[1,0].set_ylim(top=np.nanmax(self.model.phi3s), bottom=0)


        cp0=axes[1,1].contourf(self.model.phi2s, 
                        self.model.phi3s, 
                        self.model.TdS_mix_s3,
                        levels=10)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\phi_2^s$')
        axes[1,1].set_ylabel(r'$\phi_3^s$')
        axes[1,1].set_title(r'$T\Delta S_{mix}/(k \cdot Ms_3)$')
        fig.colorbar(cp0)
        axes[1,1].set_xlim(right=np.nanmax(self.model.phi2s), left=0)
        axes[1,1].set_ylim(top=np.nanmax(self.model.phi3s), bottom=0)

        cp0=axes[1,2].contourf(self.model.phi2s, 
                        self.model.phi3s, 
                        self.model.TdS_mix_s_tot,
                        levels=10)
        axes[1,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,2].set_xlabel(r'$\phi_2^s$')
        axes[1,2].set_ylabel(r'$\phi_3^s$')
        axes[1,2].set_title(r'$T\Delta S_{mix}/(k \cdot SASA)$')
        fig.colorbar(cp0)
        axes[1,2].set_xlim(right=np.nanmax(self.model.phi2s), left=0)
        axes[1,2].set_ylim(top=np.nanmax(self.model.phi3s), bottom=0)
        return fig
        
    def plot_dG_mix(self):
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12,6), layout="constrained")

        cp0=axes[0,0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.dG_mix_s2,
                        levels=10)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\phi_2$')
        axes[0,0].set_ylabel(r'$\phi_3$')
        axes[0,0].set_title(r'$\Delta G_{mix}/(k \cdot Ms_2)$')
        fig.colorbar(cp0)
        axes[0,0].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[0,0].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)

        cp0=axes[0,1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.dG_mix_s3,
                        levels=10)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\phi_2$')
        axes[0,1].set_ylabel(r'$\phi_3$')
        axes[0,1].set_title(r'$\Delta G_{mix}/(k \cdot Ms_3)$')
        axes[0,1].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[0,1].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)
        fig.colorbar(cp0)

        cp0=axes[0,2].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.dG_mix_s_tot,
                        levels=10)
        axes[0,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,2].set_xlabel(r'$\phi_2$')
        axes[0,2].set_ylabel(r'$\phi_3$')
        axes[0,2].set_title(r'$\Delta G_{mix}/(k \cdot SASA)$')
        fig.colorbar(cp0)
        axes[0,2].set_xlim(right=np.nanmax(self.model.phi2), left=0)
        axes[0,2].set_ylim(top=np.nanmax(self.model.phi3), bottom=0)


        cp0=axes[1,0].contourf(self.model.phi2s, 
                        self.model.phi3s, 
                        self.model.dG_mix_s2,
                        levels=10)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\phi_2^s$')
        axes[1,0].set_ylabel(r'$\phi_3^s$')
        axes[1,0].set_title(r'$\Delta G_{mix}/(k \cdot Ms_2)$')
        fig.colorbar(cp0)
        axes[1,0].set_xlim(right=np.nanmax(self.model.phi2s), left=0)
        axes[1,0].set_ylim(top=np.nanmax(self.model.phi3s), bottom=0)

        cp0=axes[1,1].contourf(self.model.phi2s, 
                        self.model.phi3s, 
                        self.model.dG_mix_s3,
                        levels=10)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\phi_2^s$')
        axes[1,1].set_ylabel(r'$\phi_3^s$')
        axes[1,1].set_title(r'$\Delta G_{mix}/(k \cdot Ms_3)$')
        axes[1,1].set_xlim(right=np.nanmax(self.model.phi2s), left=0)
        axes[1,1].set_ylim(top=np.nanmax(self.model.phi3s), bottom=0)
        fig.colorbar(cp0)

        cp0=axes[1,2].contourf(self.model.phi2s, 
                        self.model.phi3s, 
                        self.model.dG_mix_s_tot,
                        levels=10)
        axes[1,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,2].set_xlabel(r'$\phi_2^s$')
        axes[1,2].set_ylabel(r'$\phi_3^s$')
        axes[1,2].set_title(r'$\Delta G_{mix}/(k \cdot SASA)$')
        fig.colorbar(cp0)
        axes[1,2].set_xlim(right=np.nanmax(self.model.phi2s), left=0)
        axes[1,2].set_ylim(top=np.nanmax(self.model.phi3s), bottom=0)
        return fig

    def plot_ddG(self):
        fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(8,6), layout="constrained")

        cp0=axes[0,0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.ddG,
                        levels=10)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\phi_2$')
        axes[0,0].set_ylabel(r'$\phi_3$')
        axes[0,0].set_title(r'$\Delta\Delta G^0/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[0,1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.ddG_nu,
                        levels=10)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\phi_2$')
        axes[0,1].set_ylabel(r'$\phi_3$')
        axes[0,1].set_title(r'$\Delta\Delta G^0_\nu/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1,0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.ddG_chi,
                        levels=10)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\phi_2$')
        axes[1,0].set_ylabel(r'$\phi_3$')
        axes[1,0].set_title(r'$\Delta\Delta G^0_\chi/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1,1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.ddG_chi12_chi13,
                        levels=10)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\phi_2$')
        axes[1,1].set_ylabel(r'$\phi_3$')
        axes[1,1].set_title(r'$\Delta\Delta G^0_{\chi_{12}+\chi_{13}}/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1,2].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.ddG_chi23,
                        levels=10)
        axes[1,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,2].set_xlabel(r'$\phi_2$')
        axes[1,2].set_ylabel(r'$\phi_3$')
        axes[1,2].set_title(r'$\Delta\Delta G^0_{\chi_{23}}/(k T)$')
        fig.colorbar(cp0)
        
        cp0=axes[2,0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.ddG_eps,
                        levels=10)
        axes[2,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[2,0].set_xlabel(r'$\phi_2$')
        axes[2,0].set_ylabel(r'$\phi_3$')
        axes[2,0].set_title(r'$\Delta\Delta G^0_\varepsilon/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[2,1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.ddG_eps12_eps13,
                        levels=10)
        axes[2,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[2,1].set_xlabel(r'$\phi_2$')
        axes[2,1].set_ylabel(r'$\phi_3$')
        axes[2,1].set_title(r'$\Delta\Delta G^0_{\varepsilon_{12}+\varepsilon_{13}}/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[2,2].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.ddG_eps23,
                        levels=10)
        axes[2,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[2,2].set_xlabel(r'$\phi_2$')
        axes[2,2].set_ylabel(r'$\phi_3$')
        axes[2,2].set_title(r'$\Delta\Delta G^0_{\varepsilon_{23}}/(k T)$')
        fig.colorbar(cp0)
        return fig

    def plot_ddG_mu(self):
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,6), layout="constrained")

        cp0=axes[0,0].contourf(self.model.mu2, 
                        self.model.mu3, 
                        self.model.ddG,
                        levels=10)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\mu_2$')
        axes[0,0].set_ylabel(r'$\mu_3$')
        axes[0,0].set_title(r'$\Delta\Delta G^0/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[0,1].contourf(self.model.mu2, 
                        self.model.mu3, 
                        self.model.ddG_nu,
                        levels=10)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\mu_2$')
        axes[0,1].set_ylabel(r'$\mu_3$')
        axes[0,1].set_title(r'$\Delta\Delta G^0_\nu/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1,0].contourf(self.model.mu2, 
                        self.model.mu3, 
                        self.model.ddG_chi,
                        levels=10)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\mu_2$')
        axes[1,0].set_ylabel(r'$\mu_3$')
        axes[1,0].set_title(r'$\Delta\Delta G^0_\chi/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1,1].contourf(self.model.mu2, 
                        self.model.mu3, 
                        self.model.ddG_eps,
                        levels=10)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\mu_2$')
        axes[1,1].set_ylabel(r'$\mu_3$')
        axes[1,1].set_title(r'$\Delta\Delta G^0_\varepsilon/(k T)$')
        fig.colorbar(cp0)
        return fig
        
    def plot_TddS(self):
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,6), layout="constrained")

        cp0=axes[0,0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.TddS,
                        levels=10)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\phi_2$')
        axes[0,0].set_ylabel(r'$\phi_3$')
        axes[0,0].set_title(r'$T\Delta\Delta S^0/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[0,1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.TddS_nu,
                        levels=10)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\phi_2$')
        axes[0,1].set_ylabel(r'$\phi_3$')
        axes[0,1].set_title(r'$T\Delta\Delta S^0_\nu/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1,0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.TddS_chi,
                        levels=10)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\phi_2$')
        axes[1,0].set_ylabel(r'$\phi_3$')
        axes[1,0].set_title(r'$T\Delta\Delta S^0_\chi/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1,1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.TddS_eps,
                        levels=10)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\phi_2$')
        axes[1,1].set_ylabel(r'$\phi_3$')
        axes[1,1].set_title(r'$T\Delta\Delta S^0_\varepsilon/(k T)$')
        fig.colorbar(cp0)
        return fig
        
    def plot_ddH(self):
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(10,3), layout="constrained")

        cp0=axes[0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.ddH,
                        levels=10)
        axes[0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0].set_xlabel(r'$\phi_2$')
        axes[0].set_ylabel(r'$\phi_3$')
        axes[0].set_title(r'$\Delta\Delta H^0/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.ddH_chi,
                        levels=10)
        axes[1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1].set_xlabel(r'$\phi_2$')
        axes[1].set_ylabel(r'$\phi_3$')
        axes[1].set_title(r'$\Delta\Delta H^0_\chi/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[2].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.ddG_eps,
                        levels=10)
        axes[2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[2].set_xlabel(r'$\phi_2$')
        axes[2].set_ylabel(r'$\phi_3$')
        axes[2].set_title(r'$\Delta\Delta H^0_\varepsilon/(k T)$')
        fig.colorbar(cp0)
        return fig

    def plot_Gamma(self):
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,6), layout="constrained")

        cp0=axes[0,0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.Gamma_2,
                        levels=10)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\phi_2$')
        axes[0,0].set_ylabel(r'$\phi_3$')
        axes[0,0].set_title(r'$\Delta \Gamma_2$')
        fig.colorbar(cp0)

        cp0=axes[0,1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.Gamma_3,
                        levels=10)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\phi_2$')
        axes[0,1].set_ylabel(r'$\phi_3$')
        axes[0,1].set_title(r'$\Delta \Gamma_3$')
        fig.colorbar(cp0)

        cp0=axes[1,0].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.Gamma_1_2,
                        levels=10)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\phi_2$')
        axes[1,0].set_ylabel(r'$\phi_3$')
        axes[1,0].set_title(r'$\Delta \Gamma_{1,2}$')
        fig.colorbar(cp0)

        cp0=axes[1,1].contourf(self.model.phi2, 
                        self.model.phi3, 
                        self.model.Gamma_1_3,
                        levels=10)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\phi_2$')
        axes[1,1].set_ylabel(r'$\phi_3$')
        axes[1,1].set_title(r'$\Delta \Gamma_{1,3}$')
        fig.colorbar(cp0)
        return fig

    def plot_Gamma_mu(self):
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,6), layout="constrained")

        cp0=axes[0,0].contourf(self.model.mu2, 
                        self.model.mu3, 
                        self.model.Gamma_2,
                        levels=10)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\mu_2$')
        axes[0,0].set_ylabel(r'$\mu_3$')
        axes[0,0].set_title(r'$\Delta \Gamma_2$')
        fig.colorbar(cp0)

        cp0=axes[0,1].contourf(self.model.mu2, 
                        self.model.mu3, 
                        self.model.Gamma_3,
                        levels=10)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\mu_2$')
        axes[0,1].set_ylabel(r'$\mu_3$')
        axes[0,1].set_title(r'$\Delta \Gamma_3$')
        fig.colorbar(cp0)

        cp0=axes[1,0].contourf(self.model.mu2, 
                        self.model.mu3, 
                        self.model.Gamma_1_2,
                        levels=10)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\mu_2$')
        axes[1,0].set_ylabel(r'$\mu_3$')
        axes[1,0].set_title(r'$\Delta \Gamma_{1,2}$')
        fig.colorbar(cp0)

        cp0=axes[1,1].contourf(self.model.mu2, 
                        self.model.mu3, 
                        self.model.Gamma_1_3,
                        levels=10)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\mu_2$')
        axes[1,1].set_ylabel(r'$\mu_3$')
        axes[1,1].set_title(r'$\Delta \Gamma_{1,3}$')
        fig.colorbar(cp0)
        return fig

        
    def plot_Gamma_mu_der(self):
        L=np.linspace(self.model.Gamma_2_der.min(),self.model.Gamma_2_der.max(),20)
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10,4), layout="constrained")

        cp0=axes[0].contourf(self.model.mu2_der, 
                        self.model.mu3_der, 
                        self.model.Gamma_2_der,
                        levels=L)
        axes[0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0].set_xlabel(r'$\mu_2$')
        axes[0].set_ylabel(r'$\mu_3$')
        axes[0].set_title(r'$\Delta \Gamma_2$')
        fig.colorbar(cp0)

        cp0=axes[1].contourf(self.model.mu2_der, 
                        self.model.mu3_der, 
                        self.model.Gamma_3_der,
                        levels=L)
        axes[1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1].set_xlabel(r'$\mu_2$')
        axes[1].set_ylabel(r'$\mu_3$')
        axes[1].set_title(r'$\Delta \Gamma_3$')
        fig.colorbar(cp0)
        return fig
              
