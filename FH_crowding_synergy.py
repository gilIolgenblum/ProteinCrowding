import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from typing import Optional, Sequence, Tuple, Union

class var:
    '''
    Base class with shared constants and small numeric helpers.
    '''
    R: float = 8.314  # J/(mol·K)
    Vs: float = 0.018  # solvent molar vol in L/mol
    _EPS: float = 1e-12  # small epsilon for stable logs

    def cal_phiC(self, phi_min: float, phi_max: float, dphi: float) -> np.ndarray:
        '''Return an array of cosolute volume fractions.'''
        return np.arange(phi_min, phi_max, dphi)

    def _clip_phi(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''Clip volume fractions to (EPS, 1-EPS) to avoid log singularities.'''
        return np.clip(x, self._EPS, 1 - self._EPS)

    def _log(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''Numerically-stable natural log for probabilities/volume fractions.'''
        return np.log(self._clip_phi(x))

    def _to_kj(self, x: Union[float, np.ndarray], T: Optional[float] = None) -> Union[float, np.ndarray]:
        T_use = T if T is not None else getattr(self, 'T', 298.0)
        return x * self.R * T_use / 1000.0

    def _to_kcal(self, x: Union[float, np.ndarray], T: Optional[float] = None) -> Union[float, np.ndarray]:
        return -self._to_kj(x, T) / 4.184


class cosolute(var):
    '''
    Cosolute class, contains class variable and methods that depend on the cosolute propeties.

    Args:
        nu: FH excluded volume parameter
        chi: FH non-ideal interaction parameter
        chiTS: The entropic contribution of chi
        phiC_min: Minimal concetration (volume fraction)
        phiC_max: Maximal concetration (volume fraction)
        dphiC: Concetraion step (volume fraction)
    '''
    
    def __init__(self, nu: float, chi: float, chiTS: float,
                 phiC_min: float = 0.0001, phiC_max: float = 0.35, dphiC: float = 0.0001):
        # set variables
        self.nu, self.chi, self.chiTS, self.phiC_min, self.phiC_max, self.dphiC = nu, chi, chiTS, phiC_min, phiC_max, dphiC
        self.chiH = self.chi + self.chiTS
        self.phiC = self.cal_phiC(self.phiC_min, self.phiC_max, self.dphiC)
        self.phiS = 1-self.phiC
        self.molar = self.phiC / (self.nu * self.Vs)
        self.molal = self.phiC / (18 * self.phiS * self.nu)*1000
        self.dG, self.dH, self.dS = self.cal_FH_free_energy(), self.cal_FH_enthalpy(), self.cal_FH_entropy()
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.osm = self.cal_osm()
        
    def __str__(self):
        return f"Cosolute (\u03BD={self.nu}, \u03C7={self.chi}, \u03C7ₛ={self.chiTS})"

    def cal_muC(self) -> np.ndarray:
        ''' Calculate the chemical potential of the cosolute '''
        return (1-self.phiC)*(1-(self.nu)) + self._log(self.phiC) + self.chi*(self.nu)*((1-self.phiC)**2)

    def cal_muS(self) -> np.ndarray:
        ''' Calculate the chemical potential of the solvent '''
        return self._log(1-self.phiC) + self.phiC*(1-(1/self.nu)) + self.chi*((self.phiC)**2)

    def cal_FH_free_energy(self) -> np.ndarray:
        ''' Calculate the cosolute-solvent FH mixing free energy '''
        return self.phiS*self._log(self.phiS) + 1/self.nu*self.phiC*self._log(self.phiC) + self.chi*self.phiS*self.phiC

    def cal_FH_entropy(self) -> np.ndarray:
        ''' Calculate the cosolute-solvent FH mixing entropy '''
        return -self.phiS*self._log(self.phiS) - 1/self.nu*self.phiC*self._log(self.phiC) + self.chiTS*self.phiS*self.phiC
    
    def cal_FH_enthalpy(self) -> np.ndarray:
        ''' Calculate the cosolute-solvent FH mixing enthalpy '''
        return self.chiH*self.phiS*self.phiC

    def cal_osm(self) -> np.ndarray:
        ''' calculates the cosolute osmotic pressure'''
        return -(self.muS) / self.Vs

class protein(var):
    '''
    Protein class, contains class variable and methods that depend on the protein propeties.

    Args:
        SASA: Change in solvent accesible surface area due to protein folding
    '''
    
    def __init__(self, SASA: float):
        self.SASA = SASA        
    def __str__(self):
        return f"Protein (SASA={self.SASA})"

class crowding(cosolute):
    '''
    Mean-field model class, contains class variable and methods that used to solve the folding thermodynamics of
    a protein-cosolute pair  for a set of SASA, nu, chi, and eps.

    
    Args:
        protien: A protein class - used to get protein parameters (SASA)
        cosol: A cosolute class - used to get cosolute parameters (nu, chi, chiTS)
        eps: soft interaction parameter
        epsTs: The entropic contribution of eps
        phiC_min: Minimal concetration (volume fraction)
        phiC_max: Maximal concetration (volume fraction)
        dphiC: Concetraion step (volume fraction)  
        T: Temperature (Kelvin)
    '''
    
    def __init__(self, protein: protein, cosolute: cosolute, eps: float = 0, epsTS: float = 0,
                 phiC_min: float = 0.0001, phiC_max: float = 0.15, dphiC: float = 0.0001, T: float = 298):
        # set variables
        self.T=T
        self.nu, self.chi, self.chiTS = cosolute.nu, cosolute.chi, cosolute.chiTS
        self.eps, self.epsTS, self.SASA = eps, epsTS, protein.SASA
        self.chiH, self.epsH = self.chi + self.chiTS, self.eps + self.epsTS
        self.a = self.nu**(1/3)
        self.phiC = self.cal_phiC(phiC_min, phiC_max, dphiC)
        self.phiS = 1-self.phiC
        self.molar = self.phiC / (self.nu * self.Vs)
        self.molal = self.phiC / (18 * self.phiS * self.nu)*1000
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.osm = self.cal_osm()
        
        # initialize empty arrays and variables
        self.phiCsurf, self.phiSsurf = np.zeros(self.phiC.shape), np.zeros(self.phiC.shape) 
        self.flag=False # Enable calculation of thermodynamic potential only after solving the equilibrium condition
        
        self.gamma, self.gammaC = 0, 0
        self.ddA, self.ddA_nu, self.ddA_chi, self.ddA_eps = 0, 0, 0, 0
        self.ddE, self.ddE_chi, self.ddE_eps = 0, 0 ,0
        self.TddS, self.TddS_nu, self.TddS_chi, self.TddS_eps = 0, 0, 0 ,0
        
        ## _Ms = per protein domain volume "Ms"
        self.gamma_Ms, self.gammaC_Ms = 0, 0
        self.ddA_Ms, self.ddA_nu_Ms, self.ddA_chi_Ms, self.ddA_eps_Ms = 0, 0, 0, 0
        self.ddE_Ms, self.ddE_chi_Ms, self.ddE_eps_Ms = 0, 0 ,0
        self.TddS_Ms, self.TddS_nu_Ms, self.TddS_chi_Ms, self.TddS_eps_Ms = 0, 0, 0 ,0
        self.dddA_Ms, self.dddE_Ms, self.TdddS_Ms = 0, 0, 0

        ## _kj = in units of kilo joul 
        self.ddA_kj, self.ddA_nu_kj, self.ddA_chi_kj, self.ddA_eps_kj = 0, 0, 0, 0
        self.ddE_kj, self.ddE_chi_kj, self.ddE_eps_kj = 0, 0 ,0
        self.TddS_kj, self.TddS_nu_kj, self.TddS_chi_kj, self.TddS_eps_kj = 0, 0, 0 ,0
        
        ## _kj = in units of kilo cal, for unfolding procees 
        self.ddA_kcal, self.ddA_nu_kcal, self.ddA_chi_kcal, self.ddA_eps_kcal = 0, 0, 0, 0
        self.ddE_kcal, self.ddE_chi_kcal, self.ddE_eps_kcal = 0, 0 ,0
        self.TddS_kcal, self.TddS_nu_kcal, self.TddS_chi_kcal, self.TddS_eps_kcal = 0, 0, 0 ,0
        
    def __str__(self):
        return f"Crowding Mean-Field Model Parameters:\nSoft Interactions (\u03B5={self.eps}, \u03B5ₛ={self.epsTS}) \nProtein (SASA={self.SASA}) \nCosolute (\u03BD={self.nu}, \u03C7={self.chi}, \u03C7ₛ={self.chiTS})"
 
    def fit_eps(self, exp_conc: Sequence[float], exp_ddG: Sequence[float], concentration_type: str = 'phi', disp: bool = True):
        ''' 
        Fit the experimental folding free energy to resolve the soft interaction parameter, eps

        Arg:
            exp_conc: Molar or volume fraction of cosolute in experiment
            exp_ddG: Folding free energy in experiment in kJ/mol
            concentration_type: type of concetration. str - 'phi', 'molar', or 'molal'
        '''
        exp_conc = np.array(exp_conc)
        exp_ddG = np.array(exp_ddG)
        if concentration_type == 'phi':
            pass
        elif concentration_type == 'molar':
            exp_conc = exp_conc * self.nu * self.Vs
        elif concentration_type=='molal':
            exp_conc = exp_conc * (18 * self.phiS * self.nu) * 1000
        else:
            raise Exception("Concetration type can be either molar, molal, or volume fraction")
        model_phiC, model_phiS = self.phiC, self.phiS # save model arrays
        # use experiment arrays for the fit
        self.phiC = exp_conc
        self.phiS = 1-self.phiC
        self.phiCsurf = np.zeros(self.phiC.shape)
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.flag=True
        self.res = minimize(self.msd_fit_eps, self.eps, (exp_ddG), options={'disp':disp})
        # return to model arrays
        self.phiC, self.phiS= model_phiC, model_phiS
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.phiCsurf = np.zeros(self.phiC.shape)
        # solve for final eps
        self.solve_equil()
        try:
            self.eps = float(self.eps[0])  # type: ignore[index]
        except Exception:
            self.eps = float(self.eps)
        self.to_pandas()

    def fit_epsTS(self, exp_conc: Sequence[float], exp_ddH: Sequence[float], exp_TddS: Sequence[float],
                  concentration_type: str = 'phi', disp: bool = True):
        ''' 
        Fit the experimental folding free energy to resolve the soft interaction parameter, eps

        Arg:
            exp_conc: Molar or volume fraction of cosolute in experiment
            exp_ddG: Folding free energy in experiment in kJ/mol
            concentration_type: type of concetration. str - 'phi', 'molar', or 'molal'
        '''
        exp_conc = np.array(exp_conc)
        exp_ddH = np.array(exp_ddH)
        exp_TddS = np.array(exp_TddS)
        if concentration_type == 'phi':
            pass
        elif concentration_type == 'molar':
            exp_conc = exp_conc * self.nu * self.Vs
        elif concentration_type=='molal':
            exp_conc = exp_conc * (18 * self.phiS * self.nu) * 1000
        else:
            raise Exception("Concetration type can be either molar or volume fraction")
        model_phiC, model_phiS = self.phiC, self.phiS # save model arrays
        # use experiment arrays for the fit
        self.phiC = exp_conc
        self.phiS = 1-self.phiC
        self.phiCsurf = np.zeros(self.phiC.shape)
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.flag=True
        self.resTS = minimize(self.msd_fit_epsTS, np.array(self.epsTS), (exp_ddH, exp_TddS), options={'disp':disp})
        # return to model arrays
        self.phiC, self.phiS= model_phiC, model_phiS
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.phiCsurf = np.zeros(self.phiC.shape)
        # solve for final eps
        self.solve_equil()
        try:
            self.epsTS = float(self.epsTS[0])  # type: ignore[index]
        except Exception:
            self.epsTS = float(self.epsTS)
        try:
            self.epsH = float(self.epsH[0])  # type: ignore[index]
        except Exception:
            self.epsH = float(self.epsH)
        self.to_pandas()
        
    def msd_fit_eps(self, eps: Union[float, np.ndarray], ddG: np.ndarray):
        self.eps = eps
        self.solve_equil()
        return ((ddG-self.ddA_kj)**2).sum()
        
    def msd_fit_epsTS(self, epsTS: Union[float, np.ndarray], ddH: np.ndarray, TddS: np.ndarray):
        self.epsTS=epsTS
        self.epsH = self.eps + self.epsTS
        self.solve_equil()
        return ((ddH-self.ddE_kj)**2).sum() + ((TddS-self.TddS_kj)**2).sum()  
    
    def fit_a(self, exp_conc: Sequence[float], exp_ddG: Sequence[float], concentration_type: str = 'phi', disp: bool = True):
        ''' 
        Fit the experimental folding free energy to resolve the protain domain size

        Arg:
            exp_conc: Molar or volume fraction of cosolute in experiment
            exp_ddG: Folding free energy in experiment in kJ/mol
            concentration_type: type of concetration. str - 'phi', 'molar', or 'molal'
        '''
        exp_conc = np.array(exp_conc)
        exp_ddG = np.array(exp_ddG)
        if concentration_type == 'phi':
            pass
        elif concentration_type == 'molar':
            exp_conc = exp_conc * self.nu * self.Vs
        elif concentration_type=='molal':
            exp_conc = exp_conc * (18 * self.phiS * self.nu) * 1000
        else:
            raise Exception("Concetration type can be either molar, molal, or volume fraction")
        model_phiC, model_phiS = self.phiC, self.phiS # save model arrays
        # use experiment arrays for the fit
        self.phiC = exp_conc
        self.phiS = 1-self.phiC
        self.phiCsurf = np.zeros(self.phiC.shape)
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.flag=True
        self.a=np.zeros(self.phiC.shape)+self.nu**(1/3)
        self.res = minimize(self.msd_fit_a, self.a, (exp_ddG), options={'disp':disp})
        self.a_vec=self.a
        self.phiC_a=self.phiC
        plt.figure()
        plt.plot(self.phiC,self.a_vec,'o')
        plt.xlabel(r'$\phi_C$')
        plt.ylabel(r'$a$')
        # return to model arrays
        self.a=self.nu**(1/3)
        self.phiC, self.phiS= model_phiC, model_phiS
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.phiCsurf = np.zeros(self.phiC.shape)
        
    def msd_fit_a(self, a: np.ndarray, ddG: np.ndarray):
        self.a=a
        self.solve_equil()
        return ((ddG-self.ddA_kj)**2).sum()
    
    def condition(self, corr_phiCsurf: float, i: int):
        ''' 
        Equilibrium condition. Solved using SciPy's fsolve at a given concetration.
        
        Args:
            corr_phiCsurf: Corrent value of volume fraction in protein domain
            i: Concetration index
        '''
        corr_phiCmix = self.cal_phiCmix(corr_phiCsurf)
        corr_phiSmix = 1-corr_phiCmix
        corr_phiSsurf = 1-corr_phiCsurf
        
        if corr_phiCmix<0 or corr_phiSmix<0 or corr_phiSsurf<0:
            return 100
        if self.a <= 0:
            return 1
        
        return self.cal_muCsurf(corr_phiSsurf,corr_phiCmix,corr_phiSmix) - (self.nu)*self.cal_muSsurf(corr_phiCsurf,corr_phiCmix,corr_phiSmix) - self.muC[i] + (self.nu)*self.muS[i]

    def solve_equil(self):
        '''Solve the equilibrium condition across the concentration range.

        Uses rolling initial guesses to accelerate fsolve convergence.
        '''
        a = self.a
        A: list = []
        guess: Optional[float] = None
        for i in range(len(self.phiC)):
            if not isinstance(a, float):
                self.a = a[i]
            if guess is None:
                guess = float(self.phiC[i])
            try:
                self.phiCsurf[i] = fsolve(self.condition, guess, args=(i,))[0]
            except Exception:
                self.phiCsurf[i] = fsolve(self.condition, float(self.phiC[i]), args=(i,))[0]
            A.append(self.a)
            guess = float(self.phiCsurf[i])
        if not isinstance(a, float):
            self.a = np.array(A)
        self.flag = True
        self.phiSsurf = 1 - self.phiCsurf
        self.cal_thermodynamics()
        
    def cal_thermodynamics(self):
        ''' Used after the equilbrium condition is solved. Calculates all model results including: preferential interaction coefficient, folding free energy, energy, and entropy.'''
        self.scaling = self.cal_scaling()
        self.phiCmix = self.cal_phiCmix()
        self.phiSmix = 1-self.phiCmix
        self.cal_gamma()
        self.cal_gammaC()
        self.cal_Free_Energy_nu()
        self.cal_Free_Energy_chi()
        self.cal_Free_Energy_eps()
        self.cal_Free_Energy()
        self.cal_Energy_chi()
        self.cal_Energy_eps()
        self.cal_Energy()
        self.cal_Entropy_nu()
        self.cal_Entropy_chi()
        self.cal_Entropy_eps()
        self.cal_Entropy()
        self.muCsurf = self.cal_muCsurf()
        self.muSsurf = self.cal_muSsurf()
        
    def cal_scaling(self, phiCsurf=-1):
        ''' 
        Calculate the scaling factor for a given volume fraction in the protein domain

        Arg:
            corr_phiCsurf: The current cosolute concentration in the protein domain
        
        Details:
            If used to solve the equilibrium condition, suply the current cosolute protein domain concetration to return the appropriate scaled volume.
            Otherwise returns the scaled volumes that corresond to the concetration array given by the instance variable. 
        '''
        
        if phiCsurf==-1:
            assert self.flag, 'Run solve_equil first'    
            phiCsurf=self.phiCsurf
        return 1-( 0.5*(1-1/self.a)*(1-phiCsurf) )
                  
    def cal_phiCmix(self, phiCsurf=-1):
        ''' 
        Calculate the scaled volume fraction in the protein domain for a given volume fraction

        Arg:
            phiCsurf: The cosolute concentration in the protein domain
        
        Details:
            If used to solve the equilibrium condition, suply the current cosolute protein domain concetration to return the appropriate scaled concetration.
            Otherwise returns the scaled concetrations that corresond to the concetration array given by the instance variable. 
        '''
        if phiCsurf==-1:
            assert self.flag, 'Run solve_equil first'  
            return self.phiCsurf / self.scaling 
        return phiCsurf / self.cal_scaling(phiCsurf)
         
    def cal_muCsurf(self, phiSsurf: Union[int, float, np.ndarray] = -1,
                    phiCmix: Union[int, float, np.ndarray] = -1,
                    phiSmix: Union[int, float, np.ndarray] = -1):
        ''' 
        Calculate the protein domain chemical potential of the cosolute

        Arg:
            phiSsurf : The solvent concentration in the protein domain
            phiCmix : The cosolute concentration in the mixing domain
            phiSmix : The solvent concetration in the mixing domain
            
        Details:
            If used to solve the equilibrium condition, suply the concetrations (see Arg) to return the chemical potential at the specific concetrations.
            Otherwise returns the chemical potentials for the concetration arrays corespnding to the instance variables. 
        '''
        
        if phiSsurf==-1 or phiCmix==-1 or phiSmix==-1:
            assert self.flag, 'Run solve_equil first'    
            phiSsurf, phiCmix, phiSmix= self.phiSsurf, self.phiCmix, self.phiSmix
        return self._log(phiCmix) + 1 - self.nu*phiSmix - phiCmix + self.chi*self.nu*phiSsurf**2 + self.eps*self.nu/self.a
        
    def cal_muSsurf(self, phiCsurf: Union[int, float, np.ndarray] = -1,
                    phiCmix: Union[int, float, np.ndarray] = -1,
                    phiSmix: Union[int, float, np.ndarray] = -1):
        ''' 
        Calculate the protein domain chemical potential of the cosolute

        Arg:
            phiCsurf : The cosolute concentration in the protein domain
            phiCmix : The cosolute concentration in the mixing domain
            phiSmix : The solvent concetration in the mixing domain
            
        Details:
            If used to solve the equilibrium condition, suply the concetrations (see Arg) to return the chemical potential at the specific concetrations.
            Otherwise returns the chemical potentials for the concetration arrays corespnding to the instance variables. 
        '''
        
        if phiCsurf==-1 or phiCmix==-1 or phiSmix==-1:
            assert self.flag, 'Run solve_equil first'    
            phiCsurf, phiCmix, phiSmix= self.phiCsurf, self.phiCmix, self.phiSmix
        return self._log(phiSmix) + 1 - phiSmix - phiCmix/self.nu + self.chi*phiCsurf**2
        
     
    def cal_Free_Energy_nu(self):
        ''' Calculate the contribution of excluded volume to the folding free energy '''
        assert self.flag, 'Run solve_equil first'
        term1 = ((self.phiC-self.phiCsurf)) + ((1-self.phiCsurf))*self._log(1-self.phiC) 
        term2 = -self.scaling*((1-self.phiCmix))*self._log(1-self.phiCmix)
        term3 = ((self.phiCsurf-self.phiC)/self.nu) + (self.phiCsurf/self.nu)*self._log(self.phiC)
        term4 = -self.scaling*(self.phiCmix/self.nu)*self._log(self.phiCmix)
        self.ddA_nu_Ms = term1+term2+term3+term4
        self.ddA_nu = self.for_all_SASA(self.ddA_nu_Ms)
        self.ddA_nu_kj = self._to_kj(self.ddA_nu)
        self.ddA_nu_kcal = self._to_kcal(self.ddA_nu)
        
    def cal_Free_Energy_chi(self):
        ''' Calculate the contribution of non-ideal interactions to the folding free energy '''
        assert self.flag, 'Run solve_equil first'
        term1 = self.chi*(self.phiCsurf-self.phiC)*(1-self.phiC)
        term2 = self.chi*self.phiC*(1-self.phiCsurf) 
        term3 = -self.chi*(1-self.phiCsurf)*self.phiCsurf
        self.ddA_chi_Ms = term1+term2+term3
        self.ddA_chi = self.for_all_SASA(self.ddA_chi_Ms)
        self.ddA_chi_kj = self._to_kj(self.ddA_chi)
        self.ddA_chi_kcal = self._to_kcal(self.ddA_chi)  

    def cal_Free_Energy_eps(self):
        ''' Calculate the contribution of soft interactions to the folding free energy '''
        assert self.flag, 'Run solve_equil first'
        self.ddA_eps_Ms =  -(self.eps*self.phiCsurf)/(self.a)
        self.ddA_eps = self.for_all_SASA(self.ddA_eps_Ms)
        self.ddA_eps_kj = self._to_kj(self.ddA_eps)
        self.ddA_eps_kcal = self._to_kcal(self.ddA_eps)

    def cal_Free_Energy(self):
        ''' Calculate the folding free energy '''
        assert self.flag, 'Run solve_equil first'
        self.ddA_Ms = self.ddA_nu_Ms + self.ddA_chi_Ms + self.ddA_eps_Ms
        self.ddA = self.ddA_nu + self.ddA_chi + self.ddA_eps
        self.ddA_kj = self.ddA_nu_kj + self.ddA_chi_kj + self.ddA_eps_kj
        self.ddA_kcal = -self.ddA_kj/4.184

    def cal_Energy_chi(self):
        ''' Calculate the contribution of non-ideal mixing to the folding energy '''
        assert self.flag, 'Run solve_equil first'
        term1 = (self.chiH)*(self.phiCsurf-self.phiC)*(1-self.phiC) 
        term2 = (self.chiH)*self.phiC*(1-self.phiCsurf) 
        term3 = -(self.chiH)*(1-self.phiCsurf)*self.phiCsurf
        self.ddE_chi_Ms = term1+term2+term3
        self.ddE_chi = self.for_all_SASA(self.ddE_chi_Ms)
        self.ddE_chi_kj = self._to_kj(self.ddE_chi)
        self.ddE_chi_kcal = self._to_kcal(self.ddE_chi)
        
    def cal_Energy_eps(self):
        ''' Calculate the contribution of soft interactions to the folding energy '''
        assert self.flag, 'Run solve_equil first'
        self.ddE_eps_Ms = -((self.epsH)*self.phiCsurf)/(self.a)
        self.ddE_eps = self.for_all_SASA(self.ddE_eps_Ms)
        self.ddE_eps_kj = self._to_kj(self.ddE_eps)
        self.ddE_eps_kcal = self._to_kcal(self.ddE_eps)

    def cal_Energy(self):
        ''' Calculate the folding energy '''
        assert self.flag, 'Run solve_equil first'
        self.ddE_Ms = self.ddE_chi_Ms + self.ddE_eps_Ms
        self.ddE = self.ddE_chi + self.ddE_eps
        self.ddE_kj = self.ddE_chi_kj + self.ddE_eps_kj
        self.ddE_kcal = -self.ddE_kj/4.184

    def cal_Entropy_nu(self):
        ''' Calculate the contribution of excluded volume to the folding entropy '''
        assert self.flag, 'Run solve_equil first'
        self.TddS_nu_Ms = -self.ddA_nu_Ms
        self.TddS_nu = -self.ddA_nu
        self.TddS_nu_kj = -self.ddA_nu_kj
        self.TddS_nu_kcal = -self.TddS_nu_kj/4.184
        
    def cal_Entropy_chi(self):
        ''' Calculate the contribution of non-ideal mixing to the folding entropy '''
        assert self.flag, 'Run solve_equil first'
        self.TddS_chi_Ms = self.ddE_chi_Ms-self.ddA_chi_Ms
        self.TddS_chi = self.ddE_chi-self.ddA_chi
        self.TddS_chi_kj = self.ddE_chi_kj-self.ddA_chi_kj
        self.TddS_chi_kcal = -self.TddS_chi_kj/4.184
        
    def cal_Entropy_eps(self):
        ''' Calculate the contribution of soft-interactions to the folding entropy '''
        assert self.flag, 'Run solve_equil first'
        self.TddS_eps_Ms = self.ddE_eps_Ms-self.ddA_eps_Ms
        self.TddS_eps = self.ddE_eps-self.ddA_eps
        self.TddS_eps_kj = self.ddE_eps_kj-self.ddA_eps_kj
        self.TddS_eps_kcal = -self.TddS_eps_kj/4.184
        
    def cal_Entropy(self):
        ''' Calculate the folding entropy '''
        assert self.flag, 'Run solve_equil first'
        self.TddS_Ms = self.TddS_nu_Ms + self.TddS_chi_Ms + self.TddS_eps_Ms
        self.TddS = self.TddS_nu + self.TddS_chi + self.TddS_eps
        self.TddS_kj = self.TddS_nu_kj + self.TddS_chi_kj + self.TddS_eps_kj
        self.TddS_kcal = -self.TddS_kj/4.184
    
    def cal_gamma(self):
        ''' Calculate the preferential hydration coefficient '''
        assert self.flag, 'Run solve_equil first'
        self.gamma_Ms = -(self.phiSsurf*(1 - (self.phiCsurf/self.phiC)*(self.phiS/self.phiSsurf) ))
        self.gamma = self.for_all_SASA(self.gamma_Ms)
        
    def cal_gammaC(self):
        ''' Calculate the preferential interactions coefficient '''
        assert self.flag, 'Run solve_equil first'
        self.gammaC_Ms = -(self.phiCsurf/self.nu*(1 - (self.phiSsurf/self.phiS)*(self.phiC/self.phiCsurf) ))
        self.gammaC = self.for_all_SASA(self.gammaC_Ms)
        
    def for_all_SASA(self, data):
        ''' 
        Convert potential from value per protein domain volume to value for the entire protein

        Arg:
            data: A thermodynamic potential: Free energy, energy, or entropy.
        '''
        return data*(self.SASA/30**(2/3))*self.a

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
        
        assert self.flag, 'Run solve_equil first'
        if concentration_type == 'phi':
            conc = self.phiC
            str_conc = r'$\phi_C$'
        elif concentration_type=='molar':
            conc = self.molar
            str_conc = 'molar'
        elif concentration_type=='molal':
            conc = self.molal
            str_conc = 'molal'

        if folding:
            ddA, ddA_nu, ddA_chi, ddA_eps = self.ddA_kj, self.ddA_nu_kj, self.ddA_chi_kj, self.ddA_eps_kj
            ddE, ddE_chi, ddE_eps = self.ddE_kj, self.ddE_chi_kj, self.ddE_eps_kj
            TddS, TddS_nu, TddS_chi, TddS_eps = self.TddS_kj, self.TddS_nu_kj, self.TddS_chi_kj, self.TddS_eps_kj
            units = '[kJ]'
        else:
            ddA, ddA_nu, ddA_chi, ddA_eps = self.ddA_kcal, self.ddA_nu_kcal, self.ddA_chi_kcal, self.ddA_eps_kcal
            ddE, ddE_chi, ddE_eps = self.ddE_kcal, self.ddE_chi_kcal, self.ddE_eps_kcal
            TddS, TddS_nu, TddS_chi, TddS_eps = self.TddS_kcal, self.TddS_nu_kcal, self.TddS_chi_kcal, self.TddS_eps_kcal
            units = '[kcal]'
            
        _, axes = plt.subplots(ncols=3, nrows=3, figsize=(8, 8), layout="constrained")
        axes[0,0].plot(conc, self.gamma)
        axes[0,0].set_xlabel(str_conc)
        axes[0,0].set_ylabel(r'$\Delta\Gamma_S$')

        axes[0,1].plot(conc, self.osm)
        axes[0,1].set_xlabel(str_conc)
        axes[0,1].set_ylabel(r'$\Pi (Osmolal)$')

        axes[0,2].plot(conc, self.phiCsurf)
        axes[0,2].set_xlabel(str_conc)
        axes[0,2].set_ylabel(r'$\phi_C^{surf}$')

        axes[1,0].plot(conc, ddA)
        axes[1,0].plot(conc, ddA_nu)
        axes[1,0].plot(conc, ddA_chi)
        axes[1,0].plot(conc, ddA_eps)
        axes[1,0].errorbar(exp_conc, exp_ddG, yerr=err_ddG, marker='o', ls='',capsize=10, label='_nolegend_')
        axes[1,0].set_xlabel(str_conc)
        axes[1,0].set_ylabel(r'$\Delta\Delta G_i^{0}$ '+units)
        axes[1,0].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])
        
        axes[1,1].plot(conc, ddE)
        axes[1,1].plot(conc, ddE_chi)
        axes[1,1].plot(conc, ddE_eps)
        axes[1,1].errorbar(exp_concT, exp_ddH, yerr=err_ddH, marker='o', ls='',capsize=10, label='_nolegend_')
        axes[1,1].set_xlabel(str_conc)
        axes[1,1].set_ylabel(r'$\Delta\Delta H_i^{0}$ '+units)
        axes[1,1].legend(['tot',r'$\chi$',r'$\varepsilon$'])

        axes[1,2].plot(conc, TddS)
        axes[1,2].plot(conc, TddS_nu)
        axes[1,2].plot(conc, TddS_chi)
        axes[1,2].plot(conc, TddS_eps)
        axes[1,2].errorbar(exp_concT, exp_TddS, yerr=err_TddS, marker='o', ls='',capsize=10, label='_nolegend_')
        axes[1,2].set_xlabel(str_conc)
        axes[1,2].set_ylabel(r'$T\Delta\Delta S_i^{0}$ '+units)
        axes[1,2].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])

        axes[2,0].plot(self.osm, ddA)
        axes[2,0].plot(self.osm, ddA_nu)
        axes[2,0].plot(self.osm, ddA_chi)
        axes[2,0].plot(self.osm, ddA_eps)
        axes[2,0].set_xlabel(r'$\Pi (Osmolal)$')
        axes[2,0].set_ylabel(r'$\Delta\Delta G_i^{0}$ '+units)
        axes[2,0].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])

        axes[2,1].plot([-max(abs(ddE)),max(abs(ddE))], [-max(abs(ddE)),max(abs(ddE))], color="darkgrey",label='_nolegend_') 
        axes[2,1].plot([-max(abs(ddE)),max(abs(ddE))], [max(abs(ddE)),-max(abs(ddE))], color="darkgrey",label='_nolegend_')
        axes[2,1].plot(ddE, TddS)
        axes[2,1].plot(np.zeros(TddS_nu.shape), TddS_nu)
        axes[2,1].plot(ddE_chi, TddS_chi)
        axes[2,1].plot(ddE_eps, TddS_eps)
        axes[2,1].plot(exp_ddH, exp_TddS,'o', label='_nolegend_')   
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
        plt.show()

    def to_pandas(self):
        self.results = pd.DataFrame({'phiC':self.phiC,
                            'phiCsurf':self.phiCsurf, 
                            'phiS':self.phiS, 
                            'phiSsurf':self.phiSsurf, 
                            'molar':self.molar,
                            'molal':self.molal,
                            'osm':self.osm,
                            'gamma_per_vol':self.gamma_Ms, 
                            'gamma':self.gamma,
                            'gammaC_per_vol':self.gammaC_Ms, 
                            'gammaC':self.gammaC,
                            'ddA_nu':self.ddA_nu, 
                            'ddA_chi':self.ddA_chi, 
                            'ddA_eps':self.ddA_eps, 
                            'ddA':self.ddA,
                            'ddE_chi':self.ddE_chi, 
                            'ddE_eps':self.ddE_eps, 
                            'ddE':self.ddE,
                            'TddS_nu':self.TddS_nu, 
                            'TddS_chi':self.TddS_chi, 
                            'TddS_eps':self.TddS_eps, 
                            'TddS':self.TddS,
                            'ddA_nu_kJ':self.ddA_nu_kj, 
                            'ddA_chi_kJ':self.ddA_chi_kj, 
                            'ddA_eps_kJ':self.ddA_eps_kj, 
                            'ddA_kJ':self.ddA_kj,
                            'ddE_chi_kJ':self.ddE_chi_kj, 
                            'ddE_eps_kJ':self.ddE_eps_kj, 
                            'ddE_kJ':self.ddE_kj,
                            'TddS_nu_kJ':self.TddS_nu_kj, 
                            'TddS_chi_kJ':self.TddS_chi_kj, 
                            'TddS_eps_kJ':self.TddS_eps_kj, 
                            'TddS_kJ':self.TddS_kj})
    
class cosolutes(var):
    '''
    Cosolutes class, contains class variable and methods that depend on the properties of two cosolutes.

    Args:
        nu2 and nu3: FH excluded volume parameters
        chi12, chi13, and chi23: FH non-ideal interaction parameters
        chiTS12, chiTS13, and chiTS23: The entropic contribution of chi12, chi13, and chi23
        phi2_min: Minimal concetration (volume fraction) of species 2
        phi2_max: Maximal concetration (volume fraction) of species 2
        dphi2: Concetraion step (volume fraction) of species 2
        phi3_min: Minimal concetration (volume fraction) of species 3
        phi3_max: Maximal concetration (volume fraction) of species 3
        dphi3: Concetraion step (volume fraction) of species 3    '''
    
    def __init__(self, nu2: float, nu3: float, chi12: float, chi13: float, chi23: float,
                 chiTS12: float, chiTS13: float, chiTS23: float,
                 phi2_min: float = 0.0001, phi2_max: float = 0.35, dphi2: float = 0.0001,
                 phi3_min: float = 0.0001, phi3_max: float = 0.35, dphi3: float = 0.0001):
        self.nu2, self.nu3 = nu2, nu3 
        self.chi12, self.chi13, self.chi23 = chi12, chi13, chi23 
        self.chiTS12, self.chiTS13, self.chiTS23 = chiTS12, chiTS13, chiTS23
        self.phi2_min, self.phi2_max, self.dphi2 = phi2_min, phi2_max, dphi2
        self.phi3_min, self.phi3_max, self.dphi3 = phi3_min, phi3_max, dphi3
        # Enthalpic parts (corrected assignments)
        self.chiH12 = self.chi12 + self.chiTS12
        self.chiH13 = self.chi13 + self.chiTS13
        self.chiH23 = self.chi23 + self.chiTS23 
        phi2 = self.cal_phiC(self.phi2_min, self.phi2_max, self.dphi2)
        phi3 = self.cal_phiC(self.phi3_min, self.phi3_max, self.dphi3)
        self.phi2, self.phi3 = np.meshgrid(phi2,phi3)
        self.phi1 = 1-self.phi2-self.phi3
        self.phi1[self.phi1<0]=np.nan
        self.molar2 = self.phi2 / (self.nu2 * self.Vs)
        self.molar3 = self.phi3 / (self.nu3 * self.Vs)
        self.molal2 = self.phi2/self.phi1 / (18*self.nu2/1000)
        self.molal3 = self.phi3/self.phi1 / (18*self.nu3/1000)
        self.dGnonIdeal, self.TdSideal = self.cal_FH_nonideal_free_energy(), self.cal_FH_ideal_TS()
        self.dG_FH =  self.cal_FH_free_energy()
        self.mu1, self.mu2, self.mu3 = self.cal_mu1(), self.cal_mu2(), self.cal_mu3()
        self.osm = self.cal_osm()
        
    def __str__(self):
        return "Cosolutes:\n(\u03BD\N{SUBSCRIPT TWO}={}, \u03C7\N{SUBSCRIPT ONE}\N{SUBSCRIPT TWO}={}, \u03C7ₛ\N{SUBSCRIPT ONE}\N{SUBSCRIPT TWO}={})\n(\u03BD\N{SUBSCRIPT THREE}={}, \u03C7\N{SUBSCRIPT ONE}\N{SUBSCRIPT THREE}={}, \u03C7ₛ\N{SUBSCRIPT ONE}\N{SUBSCRIPT THREE}={})\n(\u03C7\N{SUBSCRIPT TWO}\N{SUBSCRIPT THREE}={}, \u03C7ₛ\N{SUBSCRIPT TWO}\N{SUBSCRIPT THREE}={})".format(self.nu2,self.chi12,self.chiTS12,self.nu3,self.chi13,self.chiTS13,self.chi23,self.chiTS23)

    def cal_mu1(self):
        ''' calculates the solvent chemical potential in the bulk'''
        return self._log(self.phi1) + self.phi2*(1-1/self.nu2) + self.phi3*(1-1/self.nu3) + (1-self.phi1)* (self.chi12*self.phi2+self.chi13*self.phi3) - self.chi23*(self.phi2*self.phi3)/self.nu2
    
    def cal_mu2(self):
        ''' calculates cosolute #2 chemical potential in the bulk'''
        return self._log(self.phi2) + 1-self.nu2*self.phi1 - self.phi2 - self.nu2/self.nu3*self.phi3 + (1-self.phi2)*(self.chi12*self.nu2*self.phi1+self.chi23*self.phi3) - self.chi13*self.nu2*self.phi1*self.phi3

    def cal_mu3(self):
        ''' calculates cosolute #3 chemical potential in the bulk'''
        return self._log(self.phi3) + 1-self.nu3*self.phi1 -self.nu3/self.nu2*self.phi2 -self.phi3 + (1-self.phi3)*(self.chi13*self.nu3*self.phi1+self.chi23*self.nu3/self.nu2*self.phi2) - self.chi12*self.nu3*self.phi1*self.phi2    

    def cal_osm(self):
        ''' calculates the cosolutes osmotic pressure for ternary solution'''
        return -(self.mu1) / self.Vs

    def cal_FH_ideal_TS(self):
        ''' Calculate the cosolute-solvent FH mixing entropy '''
        return -(self.phi1*self._log(self.phi1) + 1/self.nu2*self.phi2*self._log(self.phi2) + 1/self.nu3*self.phi3*self._log(self.phi3))
    
    def cal_FH_nonideal_free_energy(self):
        ''' Calculate the cosolute-solvent FH mixing enthalpy '''
        return self.chi12*self.phi1*self.phi2 + self.chi13*self.phi1*self.phi3 + self.chi23/self.nu2*self.phi2*self.phi3

    def cal_FH_free_energy(self):
        ''' Calculate the cosolute-solvent FH mixing free energy '''
        return self.dGnonIdeal - self.TdSideal

class crowding_ter(cosolutes):
    '''
    Mean-field model class, contains class variable and methods that used to solve the folding thermodynamics of
    a protein solvated in a ternary mixuture (water, cosolute1, and cosolute2).

    
    Args:
        protien: A protein class - used to get protein parameters (SASA)
        cosolutes: A ternary mixutre cosolute class - used to get cosolute parameters (nu2, nu3, chi12, chi13, etc.)
        eps2, eps3: soft interaction parameters
        epsTs2, epsTS3: The entropic contribution of eps2 and eps3
        phiC_min: Minimal concetration (volume fraction)
        phiC_max: Maximal concetration (volume fraction)
        dphiC: Concetraion step (volume fraction)  
    '''
    
    def __init__(self, protein, cosolutes, eps2, eps3, epsTS2, epsTS3, eps23=0, epsTS23=0, phi2_min=0.0001, phi2_max=0.15, dphi2=0.0001, phi3_min=0.0001, phi3_max=0.15, dphi3=0.0001, T=298):
        self.T=T
        self.nu1 = 1
        self.nu2, self.chi12, self.chiTS12 = cosolutes.nu2, cosolutes.chi12, cosolutes.chiTS12
        self.nu3, self.chi13, self.chiTS13 = cosolutes.nu3, cosolutes.chi13, cosolutes.chiTS13
        self.chi23, self.chiTS23 = cosolutes.chi23, cosolutes.chiTS23
        self.eps2, self.epsTS2  = eps2, epsTS2
        self.eps3, self.epsTS3  = eps3, epsTS3
        self.eps23, self.epsTS23  = eps23, epsTS23
        self.SASA = protein.SASA
        self.chiH12, self.epsH2 = self.chi12 + self.chiTS12, self.eps2 + self.epsTS2
        self.chiH13, self.epsH3 = self.chi13 + self.chiTS13, self.eps3 + self.epsTS3
        self.chiH23, self.epsH23= self.chi23 + self.chiTS23, self.eps23 + self.epsTS23
        self.a1, self.a2, self.a3 = self.nu1**(1/3), self.nu2**(1/3), self.nu3**(1/3)
        
        self.phi2_min, self.phi2_max, self.dphi2 = phi2_min, phi2_max, dphi2
        self.phi3_min, self.phi3_max, self.dphi3 = phi3_min, phi3_max, dphi3
        phi2, phi3 = self.cal_phiC(self.phi2_min, self.phi2_max, self.dphi2), self.cal_phiC(self.phi3_min, self.phi3_max, self.dphi3)
        self.phi2, self.phi3 = np.meshgrid(phi2,phi3)
        self.phi1 = 1-self.phi2-self.phi3
        self.phi1[self.phi1<=0], self.phi2[self.phi1<=0], self.phi3[self.phi1<=0]=np.nan,np.nan,np.nan
        self.molar2 = self.phi2 / (self.nu2 * self.Vs)
        self.molar3 = self.phi3 / (self.nu3 * self.Vs)
        self.molal2 = self.phi2/self.phi1 / (18*self.nu2/1000)
        self.molal3 = self.phi3/self.phi1 / (18*self.nu3/1000)
        
        self.phi1s, self.phi2s, self.phi3s = np.zeros(self.phi1.shape), np.zeros(self.phi2.shape), np.zeros(self.phi3.shape)
        self.phi1s2, self.phi2s2 = np.zeros(self.phi1.shape), np.zeros(self.phi2.shape)
        self.phi1s3, self.phi2s3, self.phi3s3 = np.zeros(self.phi1.shape), np.zeros(self.phi2.shape), np.zeros(self.phi3.shape) 
        #self.Gamma_2, self.Gamma_3, self.Gamma_1_2, self.Gamma_1_3 = np.zeros(self.phi1.shape), np.zeros(self.phi1.shape), np.zeros(self.phi1.shape), np.zeros(self.phi1.shape)
        
        self.A, self.B, self.C = self.a1/self.a3, self.a1/self.a2, self.a2/self.a3
        self.F=np.exp(1-self.a3/self.a2)

        self.mu1, self.mu2, self.mu3 = self.cal_mu1(), self.cal_mu2(), self.cal_mu3()
        self.mu1s2, self.mu2s2 = np.zeros(self.phi1.shape), np.zeros(self.phi2.shape)
        self.mu1s3, self.mu2s3, self.mu3s3 = np.zeros(self.phi1.shape), np.zeros(self.phi2.shape), np.zeros(self.phi3.shape) 

        self.Ms3_Ms = np.zeros(self.phi1.shape)
        self.Ms2_Ms = np.zeros(self.phi1.shape)
        
        self.osm = self.cal_osm()
                
    def __str__(self):
        return "Mean-Field Model ternary:\nSoft_Interaction2-> (\u03B5\N{SUBSCRIPT TWO}={}, \u03B5ₛ\N{SUBSCRIPT TWO}={})\nSoft_Interaction3-> (\u03B5\N{SUBSCRIPT THREE}={}, \u03B5ₛ\N{SUBSCRIPT THREE}={})\nSoft_Interaction3-> (\u03B5\N{SUBSCRIPT TWO}\N{SUBSCRIPT THREE}={}, \u03B5ₛ\N{SUBSCRIPT TWO}\N{SUBSCRIPT THREE}={}) \nProtein-> (SASA={})\nCosolutes:\nCosolute2-> (\u03BD\N{SUBSCRIPT TWO}={}, \u03C7\N{SUBSCRIPT ONE}\N{SUBSCRIPT TWO}={}, \u03C7ₛ\N{SUBSCRIPT ONE}\N{SUBSCRIPT TWO}={})\nCosolute3-> (\u03BD\N{SUBSCRIPT THREE}={}, \u03C7\N{SUBSCRIPT ONE}\N{SUBSCRIPT THREE}={}, \u03C7ₛ\N{SUBSCRIPT ONE}\N{SUBSCRIPT THREE}={})\n(\u03C7ₛ\N{SUBSCRIPT TWO}\N{SUBSCRIPT THREE}={}, \u03C7ₛ\N{SUBSCRIPT TWO}\N{SUBSCRIPT THREE}={})".format(self.eps2, self.epsTS2,self.eps3, self.epsTS3, self.eps23, self.epsTS23, self.SASA, self.nu2, self.chi12, self.chiTS12, self.nu3, self.chi13, self.chiTS13, self.chi23, self.chiTS23)

    def for_all_SASA(self, data):
        ''' 
        Convert potential from value per protein domain volume to value for the entire protein

        Arg:
            data: A thermodynamic potential: Free energy, energy, or entropy.
        '''
        return data*(self.SASA/30**(2/3))*self.a3
        
    def cal_TdS_mix_s2(self):
        ''' Calculate the mixing entropy in sub-domain 2 '''
        return -self.Ms2_Ms*(self.phi1s2*self._log(self.phi1s2) + 1/self.nu2*self.phi2s2*self._log(self.phi2s2))

    def cal_TdS_mix_s3(self):
        ''' Calculate the mixing entropy in sub-domain 3 '''
        return -self.Ms3_Ms*(self.phi1s3*self._log(self.phi1s3) + 1/self.nu2*self.phi2s3*self._log(self.phi2s3) + 1/self.nu3*self.phi3s3*self._log(self.phi3s3)) 
        
    def cal_TdS_mix_s_tot(self):
        ''' Calculate the mixing entropy in the protein domain '''
        return self.TdS_mix_s2 + self.TdS_mix_s3

    def cal_dG_mix_chi_s2(self):
        ''' Calculate the non-ideal mixing free energy in sub-domain 2 '''
        return self.Ms2_Ms*(self.chi12*self.phi1s*self.phi2s + self.chi13*self.phi1s*self.phi3s + self.chi23/self.nu2*self.phi2s*self.phi3s) 
        
    def cal_dG_mix_eps_s2(self):
        ''' Calculate the soft-interactions mixing free energy in sub-domain 2 '''
        return 1/self.a3*( (self.eps2+self.eps23*self.phi3s)*(self.phi2s2*self.Ms2_Ms) )
    #    return self.Ms2_Ms*(self.eps2/self.a3*self.phi2s2)
        
    def cal_dG_mix_s2(self):
        ''' Calculate the mixing free energy in sub-domain 2 '''
        return self.dG_mix_chi_s2 + self.dG_mix_eps_s2 - self.TdS_mix_s2
        
    def cal_dG_mix_chi_s3(self):
        ''' Calculate the non-ideal mixing free energy in sub-domain 3 '''
        return self.Ms3_Ms*(self.chi12*self.phi1s*self.phi2s + self.chi13*self.phi1s*self.phi3s + self.chi23/self.nu2*self.phi2s*self.phi3s)
    
    def cal_dG_mix_eps_s3(self):
        ''' Calculate the soft-interactions mixing free energy in sub-domain 3 '''
        return self.Ms3_Ms/self.a3*((self.eps2+self.eps23*self.phi3)*self.F*self.phi2s3+self.eps3*self.phi3s3)
        #return self.Ms3_Ms/self.a3*(self.eps2*self.F*self.phi2s3+self.eps3*self.phi3s3)

    def cal_dG_mix_s3(self):
        ''' Calculate the mixing free energy in sub-domain 3 '''
        return self.dG_mix_chi_s3 + self.dG_mix_eps_s3 - self.TdS_mix_s3

    def cal_dG_mix_chi_s1(self):
        ''' Calculate the mixing free energy in sub-domain 1 '''
        return ( 1-self.Ms2_Ms-self.Ms3_Ms ) * (self.chi12*self.phi1s*self.phi2s + self.chi13*self.phi1s*self.phi3s + self.chi23/self.nu2*self.phi2s*self.phi3s)
    
    def cal_dG_mix_s_tot(self):
        ''' Calculate the mixing free energy in the protein domain '''
        return self.dG_mix_s1+self.dG_mix_s2+self.dG_mix_s3

    def cal_mu1s2(self, phi1s2 , phi2s2, phi1s3, phi2s3, phi3s3, phi1s, phi2s, phi3s, Ms2_Ms, Ms3_Ms):
        ''' Calculate the solvent chemical potential in sub-domain 2 '''
        return self._log(phi1s2)+1-phi1s2-phi2s2/self.nu2 + (1-phi1s)*(self.chi12*phi2s+self.chi13*phi3s) - self.chi23/self.nu2*phi2s*phi3s - self.eps23*phi3s/self.a3*(phi2s2*Ms2_Ms+self.F*phi2s3*Ms3_Ms)

    def cal_mu2s2(self, phi1s2 , phi2s2, phi1s3, phi2s3, phi3s3, phi1s, phi2s, phi3s, Ms2_Ms, Ms3_Ms):
        ''' Calculate cosolute 2 chemical potential in sub-domain 2 '''
        return self._log(phi2s2)+1-self.nu2*phi1s2-phi2s2 + self.nu2 * ( (1-phi2s)*(self.chi12*phi1s+self.chi23/self.nu2*phi3s)-self.chi13*phi1s*phi3s ) + self.nu2/self.a3*(self.eps2+self.eps23*phi3s*(1-phi2s2*Ms2_Ms-self.F*phi2s3*Ms3_Ms))
        
    def cal_mu1s3(self, phi1s2 , phi2s2, phi1s3, phi2s3, phi3s3, phi1s, phi2s, phi3s, Ms2_Ms, Ms3_Ms):
        ''' Calculate the solvent chemical potential in sub-domain 3 '''
        return self._log(phi1s3)+(1-phi1s3)-phi2s3/self.nu2-phi3s3/self.nu3 + (1-phi1s)*(self.chi12*phi2s+self.chi13*phi3s) - self.chi23/self.nu2*phi2s*phi3s - self.eps23*phi3s/self.a3*(phi2s2*Ms2_Ms+self.F*phi2s3*Ms3_Ms)
        
    def cal_mu2s3(self, phi1s2 , phi2s2, phi1s3, phi2s3, phi3s3, phi1s, phi2s, phi3s, Ms2_Ms, Ms3_Ms):
        ''' Calculate cosolute 2 chemical potential in sub-domain 3 '''
        return self._log(phi2s3)+(1-self.nu2*phi1s3)-phi2s3-self.nu2/self.nu3*phi3s3 + self.nu2 * ( (1-phi2s)*(self.chi12*phi1s+self.chi23/self.nu2*phi3s)-self.chi13*phi1s*phi3s ) + self.nu2/self.a3*(self.F*self.eps2+self.eps23*phi3s*(self.F-phi2s2*Ms2_Ms-self.F*phi2s3*Ms3_Ms)) 
       
    def cal_mu3s3(self, phi1s2 , phi2s2, phi1s3, phi2s3, phi3s3, phi1s, phi2s, phi3s, Ms2_Ms, Ms3_Ms):
        ''' Calculate cosolute 3 chemical potential in sub-domain 3 '''
        return self._log(phi3s3)+(1-self.nu3*phi1s3)-self.nu3/self.nu2*phi2s3-phi3s3 + self.nu3 * ( -self.chi12*phi1s*phi2s + (1-phi3s)*(self.chi13*phi1s + self.chi23/self.nu2*phi2s) ) + self.nu3/self.a3*(self.eps3 + self.eps23*(1-phi3s)*(phi2s2*Ms2_Ms+self.F*phi2s3*Ms3_Ms))
        
    def cal_mu1s2_global(self):
        return self._log(self.phi1s2)+1-self.phi1s2-self.phi2s2/self.nu2 + (1-self.phi1s)*(self.chi12*self.phi2s+self.chi13*self.phi3s) - self.chi23/self.nu2*self.phi2s*self.phi3s - self.eps23*self.phi3s/self.a3*(self.phi2s2*self.Ms2_Ms+self.F*self.phi2s3*self.Ms3_Ms)
   
    def cal_mu2s2_global(self):
        return self._log(self.phi2s2)+1-self.nu2*self.phi1s2-self.phi2s2 + self.nu2 * ( (1-self.phi2s)*(self.chi12*self.phi1s+self.chi23/self.nu2*self.phi3s)-self.chi13*self.phi1s*self.phi3s ) + self.nu2/self.a3*(self.eps2+self.eps23*self.phi3s*(1-self.phi2s2*self.Ms2_Ms-self.F*self.phi2s3*self.Ms3_Ms))
        
    def cal_mu1s3_global(self):
        return self._log(self.phi1s3)+(1-self.phi1s3)-self.phi2s3/self.nu2-self.phi3s3/self.nu3 + (1-self.phi1s)*(self.chi12*self.phi2s+self.chi13*self.phi3s) - self.chi23/self.nu2*self.phi2s*self.phi3s - self.eps23*self.phi3s/self.a3*(self.phi2s2*self.Ms2_Ms+self.F*self.phi2s3*self.Ms3_Ms) 
        
    def cal_mu2s3_global(self):
        return self._log(self.phi2s3)+(1-self.nu2*self.phi1s3)-self.phi2s3-self.nu2/self.nu3*self.phi3s3 + self.nu2 * ( (1-self.phi2s)*(self.chi12*self.phi1s+self.chi23/self.nu2*self.phi3s)-self.chi13*self.phi1s*self.phi3s ) + self.nu2/self.a3*(self.F*self.eps2+self.eps23*self.phi3s*(self.F-self.phi2s2*self.Ms2_Ms-self.F*self.phi2s3*self.Ms3_Ms)) 
       
    def cal_mu3s3_global(self):
        return self._log(self.phi3s3)+(1-self.nu3*self.phi1s3)-self.nu3/self.nu2*self.phi2s3-self.phi3s3 + self.nu3 * ( -self.chi12*self.phi1s*self.phi2s + (1-self.phi3s)*(self.chi13*self.phi1s + self.chi23/self.nu2*self.phi2s) )  + self.nu3/self.a3*(self.eps3 + self.eps23*(1-self.phi3s)*(self.phi2s2*self.Ms2_Ms+self.F*self.phi2s3*self.Ms3_Ms))
   
    def cal_ddG_nu(self):
        ''' Calculate the excluded volume folding free energy '''
        return self.phi1s-self.phi1+self.phi1s*np.log(self.phi1)-self.phi1s2*self.Ms2_Ms*np.log(self.phi1s2)-self.phi1s3*self.Ms3_Ms*np.log(self.phi1s3)+ 1/self.nu2*( self.phi2s-self.phi2+self.phi2s*np.log(self.phi2)-self.phi2s2*self.Ms2_Ms*np.log(self.phi2s2)-self.phi2s3*self.Ms3_Ms*np.log(self.phi2s3) ) + 1/self.nu3*( self.phi3s-self.phi3+self.phi3s*np.log(self.phi3)-self.phi3s3*self.Ms3_Ms*np.log(self.phi3s3) ) 
        
    def cal_ddG_chi(self):
        ''' Calculate the non-ideal folding free energy '''
        return  self.cal_ddG_chi12_chi13() + self.cal_ddG_chi23()

    def cal_ddG_chi12_chi13(self):
        return self.chi12*(self.phi1-self.phi1s)*(self.phi2s-self.phi2) + self.chi13*(self.phi1-self.phi1s)*(self.phi3s-self.phi3)
    
    def cal_ddG_chi23(self):
        return self.chi23/self.nu2*(self.phi2-self.phi2s)*(self.phi3s-self.phi3)
        
    def cal_ddG_eps(self):
        ''' Calculate the soft interactions folding free energy '''
        return -1/self.a3*((self.eps2+self.eps23*self.phi3s)*(self.phi2s2*self.Ms2_Ms + self.F*self.phi2s3*self.Ms3_Ms) + self.eps3*self.phi3s) 

    def cal_ddG_eps12_eps13(self):
        ''' Calculate the soft interactions folding free energy '''
        return -1/self.a3*((self.eps2+0*self.phi3s)*(self.phi2s2*self.Ms2_Ms + self.F*self.phi2s3*self.Ms3_Ms) + self.eps3*self.phi3s) 

    def cal_ddG_eps23(self):
        ''' Calculate the soft interactions folding free energy '''
        return self.cal_ddG_eps()-self.cal_ddG_eps12_eps13()
   
    def cal_ddG(self):
        ''' Calculate the folding free energy '''
        return self.cal_ddG_nu() + self.cal_ddG_chi() + self.cal_ddG_eps()
        
    def cal_ddH_chi(self):
        ''' Calculate the non-ideal folding enthalpy '''
        return self.cal_ddH_chi12_chi13() + self.cal_ddH_chi23()

    def cal_ddH_chi12_chi13(self):
        ''' Calculate the non-ideal folding enthalpy '''
        return self.chiH12*(self.phi1-self.phi1s)*(self.phi2s-self.phi2) + self.chiH13*(self.phi1-self.phi1s)*(self.phi3s-self.phi3) 

    def cal_ddH_chi23(self):
        ''' Calculate the non-ideal folding enthalpy '''
        return self.chiH23/self.nu2*(self.phi2-self.phi2s)*(self.phi3s-self.phi3)

    def cal_ddH_eps(self):
        ''' Calculate the soft interactions folding enthalpy '''
        return -1/self.a3*((self.epsH2+self.epsH23*self.phi3s)*(self.phi2s2*self.Ms2_Ms + self.F*self.phi2s3*self.Ms3_Ms) + self.epsH3*self.phi3s) 

    def cal_ddH_eps12_eps13(self):
        ''' Calculate the soft interactions folding enthalpy '''
        return -1/self.a3*((self.epsH2+0*self.phi3s)*(self.phi2s2*self.Ms2_Ms + self.F*self.phi2s3*self.Ms3_Ms) + self.epsH3*self.phi3s) 

    def cal_ddH_eps23(self):
        ''' Calculate the soft interactions folding enthalpy '''
        return self.cal_ddH_eps()-self.cal_ddH_eps12_eps13()

    def cal_ddH(self):
        ''' Calculate the folding enthalpy '''
        return self.cal_ddH_chi() + self.cal_ddH_eps()

    def cal_Gamma_2(self):
        ''' Calculate the PCI of cosolute 2 around the protein '''
        return -(self.phi2s/self.nu2*(1-self.phi2/self.phi1 * self.phi1s/self.phi2s))

    def cal_Gamma_3(self):
        ''' Calculate the PCI of cosolute 3 around the protein '''
        return -(self.phi3s/self.nu3*(1-self.phi3/self.phi1 * self.phi1s/self.phi3s))

    def cal_Gamma_1_3(self):
        ''' Calculate the PCI of solvent around the protein holding mu3 constant '''
        return -(self.phi1s*(1-self.phi1/self.phi2 * self.phi2s/self.phi1s))

    def cal_Gamma_1_2(self):
        ''' Calculate the PCI of solvent around the protein holding mu2 constant '''
        return -(self.phi1s*(1-self.phi1/self.phi3 * self.phi3s/self.phi1s))


    def equil_cond(self,phiSlist,i,j):
        ''' equilibrium condition, solved by SciPy's fsolve '''
        phi3s, phi2s2, phi2s = phiSlist
        phi1s, phi1s2  = 1-phi2s-phi3s, 1-phi2s2

        Ms3_Ms = 1-0.5*( (phi1s+phi2s) - self.C*phi2s - self.A*phi1s)
        Ms2_Ms = 0.5*( (phi1s+phi2s)*(1-self.C) )
        phi3s3=phi3s/Ms3_Ms
        phi2s3=(phi2s-phi2s2*Ms2_Ms)/Ms3_Ms
        phi1s3= 1-phi2s3-phi3s3
        
        if phi1s<=0 or phi2s<=0 or phi3s<=0 or phi1s2<=0 or phi2s2<=0 or phi1s3<=0 or phi2s3<=0 or phi3s3<=0:
            return [100,100,100]    
        else:
            mu1s2 = self.cal_mu1s2(phi1s2 , phi2s2, phi1s3, phi2s3, phi3s3, phi1s, phi2s, phi3s, Ms2_Ms, Ms3_Ms)
            mu2s2 = self.cal_mu2s2(phi1s2 , phi2s2, phi1s3, phi2s3, phi3s3, phi1s, phi2s, phi3s, Ms2_Ms, Ms3_Ms)
            mu1s3 = self.cal_mu1s3(phi1s2 , phi2s2, phi1s3, phi2s3, phi3s3, phi1s, phi2s, phi3s, Ms2_Ms, Ms3_Ms)     
            mu2s3, mu3s3 = self.cal_mu2s3(phi1s2 , phi2s2, phi1s3, phi2s3, phi3s3, phi1s, phi2s, phi3s, Ms2_Ms, Ms3_Ms), self.cal_mu3s3(phi1s2 , phi2s2, phi1s3, phi2s3, phi3s3, phi1s, phi2s, phi3s, Ms2_Ms, Ms3_Ms)     
            
            return [mu2s2-self.mu2[i,j]+self.nu2*(self.mu1[i,j]-mu1s2),
                    mu2s3-self.mu2[i,j]+self.nu2*(self.mu1[i,j]-mu1s3),
                    mu3s3-self.mu3[i,j]+self.nu3*(self.mu1[i,j]-mu1s3)]


    def solve_equil(self, print_msg: bool = True):
        '''Solve the equilibrium condition over the phi2/phi3 grid.

        Records per-point convergence diagnostics in:
          - self.converged: bool array (same shape as phi2)
          - self.flags: int array of fsolve ier
          - self.messages: object array of fsolve messages
        Optionally prints failures when print_msg=True.
        '''
        shape = self.phi2.shape
        self.converged = np.zeros(shape, dtype=bool)
        self.flags = np.zeros(shape, dtype=int)
        self.messages = np.empty(shape, dtype=object)
        self.messages[:] = ''

        I, J = [], []
        for i in range(shape[0]):
            for j in range(shape[1]):
                if np.isnan(self.phi1[i, j]):
                    self.phi2s[i, j], self.phi3s[i, j], self.phi2s2[i, j], self.phi2s3[i, j], self.phi3s3[i, j] = np.nan, np.nan, np.nan, np.nan, np.nan
                    self.flags[i, j] = 0
                    self.messages[i, j] = 'nan in bulk composition'
                    continue

                if i == 0 and j == 0:
                    guess = (self.phi3[i, j], self.phi2[i, j], self.phi2[i, j])
                elif i == 0:
                    guess = (self.phi3s[i, j-1], self.phi2s2[i, j-1], self.phi2s[i, j-1])
                elif j == 0:
                    guess = (self.phi3s[i-1, j], self.phi2s2[i-1, j], self.phi2s[i-1, j])
                else:
                    guess = (self.phi3s[i-1, j-1], self.phi2s2[i-1, j-1], self.phi2s[i-1, j-1])

                output = fsolve(self.equil_cond, guess, args=(i, j), xtol=1.49012e-12, maxfev=10000, full_output=True)
                self.phi3s[i, j], self.phi2s2[i, j], self.phi2s[i, j] = output[0]
                flag, mesg = output[2], output[3]
                self.flags[i, j] = int(flag)
                self.messages[i, j] = str(mesg)
                if flag != 1:
                    if print_msg:
                        print(self.phi2[i, j], self.phi3[i, j], mesg)
                    I.append(i)
                    J.append(j)
                else:
                    self.converged[i, j] = True

        for i, j in zip(I, J):
            self.phi2s[i, j], self.phi3s[i, j], self.phi2s2[i, j], self.phi2s3[i, j], self.phi3s3[i, j] = np.nan, np.nan, np.nan, np.nan, np.nan
        self.phi1s = 1 - self.phi2s - self.phi3s
        self.phi1s2 = 1-self.phi2s2
        
        self.Ms3_Ms = 1-0.5*( (self.phi1s+self.phi2s) - self.C*self.phi2s - self.A*self.phi1s)
        self.Ms2_Ms = 0.5*( (self.phi1s+self.phi2s)*(1-self.C) )
        check=self.Ms2_Ms.copy()
        check[np.isnan(check)]=0 
        if  np.all(check==0):
            self.Ms_Ms2=0
        else:
            self.Ms_Ms2=self.Ms2_Ms**(-1)
        self.phi3s3 = self.phi3s/self.Ms3_Ms  
        self.phi2s3=(self.phi2s-self.phi2s2*self.Ms2_Ms)/self.Ms3_Ms
        self.phi1s3 = 1-self.phi2s3-self.phi3s3
        
        self.mu1s2, self.mu2s2 = self.cal_mu1s2_global(), self.cal_mu2s2_global()
        self.mu1s3, self.mu2s3, self.mu3s3 = self.cal_mu1s3_global(), self.cal_mu2s3_global(), self.cal_mu3s3_global()     

        self.ddG_nu_Ms = self.cal_ddG_nu()
        self.ddG_chi_Ms = self.cal_ddG_chi()
        self.ddG_chi12_chi13_Ms = self.cal_ddG_chi12_chi13()
        self.ddG_chi23_Ms = self.cal_ddG_chi23()
        self.ddG_eps_Ms = self.cal_ddG_eps()
        self.ddG_eps12_eps13_Ms = self.cal_ddG_eps12_eps13()
        self.ddG_eps23_Ms = self.cal_ddG_eps23()
        self.ddG_Ms = self.cal_ddG()
        
        self.ddH_chi_Ms = self.cal_ddH_chi()
        self.ddH_chi12_chi13_Ms = self.cal_ddH_chi12_chi13()
        self.ddH_chi23_Ms = self.cal_ddH_chi23()
        self.ddH_eps_Ms = self.cal_ddH_eps()
        self.ddH_eps12_eps13_Ms = self.cal_ddH_eps12_eps13()
        self.ddH_eps23_Ms = self.cal_ddH_eps23()
        self.ddH_Ms = self.cal_ddH()
        
        self.TddS_nu_Ms = -self.ddG_nu_Ms
        self.TddS_chi_Ms = self.ddH_chi_Ms-self.ddG_chi_Ms
        self.TddS_chi12_chi13_Ms = self.ddH_chi12_chi13_Ms-self.ddG_chi12_chi13_Ms
        self.TddS_chi23_Ms = self.ddH_chi23_Ms-self.ddG_chi23_Ms
        self.TddS_eps_Ms = self.ddH_eps_Ms-self.ddG_eps_Ms
        self.TddS_eps12_eps13_Ms = self.ddH_eps12_eps13_Ms-self.ddG_eps12_eps13_Ms
        self.TddS_eps23_Ms = self.ddH_eps23_Ms-self.ddG_eps23_Ms
        self.TddS_Ms = self.ddH_Ms-self.ddG_Ms
        
        self.ddG_nu = self.for_all_SASA(self.ddG_nu_Ms)
        self.ddG_chi = self.for_all_SASA(self.ddG_chi_Ms)
        self.ddG_chi12_chi13 = self.for_all_SASA(self.ddG_chi12_chi13_Ms)
        self.ddG_chi23 = self.for_all_SASA(self.ddG_chi23_Ms)
        self.ddG_eps = self.for_all_SASA(self.ddG_eps_Ms)
        self.ddG_eps12_eps13 = self.for_all_SASA(self.ddG_eps12_eps13_Ms)
        self.ddG_eps23 = self.for_all_SASA(self.ddG_eps23_Ms)
        self.ddG = self.for_all_SASA(self.ddG_Ms)

        self.ddH_chi = self.for_all_SASA(self.ddH_chi_Ms)
        self.ddH_chi12_chi13 = self.for_all_SASA(self.ddH_chi12_chi13_Ms)
        self.ddH_chi23 = self.for_all_SASA(self.ddH_chi23_Ms)
        self.ddH_eps = self.for_all_SASA(self.ddH_eps_Ms)
        self.ddH_eps12_eps13 = self.for_all_SASA(self.ddH_eps12_eps13_Ms)
        self.ddH_eps23 = self.for_all_SASA(self.ddH_eps23_Ms)
        self.ddH = self.for_all_SASA(self.ddH_Ms)
        
        self.TddS_nu = self.for_all_SASA(self.TddS_nu_Ms)
        self.TddS_chi = self.for_all_SASA(self.TddS_chi_Ms)
        self.TddS_chi12_chi13 = self.for_all_SASA(self.TddS_chi12_chi13_Ms)
        self.TddS_chi23 = self.for_all_SASA(self.TddS_chi23_Ms)
        self.TddS_eps = self.for_all_SASA(self.TddS_eps_Ms)
        self.TddS_eps12_eps13 = self.for_all_SASA(self.TddS_eps12_eps13_Ms)
        self.TddS_eps23 = self.for_all_SASA(self.TddS_eps23_Ms)
        self.TddS = self.for_all_SASA(self.TddS_Ms)
        
        self.ddG_nu_kJ = self._to_kj(self.ddG_nu)
        self.ddG_chi_kJ = self._to_kj(self.ddG_chi)
        self.ddG_chi12_chi13_kJ = self._to_kj(self.ddG_chi12_chi13)
        self.ddG_chi23_kJ = self._to_kj(self.ddG_chi23)
        self.ddG_eps_kJ = self._to_kj(self.ddG_eps)
        self.ddG_eps12_eps13_kJ = self._to_kj(self.ddG_eps12_eps13)
        self.ddG_eps23_kJ = self._to_kj(self.ddG_eps23)
        self.ddG_kJ = self._to_kj(self.ddG)
        
        self.ddH_chi_kJ = self._to_kj(self.ddH_chi)
        self.ddH_chi12_chi13_kJ = self._to_kj(self.ddH_chi12_chi13)
        self.ddH_chi23_kJ = self._to_kj(self.ddH_chi23)
        self.ddH_eps_kJ = self._to_kj(self.ddH_eps)
        self.ddH_eps12_eps13_kJ = self._to_kj(self.ddH_eps12_eps13)
        self.ddH_eps23_kJ = self._to_kj(self.ddH_eps23)
        self.ddH_kJ = self._to_kj(self.ddH)
        
        self.TddS_nu_kJ = self._to_kj(self.TddS_nu)
        self.TddS_chi_kJ = self._to_kj(self.TddS_chi)
        self.TddS_chi12_chi13_kJ = self._to_kj(self.TddS_chi12_chi13)
        self.TddS_chi23_kJ = self._to_kj(self.TddS_chi23)
        self.TddS_eps_kJ = self._to_kj(self.TddS_eps)
        self.TddS_eps12_eps13_kJ = self._to_kj(self.TddS_eps12_eps13)
        self.TddS_eps23_kJ = self._to_kj(self.TddS_eps23)
        self.TddS_kJ = self._to_kj(self.TddS)
                
        if self.a3==self.a2:
            self.phi1s2+=np.nan
            self.phi2s2+=np.nan

        self.TdS_mix_s2 = self.cal_TdS_mix_s2()
        self.TdS_mix_s3 = self.cal_TdS_mix_s3()
        self.TdS_mix_s_tot = self.cal_TdS_mix_s_tot()

        self.dG_mix_chi_s1 = self.cal_dG_mix_chi_s1()
        self.dG_mix_chi_s2 = self.cal_dG_mix_chi_s2()
        self.dG_mix_eps_s2 = self.cal_dG_mix_eps_s2()
        self.dG_mix_chi_s3 = self.cal_dG_mix_chi_s3()
        self.dG_mix_eps_s3 = self.cal_dG_mix_eps_s3()
        
        self.dG_mix_s1 = self.cal_dG_mix_chi_s1()
        self.dG_mix_s2 = self.cal_dG_mix_s2()
        self.dG_mix_s3 = self.cal_dG_mix_s3()
        self.dG_mix_s_tot = self.cal_dG_mix_s_tot()

        self.Gamma_2_Ms = self.cal_Gamma_2()
        self.Gamma_3_Ms = self.cal_Gamma_3()
        self.Gamma_1_2_Ms = self.cal_Gamma_1_2()
        self.Gamma_1_3_Ms = self.cal_Gamma_1_3()

        self.Gamma_2 = self.for_all_SASA(self.Gamma_2_Ms)
        self.Gamma_3 = self.for_all_SASA(self.Gamma_3_Ms)
        self.Gamma_1_2 = self.for_all_SASA(self.Gamma_1_2_Ms)
        self.Gamma_1_3 = self.for_all_SASA(self.Gamma_1_3_Ms)
        
    def get_solver_report(self) -> pd.DataFrame:
        '''Return a DataFrame with per-grid-point fsolve diagnostics.'''
        data = {
            'phi2': self.phi2.flatten(),
            'phi3': self.phi3.flatten(),
            'converged': self.converged.flatten(),
            'flag': self.flags.flatten(),
            'message': self.messages.flatten(),
        }
        return pd.DataFrame(data)

        

        
   
        
    def fit_eps(self, exp_conc2, exp_conc3, exp_ddG, concentration_type='phi', disp=True,
                fit_eps2=True, fit_eps3=True, fit_eps23=False,
                method: str = 'Powell', bounds=None, robust: bool = True, max_nfev: int = 200):
        ''' 
        Fit the experimental folding free energy to resolve the soft interaction parameters

        Arg:
            exp_conc2: Matrix of molal or volume fraction of cosolute 2 in the experiment
            exp_conc3: Matrix of molal or volume fraction of cosolute 3 in the experiment
            exp_ddG: Matrix of folding free energy in experiment in kJ/mol
            concentration_type: type of concetration. str - 'phi' or 'molal'
        '''
        if concentration_type == 'phi':
            pass
        elif concentration_type=='molal':
            exp_conc2 = exp_conc2 * (18 * self.nu2) * 1E-3 / (1+exp_conc2 * (18 * self.nu2) * 1E-3+exp_conc3 * (18 * self.nu3) * 1E-3)
            exp_conc3 = exp_conc3 * (18 * self.nu3) * 1E-3 / (1+exp_conc2 * (18 * self.nu2) * 1E-3+exp_conc3 * (18 * self.nu3) * 1E-3)
        else:
            raise Exception("Concetration type can be either molal or volume fraction")
        model_phi1, model_phi2, model_phi3 = self.phi1, self.phi2, self.phi3 # save model arrays
        # use experiment arrays for the fit
        self.phi2 = exp_conc2
        self.phi3 = exp_conc3
        
        self.phi1 = 1-self.phi2-self.phi3
        self.phi2s,self.phi3s = np.zeros(self.phi2.shape), np.zeros(self.phi3.shape)
        self.phi2s2, self.phi2s3, self.phi3s3 = np.zeros(self.phi2.shape), np.zeros(self.phi2.shape), np.zeros(self.phi3.shape)
        self.mu1, self.mu2, self.mu3 = self.cal_mu1(), self.cal_mu2(), self.cal_mu3()
        self.flag=True
        # Build initial guess vector
        x0 = []
        if fit_eps2: x0.append(self.eps2)
        if fit_eps3: x0.append(self.eps3)
        if fit_eps23: x0.append(self.eps23)

        # Residual vector for robust least-squares
        def residuals(x):
            k = 0
            if fit_eps2:
                self.eps2 = x[k]; k += 1
            if fit_eps3:
                self.eps3 = x[k]; k += 1
            if fit_eps23:
                self.eps23 = x[k]; k += 1
            self.solve_equil(print_msg=False)
            res = np.asarray(exp_ddG) - np.asarray(self.ddG_kJ)
            return res[np.isfinite(res)]

        if method == 'least_squares':
            from scipy.optimize import least_squares as _ls
            loss = 'soft_l1' if robust else 'linear'
            self.res = _ls(residuals, x0, loss=loss, max_nfev=max_nfev,
                           bounds=bounds if bounds is not None else (-np.inf, np.inf))
        else:
            def sse(x):
                r = residuals(x)
                return float(np.dot(r, r))
            self.res = minimize(sse, x0, method=method, bounds=bounds,
                                options={'disp': disp, 'maxiter': 1000})
        # return to model arrays
        self.phi1, self.phi2, self.phi3 = model_phi1, model_phi2, model_phi3
        self.mu1, self.mu2, self.mu3 = self.cal_mu1(), self.cal_mu2(), self.cal_mu3()
        self.phi2s,self.phi3s = np.zeros(self.phi2.shape), np.zeros(self.phi3.shape)
        self.phi2s2, self.phi2s3, self.phi3s3 = np.zeros(self.phi2.shape), np.zeros(self.phi2.shape), np.zeros(self.phi3.shape)
        # solve for final eps
        self.solve_equil()
        self.to_pandas()

    def fit_epsTS(self, exp_conc2, exp_conc3, exp_ddH, exp_TddS,
                  concentration_type='phi', disp=True,
                  fit_epsTS2=True, fit_epsTS3=True, fit_epsTS23=False,
                  method: str = 'least_squares', bounds=None, robust: bool = True, max_nfev: int = 300):
        ''' 
        Fit the experimental folding free energy to resolve the soft interaction parameter, eps

        Arg:
            exp_conc2: Matrix of molal or volume fraction of cosolute 2 in the experiment
            exp_conc3: Matrix of molal or volume fraction of cosolute 3 in the experiment
            exp_ddH: Folding enthalpy in experiment in kJ/mol
            exp_TddS: Folding entropy in experiment in kJ/mol 
            concentration_type: type of concetration. str - 'phi' or 'molal'
        '''
        if concentration_type == 'phi':
            pass
        elif concentration_type=='molal':
            exp_conc2 = exp_conc2 * (18 * self.nu2) * 1E-3 / (1+exp_conc2 * (18 * self.nu2) * 1E-3+exp_conc3 * (18 * self.nu3) * 1E-3)
            exp_conc3 = exp_conc3 * (18 * self.nu3) * 1E-3 / (1+exp_conc2 * (18 * self.nu2) * 1E-3+exp_conc3 * (18 * self.nu3) * 1E-3)
        else:
            raise Exception("Concetration type can be either molal or volume fraction")
        model_phi1, model_phi2, model_phi3 = self.phi1, self.phi2, self.phi3 # save model arrays
        model_phi1s2, model_phi2s2, model_phi1s3, model_phi2s3, model_phi3s3 = self.phi1s2, self.phi2s2, self.phi1s3, self.phi2s3, self.phi3s3 
        # use experiment arrays for the fit
        self.phi2 = exp_conc2
        self.phi3 = exp_conc3
        
        self.phi1 = 1-self.phi2-self.phi3
        self.phi2s,self.phi3s = np.zeros(self.phi2.shape), np.zeros(self.phi3.shape)
        self.phi2s2, self.phi2s3, self.phi3s3 = np.zeros(self.phi2.shape), np.zeros(self.phi2.shape), np.zeros(self.phi3.shape)
        self.mu1, self.mu2, self.mu3 = self.cal_mu1(), self.cal_mu2(), self.cal_mu3()
        self.flag=True

        # Build initial guess vector
        x0 = []
        if fit_epsTS2: x0.append(self.epsTS2)
        if fit_epsTS3: x0.append(self.epsTS3)
        if fit_epsTS23: x0.append(self.epsTS23)

        # Residual vector concatenating enthalpy and entropy
        def residuals(x):
            k = 0
            if fit_epsTS2:
                self.epsTS2 = x[k]; k += 1
            if fit_epsTS3:
                self.epsTS3 = x[k]; k += 1
            if fit_epsTS23:
                self.epsTS23 = x[k]; k += 1
            # update enthalpic parts
            self.epsH2 = self.eps2 + self.epsTS2
            self.epsH3 = self.eps3 + self.epsTS3
            self.epsH23 = self.eps23 + self.epsTS23 if hasattr(self, 'eps23') else 0.0
            self.solve_equil(print_msg=False)
            r1 = np.asarray(exp_ddH) - np.asarray(self.ddH_kJ)
            r2 = np.asarray(exp_TddS) - np.asarray(self.TddS_kJ)
            r = np.concatenate([r1[np.isfinite(r1)], r2[np.isfinite(r2)]])
            return r

        if method == 'least_squares':
            from scipy.optimize import least_squares as _ls
            loss = 'soft_l1' if robust else 'linear'
            self.resTS = _ls(residuals, x0, loss=loss, max_nfev=max_nfev,
                             bounds=bounds if bounds is not None else (-np.inf, np.inf))
        else:
            def sse(x):
                r = residuals(x)
                return float(np.dot(r, r))
            self.resTS = minimize(sse, x0, method=method, bounds=bounds,
                                  options={'disp': disp, 'maxiter': 1000})
        
        
        # return to model arrays
        self.phi1, self.phi2, self.phi3 = model_phi1, model_phi2, model_phi3
        self.mu1, self.mu2, self.mu3 = self.cal_mu1(), self.cal_mu2(), self.cal_mu3()
        self.phi2s,self.phi3s = np.zeros(self.phi2.shape), np.zeros(self.phi3.shape)
        self.phi2s2, self.phi2s3, self.phi3s3 = np.zeros(self.phi2.shape), np.zeros(self.phi2.shape), np.zeros(self.phi3.shape)
        # solve for final eps
        self.solve_equil()
        #self.to_pandas()

    def msd_fit_eps_2(self,eps, ddG):
        self.eps2 = eps[0]
        self.solve_equil(print_msg=False)
        return np.nansum((ddG-self.ddG_kJ)**2)

    def msd_fit_eps_3(self,eps, ddG):
        self.eps3 = eps[0]
        self.solve_equil(print_msg=False)
        return np.nansum((ddG-self.ddG_kJ)**2)
        
    def msd_fit_eps_2_3(self,eps, ddG):
        self.eps2 = eps[0]
        self.eps3 = eps[1]
        self.solve_equil(print_msg=False)
        return np.nansum((ddG-self.ddG_kJ)**2)
    
    def msd_fit_eps_2_3_23(self,eps, ddG):
        self.eps2 = eps[0]
        self.eps3 = eps[1]
        self.eps23= eps[2]
        self.solve_equil(print_msg=False)
        return np.nansum((ddG-self.ddG_kJ)**2)
    
    def msd_fit_eps_23_only(self,eps, ddG):
        self.eps23= eps[0]
        self.solve_equil(print_msg=False)
        return np.nansum((ddG-self.ddG_kJ)**2)
        
    def msd_fit_epsTS_2(self,epsTS, ddH, TddS):
        self.epsTS2=epsTS[0]
        self.epsH2 = self.eps2 + self.epsTS2
        self.epsH3 = self.eps3 + self.epsTS3
        self.solve_equil()
        return np.nansum((ddH-self.ddH_kJ)**2) + np.nansum((TddS-self.TddS_kJ)**2)  

    def msd_fit_epsTS_3(self,epsTS, ddH, TddS):
        self.epsTS3=epsTS[0]
        self.epsH2 = self.eps2 + self.epsTS2
        self.epsH3 = self.eps3 + self.epsTS3
        self.solve_equil()
        return np.nansum((ddH-self.ddH_kJ)**2) + np.nansum((TddS-self.TddS_kJ)**2)  

    def msd_fit_epsTS_2_3(self,epsTS, ddH, TddS):
        self.epsTS2=epsTS[0]
        self.epsTS3=epsTS[1]
        self.epsH2 = self.eps2 + self.epsTS2
        self.epsH3 = self.eps3 + self.epsTS3
        self.solve_equil()
        return np.nansum((ddH-self.ddH_kJ)**2) + np.nansum((TddS-self.TddS_kJ)**2)  

    def msd_fit_epsTS_2_3_23(self,epsTS, ddH, TddS):
        self.epsTS2=epsTS[0]
        self.epsTS3=epsTS[1]
        self.epsTS23=epsTS[2]
        self.epsH2 = self.eps2 + self.epsTS2
        self.epsH3 = self.eps3 + self.epsTS3
        self.epsH23 = self.eps23 + self.epsTS23
        self.solve_equil()
        return np.nansum((ddH-self.ddH_kJ)**2) + np.nansum((TddS-self.TddS_kJ)**2)  

    def msd_fit_epsTS_23_only(self,epsTS, ddH, TddS):
        self.epsTS23=epsTS[0]
        self.epsH23 = self.eps23 + self.epsTS23
        self.solve_equil()
        return np.nansum((ddH-self.ddH_kJ)**2) + np.nansum((TddS-self.TddS_kJ)**2)  
    
    def pade(self, xy, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19):
        x, y = xy
        return (a1+a2*x+a3*x**2+a4*x**3+a5*y+a6*y**2+a7*y**3+a8*x*y+a9*(x**2)*y+a10*x*(y**2))/ (1+a11*x+a12*x**2+a13*x**3+a14*y+a15*y**2+a16*y**3+a17*x*y+a18*(x**2)*y+a19*x*(y**2))

    def fit_ddG_mu2_mu3(self):
        # Sample matrices x, y, and z
        x = self.mu2
        y = self.mu3
        z = self.ddG
                
        # Flatten the 2D matrices
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        
        # Fit the data
        popt, pcov = curve_fit(self.pade, (x_flat, y_flat), z_flat)

        n_X=100#self.mu2.shape[1]
        n_Y=100#self.mu3.shape[0]
        
        X=np.linspace(x.min(),x.max(),n_X)
        Y=np.linspace(y.min(),y.max(),n_Y)
        X,Y=np.meshgrid(X,Y)

        return self.pade((X, Y), *popt), X, Y
        

    
    def cal_gamma_der(self):
        # fit ddG vs mu2 and mu3
        ddG_predicted, mu2_mat, mu3_mat = self.fit_ddG_mu2_mu3()
    
        # Calculate the partial derivative along the axes (mu2 and mu3)
        gamma_2 = (np.roll(ddG_predicted, -1, axis=1) - ddG_predicted) / (np.roll(mu2_mat, -1, axis=1) - mu2_mat)
        gamma_3 = (np.roll(ddG_predicted, -1, axis=0) - ddG_predicted) / (np.roll(mu3_mat, -1, axis=0) - mu3_mat)
    
        # Handle boundary conditions by using backward difference at the end (optional)
        gamma_2[:, -1] = (ddG_predicted[:, -1] - ddG_predicted[:, -2]) / (mu2_mat[:, -1] - mu2_mat[:, -2])
        gamma_3[-1, :] = (ddG_predicted[-1, :] - ddG_predicted[-2, :]) / (mu3_mat[-1, :] - mu3_mat[-2, :])

        self.Gamma_2_der, self.Gamma_3_der, self.mu2_der, self.mu3_der, self.ddG_predicted = gamma_2, gamma_3, mu2_mat, mu3_mat, ddG_predicted
    
    def plot_phiS(self):
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(10,3), layout="constrained")

        cp0=axes[0].contourf(self.phi2, 
                        self.phi3, 
                        self.phi1s,
                        levels=20)
        axes[0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0].set_xlabel(r'$\phi_2$')
        axes[0].set_ylabel(r'$\phi_3$')
        axes[0].set_title(r'$\phi_1^{s}$')
        fig.colorbar(cp0)

        cp1=axes[1].contourf(self.phi2, 
                        self.phi3, 
                        self.phi2s,
                        levels=20)
        axes[1].contour(cp1, colors='k', linestyles='solid', linewidths=0.5)
        axes[1].set_xlabel(r'$\phi_2$')
        axes[1].set_ylabel(r'$\phi_3$')
        axes[1].set_title(r'$\phi_2^{s}$')
        fig.colorbar(cp1)

        cp2=axes[2].contourf(self.phi2, 
                        self.phi3, 
                        self.phi3s,
                        levels=20)
        axes[2].contour(cp2, colors='k', linestyles='solid', linewidths=0.5)
        axes[2].set_xlabel(r'$\phi_2$')
        axes[2].set_ylabel(r'$\phi_3$')
        axes[2].set_title(r'$\phi_3^{s}$')
        fig.colorbar(cp2)
        plt.show()
        
    def plot_Ms(self):
        fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(12,4), layout="constrained")

        cp0=axes[0,0].contourf(self.phi2s, 
                        self.phi3s, 
                        1-self.Ms2_Ms-self.Ms3_Ms,
                        levels=20)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\phi_2^s$')
        axes[0,0].set_ylabel(r'$\phi_3^s$')
        axes[0,0].set_title(r'$Ms1/Ms$')
        fig.colorbar(cp0)
        axes[0,0].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[0,0].set_ylim(top=np.nanmax(self.phi3), bottom=0)

        cp0=axes[1,0].contourf(self.phi2, 
                        self.phi3, 
                        1-self.Ms2_Ms-self.Ms3_Ms,
                        levels=20)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\phi_2$')
        axes[1,0].set_ylabel(r'$\phi_3$')
        axes[1,0].set_title(r'$Ms1/Ms$')
        fig.colorbar(cp0)
        axes[1,0].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[1,0].set_ylim(top=np.nanmax(self.phi3), bottom=0)


        cp0=axes[0,1].contourf(self.phi2s, 
                        self.phi3s, 
                        self.Ms2_Ms,
                        levels=20)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\phi_2^s$')
        axes[0,1].set_ylabel(r'$\phi_3^s$')
        axes[0,1].set_title(r'$Ms2/Ms$')
        fig.colorbar(cp0)
        axes[0,1].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[0,1].set_ylim(top=np.nanmax(self.phi3), bottom=0)

        cp0=axes[1,1].contourf(self.phi2, 
                        self.phi3, 
                        self.Ms2_Ms,
                        levels=20)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\phi_2$')
        axes[1,1].set_ylabel(r'$\phi_3$')
        axes[1,1].set_title(r'$Ms2/Ms$')
        fig.colorbar(cp0)
        axes[1,1].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[1,1].set_ylim(top=np.nanmax(self.phi3), bottom=0)


        cp0=axes[0,2].contourf(self.phi2s, 
                        self.phi3s, 
                        self.Ms3_Ms,
                        levels=20)
        axes[0,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,2].set_xlabel(r'$\phi_2^s$')
        axes[0,2].set_ylabel(r'$\phi_3^s$')
        axes[0,2].set_title(r'$Ms3/Ms$')
        fig.colorbar(cp0)
        axes[0,2].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[0,2].set_ylim(top=np.nanmax(self.phi3), bottom=0)


        cp0=axes[1,2].contourf(self.phi2, 
                        self.phi3, 
                        self.Ms3_Ms,
                        levels=20)
        axes[1,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,2].set_xlabel(r'$\phi_2$')
        axes[1,2].set_ylabel(r'$\phi_3$')
        axes[1,2].set_title(r'$Ms3/Ms$')
        fig.colorbar(cp0)
        axes[1,2].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[1,2].set_ylim(top=np.nanmax(self.phi3), bottom=0)

        cp0=axes[0,3].contourf(self.phi2s, 
                        self.phi3s, 
                        self.Ms3_Ms+self.Ms2_Ms,
                        levels=20)
        axes[0,3].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,3].set_xlabel(r'$\phi_2^s$')
        axes[0,3].set_ylabel(r'$\phi_3^s$')
        axes[0,3].set_title(r'$Ms2/Ms+Ms3/Ms$')
        fig.colorbar(cp0)
        axes[0,3].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[0,3].set_ylim(top=np.nanmax(self.phi3), bottom=0)

        cp0=axes[1,3].contourf(self.phi2, 
                        self.phi3, 
                        self.Ms3_Ms+self.Ms2_Ms,
                        levels=20)
        axes[1,3].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,3].set_xlabel(r'$\phi_2$')
        axes[1,3].set_ylabel(r'$\phi_3$')
        axes[1,3].set_title(r'$Ms2/Ms+Ms3/Ms$')
        fig.colorbar(cp0)
        axes[1,3].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[1,3].set_ylim(top=np.nanmax(self.phi3), bottom=0)

        plt.show()
        
    def plot_mus2(self):
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3), layout="constrained")

        cp0=axes[0].contourf(self.phi2, 
                        self.phi3, 
                        self.mu1s2,
                        levels=5)
        axes[0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0].set_xlabel(r'$\phi_2$')
        axes[0].set_ylabel(r'$\phi_3$')
        axes[0].set_title(r'$\mu_1^{s2}$')
        fig.colorbar(cp0)
        axes[0].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[0].set_ylim(top=np.nanmax(self.phi3), bottom=0)

        cp0=axes[1].contourf(self.phi2, 
                        self.phi3, 
                        self.mu2s2,
                        levels=5)
        axes[1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1].set_xlabel(r'$\phi_2$')
        axes[1].set_ylabel(r'$\phi_3$')
        axes[1].set_title(r'$\mu_2^{s2}$')
        fig.colorbar(cp0)
        axes[1].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[1].set_ylim(top=np.nanmax(self.phi3), bottom=0)
        plt.show()

        
    def plot_mus3(self):
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,3), layout="constrained")

        cp0=axes[0].contourf(self.phi2, 
                        self.phi3, 
                        self.mu1s3,
                        levels=5)
        axes[0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0].set_xlabel(r'$\phi_2$')
        axes[0].set_ylabel(r'$\phi_3$')
        axes[0].set_title(r'$\mu_1^{s3}$')
        fig.colorbar(cp0)
        axes[0].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[0].set_ylim(top=np.nanmax(self.phi3), bottom=0)

        cp0=axes[1].contourf(self.phi2, 
                        self.phi3, 
                        self.mu2s3,
                        levels=5)
        axes[1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1].set_xlabel(r'$\phi_2$')
        axes[1].set_ylabel(r'$\phi_3$')
        axes[1].set_title(r'$\mu_2^{s3}$')
        fig.colorbar(cp0)
        axes[1].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[1].set_ylim(top=np.nanmax(self.phi3), bottom=0)

        cp0=axes[2].contourf(self.phi2, 
                        self.phi3, 
                        self.mu3s3,
                        levels=5)
        axes[2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[2].set_xlabel(r'$\phi_2$')
        axes[2].set_ylabel(r'$\phi_3$')
        axes[2].set_title(r'$\mu_3^{s3}$')
        fig.colorbar(cp0)
        axes[2].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[2].set_ylim(top=np.nanmax(self.phi3), bottom=0)
        plt.show()
        
    def plot_equil_cond(self):
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15,4), layout="constrained")

        cp0=axes[0].contourf(self.phi2, 
                        self.phi3, 
                        self.mu2s2-self.mu2 + 
                            self.nu2*
                            (self.mu1-self.mu1s2),
                        levels=500)
        axes[0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0].set_xlabel(r'$\phi_2$')
        axes[0].set_ylabel(r'$\phi_3$')
        axes[0].set_title(r'$(\mu_2^{s2}-\mu_2)+\nu_2(\mu_1-\mu_1^{s2})$')
        fig.colorbar(cp0)
        axes[0].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[0].set_ylim(top=np.nanmax(self.phi3), bottom=0)

        cp0=axes[1].contourf(self.phi2, 
                        self.phi3, 
                        self.mu2s3-self.mu2 + 
                            self.nu2*
                            (self.mu1-self.mu1s3),
                        levels=5)
        axes[1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1].set_xlabel(r'$\phi_2$')
        axes[1].set_ylabel(r'$\phi_3$')
        axes[1].set_title(r'$(\mu_2^{s3}-\mu_2)+\nu_3(\mu_1-\mu_1^{s3})$')
        fig.colorbar(cp0)
        axes[1].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[1].set_ylim(top=np.nanmax(self.phi3), bottom=0)

        cp0=axes[2].contourf(self.phi2, 
                        self.phi3, 
                        self.mu3s3-self.mu3 + 
                            self.nu3*
                            (self.mu1-self.mu1s3),
                        levels=5)
        axes[2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[2].set_xlabel(r'$\phi_2$')
        axes[2].set_ylabel(r'$\phi_3$')
        axes[2].set_title(r'$(\mu_3^{s3}-\mu_3)+\nu_3(\mu_1-\mu_1^{s3})$')
        fig.colorbar(cp0)
        axes[2].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[2].set_ylim(top=np.nanmax(self.phi3), bottom=0)
        plt.show()
        
    def plot_TdS_mix(self):
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12,6), layout="constrained")

        cp0=axes[0,0].contourf(self.phi2, 
                        self.phi3, 
                        self.TdS_mix_s2,
                        levels=10)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\phi_2$')
        axes[0,0].set_ylabel(r'$\phi_3$')
        axes[0,0].set_title(r'$T\Delta S_{mix}/(k \cdot Ms_2)$')
        fig.colorbar(cp0)

        cp0=axes[0,1].contourf(self.phi2, 
                        self.phi3, 
                        self.TdS_mix_s3,
                        levels=10)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\phi_2$')
        axes[0,1].set_ylabel(r'$\phi_3$')
        axes[0,1].set_title(r'$T\Delta S_{mix}/(k \cdot Ms_3)$')
        fig.colorbar(cp0)

        cp0=axes[0,2].contourf(self.phi2, 
                        self.phi3, 
                        self.TdS_mix_s_tot,
                        levels=10)
        axes[0,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,2].set_xlabel(r'$\phi_2$')
        axes[0,2].set_ylabel(r'$\phi_3$')
        axes[0,2].set_title(r'$T\Delta S_{mix}/(k \cdot SASA)$')
        fig.colorbar(cp0)

        cp0=axes[1,0].contourf(self.phi2s, 
                        self.phi3s, 
                        self.TdS_mix_s2,
                        levels=10)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\phi_2^s$')
        axes[1,0].set_ylabel(r'$\phi_3^s$')
        axes[1,0].set_title(r'$T\Delta S_{mix}/(k \cdot Ms_2)$')
        fig.colorbar(cp0)
        axes[1,0].set_xlim(right=np.nanmax(self.phi2s), left=0)
        axes[1,0].set_ylim(top=np.nanmax(self.phi3s), bottom=0)


        cp0=axes[1,1].contourf(self.phi2s, 
                        self.phi3s, 
                        self.TdS_mix_s3,
                        levels=10)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\phi_2^s$')
        axes[1,1].set_ylabel(r'$\phi_3^s$')
        axes[1,1].set_title(r'$T\Delta S_{mix}/(k \cdot Ms_3)$')
        fig.colorbar(cp0)
        axes[1,1].set_xlim(right=np.nanmax(self.phi2s), left=0)
        axes[1,1].set_ylim(top=np.nanmax(self.phi3s), bottom=0)

        cp0=axes[1,2].contourf(self.phi2s, 
                        self.phi3s, 
                        self.TdS_mix_s_tot,
                        levels=10)
        axes[1,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,2].set_xlabel(r'$\phi_2^s$')
        axes[1,2].set_ylabel(r'$\phi_3^s$')
        axes[1,2].set_title(r'$T\Delta S_{mix}/(k \cdot SASA)$')
        fig.colorbar(cp0)
        axes[1,2].set_xlim(right=np.nanmax(self.phi2s), left=0)
        axes[1,2].set_ylim(top=np.nanmax(self.phi3s), bottom=0)
        plt.show()
        
    def plot_dG_mix(self):
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12,6), layout="constrained")

        cp0=axes[0,0].contourf(self.phi2, 
                        self.phi3, 
                        self.dG_mix_s2,
                        levels=10)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\phi_2$')
        axes[0,0].set_ylabel(r'$\phi_3$')
        axes[0,0].set_title(r'$\Delta G_{mix}/(k \cdot Ms_2)$')
        fig.colorbar(cp0)
        axes[0,0].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[0,0].set_ylim(top=np.nanmax(self.phi3), bottom=0)

        cp0=axes[0,1].contourf(self.phi2, 
                        self.phi3, 
                        self.dG_mix_s3,
                        levels=10)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\phi_2$')
        axes[0,1].set_ylabel(r'$\phi_3$')
        axes[0,1].set_title(r'$\Delta G_{mix}/(k \cdot Ms_3)$')
        axes[0,1].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[0,1].set_ylim(top=np.nanmax(self.phi3), bottom=0)
        fig.colorbar(cp0)

        cp0=axes[0,2].contourf(self.phi2, 
                        self.phi3, 
                        self.dG_mix_s_tot,
                        levels=10)
        axes[0,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,2].set_xlabel(r'$\phi_2$')
        axes[0,2].set_ylabel(r'$\phi_3$')
        axes[0,2].set_title(r'$\Delta G_{mix}/(k \cdot SASA)$')
        fig.colorbar(cp0)
        axes[0,2].set_xlim(right=np.nanmax(self.phi2), left=0)
        axes[0,2].set_ylim(top=np.nanmax(self.phi3), bottom=0)


        cp0=axes[1,0].contourf(self.phi2s, 
                        self.phi3s, 
                        self.dG_mix_s2,
                        levels=10)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\phi_2^s$')
        axes[1,0].set_ylabel(r'$\phi_3^s$')
        axes[1,0].set_title(r'$\Delta G_{mix}/(k \cdot Ms_2)$')
        fig.colorbar(cp0)
        axes[1,0].set_xlim(right=np.nanmax(self.phi2s), left=0)
        axes[1,0].set_ylim(top=np.nanmax(self.phi3s), bottom=0)

        cp0=axes[1,1].contourf(self.phi2s, 
                        self.phi3s, 
                        self.dG_mix_s3,
                        levels=10)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\phi_2^s$')
        axes[1,1].set_ylabel(r'$\phi_3^s$')
        axes[1,1].set_title(r'$\Delta G_{mix}/(k \cdot Ms_3)$')
        axes[1,1].set_xlim(right=np.nanmax(self.phi2s), left=0)
        axes[1,1].set_ylim(top=np.nanmax(self.phi3s), bottom=0)
        fig.colorbar(cp0)

        cp0=axes[1,2].contourf(self.phi2s, 
                        self.phi3s, 
                        self.dG_mix_s_tot,
                        levels=10)
        axes[1,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,2].set_xlabel(r'$\phi_2^s$')
        axes[1,2].set_ylabel(r'$\phi_3^s$')
        axes[1,2].set_title(r'$\Delta G_{mix}/(k \cdot SASA)$')
        fig.colorbar(cp0)
        axes[1,2].set_xlim(right=np.nanmax(self.phi2s), left=0)
        axes[1,2].set_ylim(top=np.nanmax(self.phi3s), bottom=0)
        plt.show()

    def plot_ddG(self):
        fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(8,6), layout="constrained")

        cp0=axes[0,0].contourf(self.phi2, 
                        self.phi3, 
                        self.ddG,
                        levels=10)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\phi_2$')
        axes[0,0].set_ylabel(r'$\phi_3$')
        axes[0,0].set_title(r'$\Delta\Delta G^0/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[0,1].contourf(self.phi2, 
                        self.phi3, 
                        self.ddG_nu,
                        levels=10)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\phi_2$')
        axes[0,1].set_ylabel(r'$\phi_3$')
        axes[0,1].set_title(r'$\Delta\Delta G^0_\nu/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1,0].contourf(self.phi2, 
                        self.phi3, 
                        self.ddG_chi,
                        levels=10)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\phi_2$')
        axes[1,0].set_ylabel(r'$\phi_3$')
        axes[1,0].set_title(r'$\Delta\Delta G^0_\chi/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1,1].contourf(self.phi2, 
                        self.phi3, 
                        self.ddG_chi12_chi13,
                        levels=10)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\phi_2$')
        axes[1,1].set_ylabel(r'$\phi_3$')
        axes[1,1].set_title(r'$\Delta\Delta G^0_{\chi_{12}+\chi_{13}}/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1,2].contourf(self.phi2, 
                        self.phi3, 
                        self.ddG_chi23,
                        levels=10)
        axes[1,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,2].set_xlabel(r'$\phi_2$')
        axes[1,2].set_ylabel(r'$\phi_3$')
        axes[1,2].set_title(r'$\Delta\Delta G^0_{\chi_{23}}/(k T)$')
        fig.colorbar(cp0)
        
        cp0=axes[2,0].contourf(self.phi2, 
                        self.phi3, 
                        self.ddG_eps,
                        levels=10)
        axes[2,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[2,0].set_xlabel(r'$\phi_2$')
        axes[2,0].set_ylabel(r'$\phi_3$')
        axes[2,0].set_title(r'$\Delta\Delta G^0_\varepsilon/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[2,1].contourf(self.phi2, 
                        self.phi3, 
                        self.ddG_eps12_eps13,
                        levels=10)
        axes[2,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[2,1].set_xlabel(r'$\phi_2$')
        axes[2,1].set_ylabel(r'$\phi_3$')
        axes[2,1].set_title(r'$\Delta\Delta G^0_{\varepsilon_{12}+\varepsilon_{13}}/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[2,2].contourf(self.phi2, 
                        self.phi3, 
                        self.ddG_eps23,
                        levels=10)
        axes[2,2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[2,2].set_xlabel(r'$\phi_2$')
        axes[2,2].set_ylabel(r'$\phi_3$')
        axes[2,2].set_title(r'$\Delta\Delta G^0_{\varepsilon_{23}}/(k T)$')
        fig.colorbar(cp0)
        plt.show()

    def plot_ddG_mu(self):
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,6), layout="constrained")

        cp0=axes[0,0].contourf(self.mu2, 
                        self.mu3, 
                        self.ddG,
                        levels=10)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\mu_2$')
        axes[0,0].set_ylabel(r'$\mu_3$')
        axes[0,0].set_title(r'$\Delta\Delta G^0/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[0,1].contourf(self.mu2, 
                        self.mu3, 
                        self.ddG_nu,
                        levels=10)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\mu_2$')
        axes[0,1].set_ylabel(r'$\mu_3$')
        axes[0,1].set_title(r'$\Delta\Delta G^0_\nu/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1,0].contourf(self.mu2, 
                        self.mu3, 
                        self.ddG_chi,
                        levels=10)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\mu_2$')
        axes[1,0].set_ylabel(r'$\mu_3$')
        axes[1,0].set_title(r'$\Delta\Delta G^0_\chi/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1,1].contourf(self.mu2, 
                        self.mu3, 
                        self.ddG_eps,
                        levels=10)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\mu_2$')
        axes[1,1].set_ylabel(r'$\mu_3$')
        axes[1,1].set_title(r'$\Delta\Delta G^0_\varepsilon/(k T)$')
        fig.colorbar(cp0)
        plt.show()
        
    def plot_TddS(self):
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,6), layout="constrained")

        cp0=axes[0,0].contourf(self.phi2, 
                        self.phi3, 
                        self.TddS,
                        levels=10)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\phi_2$')
        axes[0,0].set_ylabel(r'$\phi_3$')
        axes[0,0].set_title(r'$T\Delta\Delta S^0/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[0,1].contourf(self.phi2, 
                        self.phi3, 
                        self.TddS_nu,
                        levels=10)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\phi_2$')
        axes[0,1].set_ylabel(r'$\phi_3$')
        axes[0,1].set_title(r'$T\Delta\Delta S^0_\nu/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1,0].contourf(self.phi2, 
                        self.phi3, 
                        self.TddS_chi,
                        levels=10)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\phi_2$')
        axes[1,0].set_ylabel(r'$\phi_3$')
        axes[1,0].set_title(r'$T\Delta\Delta S^0_\chi/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1,1].contourf(self.phi2, 
                        self.phi3, 
                        self.TddS_eps,
                        levels=10)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\phi_2$')
        axes[1,1].set_ylabel(r'$\phi_3$')
        axes[1,1].set_title(r'$T\Delta\Delta S^0_\varepsilon/(k T)$')
        fig.colorbar(cp0)
        plt.show()
        
    def plot_ddH(self):
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(10,3), layout="constrained")

        cp0=axes[0].contourf(self.phi2, 
                        self.phi3, 
                        self.ddH,
                        levels=10)
        axes[0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0].set_xlabel(r'$\phi_2$')
        axes[0].set_ylabel(r'$\phi_3$')
        axes[0].set_title(r'$\Delta\Delta H^0/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[1].contourf(self.phi2, 
                        self.phi3, 
                        self.ddH_chi,
                        levels=10)
        axes[1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1].set_xlabel(r'$\phi_2$')
        axes[1].set_ylabel(r'$\phi_3$')
        axes[1].set_title(r'$\Delta\Delta H^0_\chi/(k T)$')
        fig.colorbar(cp0)

        cp0=axes[2].contourf(self.phi2, 
                        self.phi3, 
                        self.ddG_eps,
                        levels=10)
        axes[2].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[2].set_xlabel(r'$\phi_2$')
        axes[2].set_ylabel(r'$\phi_3$')
        axes[2].set_title(r'$\Delta\Delta H^0_\varepsilon/(k T)$')
        fig.colorbar(cp0)
        plt.show()

    def plot_Gamma(self):
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,6), layout="constrained")

        cp0=axes[0,0].contourf(self.phi2, 
                        self.phi3, 
                        self.Gamma_2,
                        levels=10)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\phi_2$')
        axes[0,0].set_ylabel(r'$\phi_3$')
        axes[0,0].set_title(r'$\Delta \Gamma_2$')
        fig.colorbar(cp0)

        cp0=axes[0,1].contourf(self.phi2, 
                        self.phi3, 
                        self.Gamma_3,
                        levels=10)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\phi_2$')
        axes[0,1].set_ylabel(r'$\phi_3$')
        axes[0,1].set_title(r'$\Delta \Gamma_3$')
        fig.colorbar(cp0)

        cp0=axes[1,0].contourf(self.phi2, 
                        self.phi3, 
                        self.Gamma_1_2,
                        levels=10)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\phi_2$')
        axes[1,0].set_ylabel(r'$\phi_3$')
        axes[1,0].set_title(r'$\Delta \Gamma_{1,2}$')
        fig.colorbar(cp0)

        cp0=axes[1,1].contourf(self.phi2, 
                        self.phi3, 
                        self.Gamma_1_3,
                        levels=10)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\phi_2$')
        axes[1,1].set_ylabel(r'$\phi_3$')
        axes[1,1].set_title(r'$\Delta \Gamma_{1,3}$')
        fig.colorbar(cp0)
        plt.show()

    def plot_Gamma_mu(self):
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,6), layout="constrained")

        cp0=axes[0,0].contourf(self.mu2, 
                        self.mu3, 
                        self.Gamma_2,
                        levels=10)
        axes[0,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,0].set_xlabel(r'$\mu_2$')
        axes[0,0].set_ylabel(r'$\mu_3$')
        axes[0,0].set_title(r'$\Delta \Gamma_2$')
        fig.colorbar(cp0)

        cp0=axes[0,1].contourf(self.mu2, 
                        self.mu3, 
                        self.Gamma_3,
                        levels=10)
        axes[0,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0,1].set_xlabel(r'$\mu_2$')
        axes[0,1].set_ylabel(r'$\mu_3$')
        axes[0,1].set_title(r'$\Delta \Gamma_3$')
        fig.colorbar(cp0)

        cp0=axes[1,0].contourf(self.mu2, 
                        self.mu3, 
                        self.Gamma_1_2,
                        levels=10)
        axes[1,0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,0].set_xlabel(r'$\mu_2$')
        axes[1,0].set_ylabel(r'$\mu_3$')
        axes[1,0].set_title(r'$\Delta \Gamma_{1,2}$')
        fig.colorbar(cp0)

        cp0=axes[1,1].contourf(self.mu2, 
                        self.mu3, 
                        self.Gamma_1_3,
                        levels=10)
        axes[1,1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1,1].set_xlabel(r'$\mu_2$')
        axes[1,1].set_ylabel(r'$\mu_3$')
        axes[1,1].set_title(r'$\Delta \Gamma_{1,3}$')
        fig.colorbar(cp0)
        plt.show()

        
    def plot_Gamma_mu_der(self):
        L=np.linspace(self.Gamma_2_der.min(),self.Gamma_2_der.max(),20)
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10,4), layout="constrained")

        cp0=axes[0].contourf(self.mu2_der, 
                        self.mu3_der, 
                        self.Gamma_2_der,
                        levels=L)
        axes[0].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[0].set_xlabel(r'$\mu_2$')
        axes[0].set_ylabel(r'$\mu_3$')
        axes[0].set_title(r'$\Delta \Gamma_2$')
        fig.colorbar(cp0)

        cp0=axes[1].contourf(self.mu2_der, 
                        self.mu3_der, 
                        self.Gamma_3_der,
                        levels=L)
        axes[1].contour(cp0, colors='k', linestyles='solid', linewidths=0.5)
        axes[1].set_xlabel(r'$\mu_2$')
        axes[1].set_ylabel(r'$\mu_3$')
        axes[1].set_title(r'$\Delta \Gamma_3$')
        fig.colorbar(cp0)
        plt.show()
              
    def to_pandas(self):
        self.results = pd.DataFrame({'phi1':self.phi1.flatten(),
                            'phi2':self.phi2.flatten(),
                            'phi3':self.phi3.flatten(),
                            'phi1s':self.phi1s.flatten(), 
                            'phi2s':self.phi2s.flatten(), 
                            'phi3s':self.phi3s.flatten(), 
                            'phi1s2':self.phi1s2.flatten(), 
                            'phi2s2':self.phi2s2.flatten(), 
                            'phi1s3':self.phi1s3.flatten(), 
                            'phi2s3':self.phi2s3.flatten(), 
                            'phi3s3':self.phi3s3.flatten(), 
                            'molar2':self.molar2.flatten(),
                            'molar3':self.molar3.flatten(),
                            'molal2':self.molal2.flatten(),
                            'molal3':self.molal3.flatten(),
                            'osm':self.osm.flatten(),
                            'ddG_nu':self.ddG_nu.flatten(), 
                            'ddG_chi':self.ddG_chi.flatten(), 
                            'ddG_chi12_chi13':self.ddG_chi12_chi13.flatten(), 
                            'ddG_chi23':self.ddG_chi23.flatten(), 
                            'ddG_eps':self.ddG_eps.flatten(), 
                            'ddG_eps12_eps13':self.ddG_eps12_eps13.flatten(), 
                            'ddG_eps23':self.ddG_eps23.flatten(), 
                            'ddG':self.ddG.flatten(),
                            'ddH_chi':self.ddH_chi.flatten(), 
                            'ddH_chi12_chi13':self.ddH_chi12_chi13.flatten(), 
                            'ddH_chi23':self.ddH_chi23.flatten(), 
                            'ddH_eps':self.ddH_eps.flatten(), 
                            'ddH_eps12_eps13':self.ddH_eps12_eps13.flatten(), 
                            'ddH_eps23':self.ddH_eps23.flatten(), 
                            'ddH':self.ddH.flatten(),
                            'TddS_nu':self.TddS_nu.flatten(), 
                            'TddS_chi':self.TddS_chi.flatten(), 
                            'TddS_chi12_chi13':self.TddS_chi12_chi13.flatten(), 
                            'TddS_chi23':self.TddS_chi23.flatten(), 
                            'TddS_eps':self.TddS_eps.flatten(), 
                            'TddS_eps12_eps13':self.TddS_eps12_eps13.flatten(), 
                            'TddS_eps23':self.TddS_eps23.flatten(), 
                            'TddS':self.TddS.flatten(),
                            'ddG_nu_kJ':self.ddG_nu_kJ.flatten(), 
                            'ddG_chi_kJ':self.ddG_chi_kJ.flatten(), 
                            'ddG_chi12_chi13_kJ':self.ddG_chi12_chi13_kJ.flatten(), 
                            'ddG_chi23_kJ':self.ddG_chi23_kJ.flatten(), 
                            'ddG_eps_kJ':self.ddG_eps_kJ.flatten(), 
                            'ddG_eps12_eps13_kJ':self.ddG_eps12_eps13_kJ.flatten(), 
                            'ddG_eps23_kJ':self.ddG_eps23_kJ.flatten(), 
                            'ddG_kJ':self.ddG_kJ.flatten(),
                            'ddH_chi_kJ':self.ddH_chi_kJ.flatten(), 
                            'ddH_chi12_chi13_kJ':self.ddH_chi12_chi13_kJ.flatten(), 
                            'ddH_chi23_kJ':self.ddH_chi23_kJ.flatten(), 
                            'ddH_eps_kJ':self.ddH_eps_kJ.flatten(), 
                            'ddH_eps12_eps13_kJ':self.ddH_eps12_eps13_kJ.flatten(), 
                            'ddH_eps23_kJ':self.ddH_eps23_kJ.flatten(), 
                            'ddH_kJ':self.ddH_kJ.flatten(),
                            'TddS_nu_kJ':self.TddS_nu_kJ.flatten(), 
                            'TddS_chi_kJ':self.TddS_chi_kJ.flatten(), 
                            'TddS_chi12_chi13_kJ':self.TddS_chi12_chi13_kJ.flatten(), 
                            'TddS_chi23_kJ':self.TddS_chi23_kJ.flatten(), 
                            'TddS_eps_kJ':self.TddS_eps_kJ.flatten(), 
                            'TddS_eps12_eps13_kJ':self.TddS_eps12_eps13_kJ.flatten(), 
                            'TddS_eps23_kJ':self.TddS_eps23_kJ.flatten(), 
                            'TddS_kJ':self.TddS_kJ.flatten(),
                            'mu1':self.mu1.flatten(),
                            'mu2':self.mu2.flatten(),
                            'mu3':self.mu3.flatten(),
                            'Gamma2':self.Gamma_2.flatten(),
                            'Gamma3':self.Gamma_3.flatten(),
                            'Gamma12':self.Gamma_1_2.flatten(),
                            'Gamma13':self.Gamma_1_3.flatten()})
    
