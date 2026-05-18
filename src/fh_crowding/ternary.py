import numpy as np
import pandas as pd
from typing import Optional, Sequence, Tuple, Union
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from .protein import Protein
from .cosolute import CosoluteMixture

class TernaryCrowdingModel(CosoluteMixture):
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


    def solve_equil(self, print_msg: bool = True, callback=None):
        '''Solve the equilibrium condition over the phi2/phi3 grid.

        Records per-point convergence diagnostics in:
          - self.converged: bool array (same shape as phi2)
          - self.flags: int array of fsolve ier
          - self.messages: object array of fsolve messages
        Optionally prints failures when print_msg=True.

        Args:
            callback: Optional callable ``(fraction: float) -> None`` invoked
                once per outer row to report progress in [0, 1].  To avoid
                excessive overhead, it is called at most every
                ``max(1, n_rows // 100)`` rows (~1 % increments).  Pass
                ``None`` (default) to disable — notebooks are unaffected.
        '''
        shape = self.phi2.shape
        n_rows = shape[0]
        report_every = max(1, n_rows // 100)  # report at most ~100 times
        self.converged = np.zeros(shape, dtype=bool)
        self.flags = np.zeros(shape, dtype=int)
        self.messages = np.empty(shape, dtype=object)
        self.messages[:] = ''

        I, J = [], []
        for i in range(n_rows):
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

            # report progress at most ~100 times (every ~1 %)
            if callback is not None and ((i + 1) % report_every == 0 or i == n_rows - 1):
                callback((i + 1) / n_rows)
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
        exp_conc2 = np.atleast_2d(exp_conc2)
        exp_conc3 = np.atleast_2d(exp_conc3)
        if concentration_type == 'phi':
            pass
        elif concentration_type=='molal':
            denom = 1 + exp_conc2 * (18 * self.nu2) * 1E-3 + exp_conc3 * (18 * self.nu3) * 1E-3
            exp_conc2 = exp_conc2 * (18 * self.nu2) * 1E-3 / denom
            exp_conc3 = exp_conc3 * (18 * self.nu3) * 1E-3 / denom
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
        exp_conc2 = np.atleast_2d(exp_conc2)
        exp_conc3 = np.atleast_2d(exp_conc3)
        if concentration_type == 'phi':
            pass
        elif concentration_type=='molal':
            denom = 1 + exp_conc2 * (18 * self.nu2) * 1E-3 + exp_conc3 * (18 * self.nu3) * 1E-3
            exp_conc2 = exp_conc2 * (18 * self.nu2) * 1E-3 / denom
            exp_conc3 = exp_conc3 * (18 * self.nu3) * 1E-3 / denom
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
    
