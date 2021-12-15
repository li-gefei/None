'''
In this page, we will calculate the dispersion coefficients up to 4-th order
'''

import sympy as sy
import numpy as np
import math
from Pulse_Parameter import PumpDict

c0 = float(299792458)

class Dispersion_Coefficients_Calculation:

    def __init__(self, Wavelength, Crystal_Length, Theta):
        self.Wavelength = Wavelength*1e-3  # Unit um
        self.Crystal_Length = Crystal_Length*1e-3
        self.Theta = Theta/180*math.pi
        self.C1_O = 2.7359
        self.C2_O = 0.01878
        self.C3_O = 0.01822
        self.C4_O = 0.01354

        self.C1_E = 2.3753
        self.C2_E = 0.01224
        self.C3_E = 0.01667
        self.C4_E = 0.01516

    def Sellmerier_Equation(self, C1, C2, C3, C4):
        Wavelength = self.Wavelength  # Unit um
        Refrective_Index = np.sqrt(C1 + C2/(Wavelength*Wavelength - C3) -C4*Wavelength*Wavelength)
        return Refrective_Index

    '''The differential coefficients are calculated based on Sellmerier euqation. We obtain these expresion from sympy package of python''' 
      
    def Differential_Coefficients(self, C1, C2, C3, C4):
        Wavelength = self.Wavelength  # Unit um
        D_1 = (-C2*Wavelength/(-C3 + Wavelength**2)**2 - C4*Wavelength)/np.sqrt(C1 + C2/(-C3 + Wavelength**2) - C4*Wavelength**2)
        D_2 = -(4*C2*Wavelength**2/(C3 - Wavelength**2)**3 + C2/(C3 - Wavelength**2)**2 + C4 + Wavelength**2*(C2/(C3 - Wavelength**2)**2 + C4)**2/(C1 - C2/(C3 - Wavelength**2) - C4*Wavelength**2))/np.sqrt(C1 - C2/(C3 - Wavelength**2) - C4*Wavelength**2)
        D_3 = -3*Wavelength*(4*C2*(2*Wavelength**2/(C3 - Wavelength**2) + 1)/(C3 - Wavelength**2)**3 + Wavelength**2*(C2/(C3 - Wavelength**2)**2 + C4)**3/(C1 - C2/(C3 - Wavelength**2) - C4*Wavelength**2)**2 + (C2/(C3 - Wavelength**2)**2 + C4)*(4*C2*Wavelength**2/(C3 - Wavelength**2)**3 + C2/(C3 - Wavelength**2)**2 + C4)/(C1 - C2/(C3 - Wavelength**2) - C4*Wavelength**2))/np.sqrt(C1 - C2/(C3 - Wavelength**2) - C4*Wavelength**2)
        D_4 = -3*(16*C2*Wavelength**2*(C2/(C3 - Wavelength**2)**2 + C4)*(2*Wavelength**2/(C3 - Wavelength**2) + 1)/((C3 - Wavelength**2)**3*(C1 - C2/(C3 - Wavelength**2) - C4*Wavelength**2)) + 4*C2*(16*Wavelength**4/(C3 - Wavelength**2)**2 + 12*Wavelength**2/(C3 - Wavelength**2) + 1)/(C3 - Wavelength**2)**3 + 5*Wavelength**4*(C2/(C3 - Wavelength**2)**2 + C4)**4/(C1 - C2/(C3 - Wavelength**2) - C4*Wavelength**2)**3 + 6*Wavelength**2*(C2/(C3 - Wavelength**2)**2 + C4)**2*(4*C2*Wavelength**2/(C3 - Wavelength**2)**3 + C2/(C3 - Wavelength**2)**2 + C4)/(C1 - C2/(C3 - Wavelength**2) - C4*Wavelength**2)**2 + (4*C2*Wavelength**2/(C3 - Wavelength**2)**3 + C2/(C3 - Wavelength**2)**2 + C4)**2/(C1 - C2/(C3 - Wavelength**2) - C4*Wavelength**2))/np.sqrt(C1 - C2/(C3 - Wavelength**2) - C4*Wavelength**2)
        return D_1, D_2, D_3, D_4
        
    '''The dispersion coefficients are calculated according to the phd thesis of Jianwang JIang. 'Generation and amplification of ultra-broadband femtosecond pulses and carrier-envolope phase offset stabilization'    ''' 
    def Dispersion_Coefficients_O(self):
        Wavelength = self.Wavelength  # Unit  um
        Length = self.Crystal_Length  # Unit m
        N = self.Sellmerier_Equation(self.C1_O, self.C2_O, self.C3_O, self.C4_O)
        D1, D2, D3, D4 = self.Differential_Coefficients(self.C1_O, self.C2_O, self.C3_O, self.C4_O)
        Dispersion_1st = 1/c0*(N - Wavelength*D1)
        Dispersion_2nd = (Wavelength**3)/2/math.pi/c0/c0*D2*1e-6
        Dispersion_3rd = -(Wavelength**4) /4/(math.pi**2)/c0/c0/c0*(3*D2 + Wavelength*D3)*1e-12
        Dispersion_4th = (Wavelength**5) /8/(math.pi**3)/c0/c0/c0/c0*(12*D2 + 8*Wavelength*D3 + Wavelength**2 * D4)*1e-18
     
        #return  Dispersion_1st, Dispersion_2nd, Dispersion_3rd, Dispersion_4th
        return 0, 0, 0, Dispersion_4th

    def Dispersion_Coefficients_E(self):
        Wavelength = self.Wavelength  # Unit  um
        Length = self.Crystal_Length  # Unit m
        N = self.Sellmerier_Equation(self.C1_E, self.C2_E, self.C3_E, self.C4_E)
        D1, D2, D3, D4 = self.Differential_Coefficients(self.C1_E, self.C2_E, self.C3_E, self.C4_E)
        Dispersion_1st = 1/c0*(N - Wavelength*D1)
        Dispersion_2nd = (Wavelength**3)/2/math.pi/c0/c0*D2*1e-6
        Dispersion_3rd = -(Wavelength**4) /4/(math.pi**2)/c0/c0/c0*(3*D2 + Wavelength*D3)*1e-12
        Dispersion_4th = (Wavelength**5) /8/(math.pi**3)/c0/c0/c0/c0*(12*D2 + 8*Wavelength*D3 + Wavelength**2 * D4)*1e-18
        return  Dispersion_1st, Dispersion_2nd, Dispersion_3rd, Dispersion_4th
        #return 0, 0, 0, Dispersion_4th

if __name__ == "__main__":
    c0 = float(299792458) # Unit SI
    Epsilon = 8.854187817e-12 # Unit SI
    d_eff = 2.02*1e-12  # Unit SI
    n = int(math.pow(2, 16))
    


    