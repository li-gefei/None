import sympy as sy
import numpy as np
import math
from Pulse_Parameter import PumpDict

Wavelength_Pump = PumpDict['Central_Wavelength(nm)']
c0 = float(299792458)

'''Because we apply type I phase matching in this package, the refractive index of pump beam will be a function of phase matching angle, i.e. 
N_pump = N_pump (\theta). So the dispersion coefficients will also be a function of theta'''

class Dispersion_Coefficients_Calculation_Pump:

    def __init__(self, Wavelength, Crystal_Length, Theta):
        self.Wavelength = Wavelength*1e-3
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
        Refrective_Index = np.sqrt(C1 + C2/(Wavelength*Wavelength - C3) - C4*Wavelength*Wavelength)
        return Refrective_Index
    
    def Refrective_Index_Pump(self):
        Theta = self.Theta
        N_O = self.Sellmerier_Equation(self.C1_O, self.C2_O, self.C3_O, self.C4_O)
        N_E = self.Sellmerier_Equation(self.C1_E, self.C2_E, self.C3_E, self.C4_E)
        N_Pump = N_O*N_E/np.sqrt(N_E**2 * (np.cos(Theta))**2   +   N_O**2 * (np.sin(Theta))**2)
        return N_Pump

    def Differential_Coefficients_Pump(self):
        Wavelength = Wavelength_Pump*1e-3  # Unit  um
        Theta = self.Theta
        C1O = self.C1_O 
        C2O = self.C2_O 
        C3O = self.C3_O 
        C4O = self.C4_O 

        C1E = self.C1_E 
        C2E = self.C2_E 
        C3E = self.C3_E 
        C4E = self.C4_E 
      
        D1_Pump = (-(-2*C2E*Wavelength/(-C3E + Wavelength**2)**2 - 2*C4E*Wavelength)*np.sin(Theta)**2/2 - (-2*C2O*Wavelength/(-C3O + Wavelength**2)**2 - 2*C4O*Wavelength)*np.cos(Theta)**2/2)*np.sqrt(C1E + C2E/(-C3E + Wavelength**2) - C4E*Wavelength**2)*np.sqrt(C1O + C2O/(-C3O + Wavelength**2) - C4O*Wavelength**2)/((C1E + C2E/(-C3E + Wavelength**2) - C4E*Wavelength**2)*np.sin(Theta)**2 + (C1O + C2O/(-C3O + Wavelength**2) - C4O*Wavelength**2)*np.cos(Theta)**2)**(3/2) + (-C2E*Wavelength/(-C3E + Wavelength**2)**2 - C4E*Wavelength)*np.sqrt(C1O + C2O/(-C3O + Wavelength**2) - C4O*Wavelength**2)/(np.sqrt((C1E + C2E/(-C3E + Wavelength**2) - C4E*Wavelength**2)*np.sin(Theta)**2 + (C1O + C2O/(-C3O + Wavelength**2) - C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1E + C2E/(-C3E + Wavelength**2) - C4E*Wavelength**2)) + (-C2O*Wavelength/(-C3O + Wavelength**2)**2 - C4O*Wavelength)*np.sqrt(C1E + C2E/(-C3E + Wavelength**2) - C4E*Wavelength**2)/(np.sqrt((C1E + C2E/(-C3E + Wavelength**2) - C4E*Wavelength**2)*np.sin(Theta)**2 + (C1O + C2O/(-C3O + Wavelength**2) - C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1O + C2O/(-C3O + Wavelength**2) - C4O*Wavelength**2))
        D2_Pump = (2*Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)*(C2O/(C3O - Wavelength**2)**2 + C4O)/(np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) - 2*Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)) - 2*Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) - np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*(4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O - Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)**2/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2))/np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2) - np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)*(4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E - Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)**2/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2))/np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2) + np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)*(-3*Wavelength**2*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)**2/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2) + (4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)/(-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2))/np.sqrt(-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)
        D3_Pump = 3*Wavelength*(2*Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)*(C2O/(C3O - Wavelength**2)**2 + C4O)*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) + (C2E/(C3E - Wavelength**2)**2 + C4E)*(4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O - Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)**2/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2))/(np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) - (C2E/(C3E - Wavelength**2)**2 + C4E)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)*(-3*Wavelength**2*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)**2/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2) + (4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)) + (C2O/(C3O - Wavelength**2)**2 + C4O)*(4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E - Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)**2/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2))/(np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) - (C2O/(C3O - Wavelength**2)**2 + C4O)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*(-3*Wavelength**2*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)**2/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2) + (4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) - ((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*(4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O - Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)**2/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2))/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) - ((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)*(4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E - Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)**2/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2))/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)) - np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*(4*C2O*(2*Wavelength**2/(C3O - Wavelength**2) + 1)/(C3O - Wavelength**2)**3 + Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)**3/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)**2 - (C2O/(C3O - Wavelength**2)**2 + C4O)*(4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2))/np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2) - np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)*(4*C2E*(2*Wavelength**2/(C3E - Wavelength**2) + 1)/(C3E - Wavelength**2)**3 + Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)**3/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)**2 - (C2E/(C3E - Wavelength**2)**2 + C4E)*(4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2))/np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2) + np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)*(4*C2E*(2*Wavelength**2/(C3E - Wavelength**2) + 1)*np.sin(Theta)**2/(C3E - Wavelength**2)**3 + 4*C2O*(2*Wavelength**2/(C3O - Wavelength**2) + 1)*np.cos(Theta)**2/(C3O - Wavelength**2)**3 + 5*Wavelength**2*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)**3/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)**2 - 3*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)*((4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2))/(-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2))/np.sqrt(-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)
        D4_Pump = 3*(4*Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)*(C2O/(C3O - Wavelength**2)**2 + C4O)*(-3*Wavelength**2*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)**2/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2) + (4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) + 4*Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)*(4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O - Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)**2/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2))/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) + 4*Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)*(4*C2O*(2*Wavelength**2/(C3O - Wavelength**2) + 1)/(C3O - Wavelength**2)**3 + Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)**3/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)**2 - (C2O/(C3O - Wavelength**2)**2 + C4O)*(4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2))/(np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) - 4*Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)*(4*C2E*(2*Wavelength**2/(C3E - Wavelength**2) + 1)*np.sin(Theta)**2/(C3E - Wavelength**2)**3 + 4*C2O*(2*Wavelength**2/(C3O - Wavelength**2) + 1)*np.cos(Theta)**2/(C3O - Wavelength**2)**3 + 5*Wavelength**2*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)**3/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)**2 - 3*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)*((4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2))/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)) + 4*Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)*(4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E - Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)**2/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2))/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) + 4*Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)*(4*C2E*(2*Wavelength**2/(C3E - Wavelength**2) + 1)/(C3E - Wavelength**2)**3 + Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)**3/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)**2 - (C2E/(C3E - Wavelength**2)**2 + C4E)*(4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2))/(np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) - 4*Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*(4*C2E*(2*Wavelength**2/(C3E - Wavelength**2) + 1)*np.sin(Theta)**2/(C3E - Wavelength**2)**3 + 4*C2O*(2*Wavelength**2/(C3O - Wavelength**2) + 1)*np.cos(Theta)**2/(C3O - Wavelength**2)**3 + 5*Wavelength**2*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)**3/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)**2 - 3*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)*((4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2))/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) - 4*Wavelength**2*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*(4*C2O*(2*Wavelength**2/(C3O - Wavelength**2) + 1)/(C3O - Wavelength**2)**3 + Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)**3/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)**2 - (C2O/(C3O - Wavelength**2)**2 + C4O)*(4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2))/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*
        np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) - 4*Wavelength**2*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)*(4*C2E*(2*Wavelength**2/(C3E - Wavelength**2) + 1)/(C3E - Wavelength**2)**3 + Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)**3/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)**2 - (C2E/(C3E - Wavelength**2)**2 + C4E)*(4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2))/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)) + np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*(16*C2O*Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)*(2*Wavelength**2/(C3O - Wavelength**2) + 1)/((C3O - Wavelength**2)**3*(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)) - 4*C2O*(16*Wavelength**4/(C3O - Wavelength**2)**2 + 12*Wavelength**2/(C3O - Wavelength**2) + 1)/(C3O - Wavelength**2)**3 + 5*Wavelength**4*(C2O/(C3O - Wavelength**2)**2 + C4O)**4/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)**3 - 6*Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)**2*(4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)**2 + (4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)**2/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2))/np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2) + np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)*(16*C2E*Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)*(2*Wavelength**2/(C3E - Wavelength**2) + 1)/((C3E - Wavelength**2)**3*(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)) - 4*C2E*(16*Wavelength**4/(C3E - Wavelength**2)**2 + 12*Wavelength**2/(C3E - Wavelength**2) + 1)/(C3E - Wavelength**2)**3 + 5*Wavelength**4*(C2E/(C3E - Wavelength**2)**2 + C4E)**4/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)**3 - 6*Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)**2*(4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)**2 + (4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)**2/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2))/np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2) + 2*(4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E - Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)**2/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2))*(4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O - Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)**2/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2))/(np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) + np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)*(4*C2E*(16*Wavelength**4/(C3E - Wavelength**2)**2 + 12*Wavelength**2/(C3E - Wavelength**2) + 1)*np.sin(Theta)**2/(C3E - Wavelength**2)**3 + 4*C2O*(16*Wavelength**4/(C3O - Wavelength**2)**2 + 12*Wavelength**2/(C3O - Wavelength**2) + 1)*np.cos(Theta)**2/(C3O - Wavelength**2)**3 - 35*Wavelength**4*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)**4/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)**3 + 30*Wavelength**2*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)**2*((4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)**2 - 48*Wavelength**2*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)*(C2E*(2*Wavelength**2/(C3E - Wavelength**2) + 1)*np.sin(Theta)**2/(C3E - Wavelength**2)**3 + C2O*(2*Wavelength**2/(C3O - Wavelength**2) + 1)*np.cos(Theta)**2/(C3O - Wavelength**2)**3)/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2) - 3*((4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)**2/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2))/(-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2) - 2*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)*(-3*Wavelength**2*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)**2/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2) + (4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)*(4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O - Wavelength**2*(C2O/(C3O - Wavelength**2)**2 + C4O)**2/(-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2))/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)) - 2*np.sqrt(C1O - C2O/(C3O - Wavelength**2) - C4O*Wavelength**2)*(-3*Wavelength**2*((C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)**2/((-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 + (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2) + (4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E)*np.sin(Theta)**2 + (4*C2O*Wavelength**2/(C3O - Wavelength**2)**3 + C2O/(C3O - Wavelength**2)**2 + C4O)*np.cos(Theta)**2)*(4*C2E*Wavelength**2/(C3E - Wavelength**2)**3 + C2E/(C3E - Wavelength**2)**2 + C4E - Wavelength**2*(C2E/(C3E - Wavelength**2)**2 + C4E)**2/(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2))/((-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)*np.sqrt(C1E - C2E/(C3E - Wavelength**2) - C4E*Wavelength**2)))/np.sqrt(-(-C1E + C2E/(C3E - Wavelength**2) + C4E*Wavelength**2)*np.sin(Theta)**2 - (-C1O + C2O/(C3O - Wavelength**2) + C4O*Wavelength**2)*np.cos(Theta)**2)
        return D1_Pump, D2_Pump, D3_Pump, D4_Pump

    def Dispersion_Coefficients_Pump(self):
        Wavelength = self.Wavelength  # Unit  um
        Length = self.Crystal_Length  # Unit m
        N = self.Refrective_Index_Pump()
        D1, D2, D3, D4 = self.Differential_Coefficients_Pump()
        Dispersion_1st = 1/c0*(N - Wavelength*D1)
        Dispersion_2nd = (Wavelength**3)/2/math.pi/c0/c0*D2*1e-6
        Dispersion_3rd = -(Wavelength**4) /4/(math.pi**2)/c0/c0/c0*(3*D2 + Wavelength*D3)*1e-12
        Dispersion_4th = (Wavelength**5) /8/(math.pi**3)/c0/c0/c0/c0*(12*D2 + 8*Wavelength*D3 + Wavelength**2 * D4)*1e-18
       
        
        return  Dispersion_1st, Dispersion_2nd, Dispersion_3rd, Dispersion_4th
        #return 0, 0, 0, Dispersion_4th  
