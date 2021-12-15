'''
In this page, we will sovle three-wave coupling equation with 4th-order Runge-Kutta method as well as symmetric split step fourier transform

The basic idea of split step fourier transform is to slice the crytal up. At each slice (has a length of h), the dispersion effect and the nonlinear effect can be 
calculated separately and the error can be O(h^2). For symmetric split step fourier transform, the calculation erroer would be further decreased. The main process can be 
illustrate as 'Dispersion in frequency domain (h/2) ---> Nonlinear in time domain (h) ---> Dispersion in frequency domain (h/2)' 
'''

import numpy as np
import numpy.fft as nfft
from Dispersion_Coeffieient_Calculation import Dispersion_Coefficients_Calculation
from Phase_Matching_Angle_and_Gain import Crystal_Length, Phase_Matching_Angle, Noncollinear_Angle, d_eff, c0, Pulse_Duration_Seed, Pulse_Duration_Pump, Refrective_Index_O, Refrective_Index_Pump, Delta_k_Calculation
from Pulse_Parameter import PumpDict, SeedDict
from Dispersion_Coeffieient_Calculation_Pump import Dispersion_Coefficients_Calculation_Pump


Wavelength_Pump = PumpDict['Central_Wavelength(nm)']
Chirp_Constant_Seed = SeedDict['Chirp_Constant']
Chirp_Constant_Pump = PumpDict['Chirp_Constant']


class Runge_Kutta_4th:

    def __init__(self, Wavelength_Seed, Omega_List, Electric_Field_Frequency_Seed, Electric_Field_Frequency_Idler, Electric_Field_Frequency_Pump, Z_Length, Iteration_Number, Time_List, Z0 = 0):
        self.Wavelength_Seed = Wavelength_Seed #Unit nm
        self.Wavelength_Idler = 1/(1/Wavelength_Pump-1/Wavelength_Seed) #Unit nm
        self.Omega_List_Z0 = 2*np.pi*Omega_List
        self.Electrid_Field_Frequency_Seed_Z = []
        self.Electrid_Field_Frequency_Idler_Z = []
        self.Electrid_Field_Frequency_Pump_Z = []
        self.Time_List = Time_List
        self.Z0 = Z0
        self.Z_Distance, self.Delta_Z = np.linspace(self.Z0*1e-3, Z_Length*1e-3, Iteration_Number + 1, retstep = True)
        self.Electrid_Field_Frequency_Seed_Z.append(Electric_Field_Frequency_Seed)
        self.Electrid_Field_Frequency_Idler_Z.append(Electric_Field_Frequency_Idler)
        self.Electrid_Field_Frequency_Pump_Z.append(Electric_Field_Frequency_Pump)

    '''In frequency domain, '(partial f)/(partial time)' becomes (j*frequency)*f (Fourier Transform). So we will deal with dispersion in frequency domain 
    and solve nonlinear problem in time domain with 4th-order Runge-Kutta method'''
    
    def Dispersion_Operator_Seed(self, D1, D2, D3, D4, Wavelength, Electrid_Field_Frequency_Z):
        After_D_Operation = []
        Number = len(Electrid_Field_Frequency_Z)
        Omega_0 = 2*np.pi*1e9*c0/Wavelength
        for i in range(Number):
            Wavelength_i = 2*np.pi*c0*1e9/(Omega_0 + self.Omega_List_Z0[i])
            n_i = Refrective_Index_O(Wavelength_i)   
            After_D_Operation.append(Electrid_Field_Frequency_Z[i]*np.exp((self.Delta_Z/2)*(-1j*n_i*self.Omega_List_Z0[i]/c0 +1j*D1*self.Omega_List_Z0[i] - 1j*D2/2*(self.Omega_List_Z0[i])**2 + 1j*D3/6*(self.Omega_List_Z0[i])**3 - 1j*D4/24*(self.Omega_List_Z0[i])**4)))     
            #After_D_Operation.append(Electrid_Field_Frequency_Z[i]*np.exp((self.Delta_Z/2)*(-1j*D1*self.Omega_List_Z0[i] + 1j*D2/2*(self.Omega_List_Z0[i])**2 - 1j*D3/6*(self.Omega_List_Z0[i])**3 + 1j*D4/24*(self.Omega_List_Z0[i])**4)))     
        return After_D_Operation
    
        
    def Dispersion_Operator_Idler(self, D1, D2, D3, D4, Wavelength, Electrid_Field_Frequency_Z):
        After_D_Operation = []
        Number = len(Electrid_Field_Frequency_Z)
        Omega_0 = 2*np.pi*1e9*c0/Wavelength
        for i in range(Number):
            Wavelength_i = 2*np.pi*c0*1e9/(Omega_0 + self.Omega_List_Z0[i])
            n_i = Refrective_Index_O(Wavelength_i)
            After_D_Operation.append(Electrid_Field_Frequency_Z[i]*np.exp((self.Delta_Z/2)*(-1j*n_i*self.Omega_List_Z0[i]/c0 +1j*D1*(self.Omega_List_Z0[i]) - 1j*D2/2*(self.Omega_List_Z0[i])**2 + 1j*D3/6*(self.Omega_List_Z0[i])**3 - 1j*D4/24*(self.Omega_List_Z0[i])**4)))     
            #After_D_Operation.append(Electrid_Field_Frequency_Z[i]*np.exp((self.Delta_Z/2)*(-1j*D1*(self.Omega_List_Z0[i]) + 1j*D2/2*(self.Omega_List_Z0[i])**2 - 1j*D3/6*(self.Omega_List_Z0[i])**3 + 1j*D4/24*(self.Omega_List_Z0[i])**4)))     
        return After_D_Operation
   

    def Dispersion_Operator_Pump(self, D1, D2, D3, D4, Wavelength, Electrid_Field_Frequency_Z):
        After_D_Operation = []
        Number = len(Electrid_Field_Frequency_Z)
        Omega_0 = 2*np.pi*1e9*c0/Wavelength
        for i in range(Number):
            Wavelength_i = 2*np.pi*c0*1e9/(Omega_0 + self.Omega_List_Z0[i])
            n_i = Refrective_Index_Pump(Wavelength_i, Phase_Matching_Angle)
            After_D_Operation.append(Electrid_Field_Frequency_Z[i]*np.exp((self.Delta_Z/2)*(-1j*n_i*self.Omega_List_Z0[i]/c0 +1j*D1*(self.Omega_List_Z0[i]) - 1j*D2/2*(self.Omega_List_Z0[i])**2 + 1j*D3/6*(self.Omega_List_Z0[i])**3 - 1j*D4/24*(self.Omega_List_Z0[i])**4)))     
            #After_D_Operation.append(Electrid_Field_Frequency_Z[i]*np.exp((self.Delta_Z/2)*(-1j*D1*(self.Omega_List_Z0[i]) + 1j*D2/2*(self.Omega_List_Z0[i])**2 - 1j*D3/6*(self.Omega_List_Z0[i])**3 + 1j*D4/24*(self.Omega_List_Z0[i])**4)))     
        return After_D_Operation
    
    '''The dispersion operator for signal, idler and pump pulse. The high-order dispersion (up to 4th order have been taken into consideration)'''
        
    def Dispersion_Operation_Seed(self, Current_Step, Electrid_Field_Frequency_Z, Wavelength): #Current_Step start from 0
        Dispersion_Coefficient = Dispersion_Coefficients_Calculation(Wavelength, Crystal_Length, Phase_Matching_Angle)
        D1, D2, D3, D4 = Dispersion_Coefficient.Dispersion_Coefficients_O()
        After_D_Operation = self.Dispersion_Operator_Seed(D1, D2, D3, D4, Wavelength, Electrid_Field_Frequency_Z)
        After_D_Operation = nfft.fft(After_D_Operation)
        return After_D_Operation
    
    def Dispersion_Operation_Idler(self, Current_Step, Electrid_Field_Frequency_Z, Wavelength): #Current_Step start from 0
        Dispersion_Coefficient = Dispersion_Coefficients_Calculation(Wavelength, Crystal_Length, Phase_Matching_Angle)
        D1, D2, D3, D4 = Dispersion_Coefficient.Dispersion_Coefficients_O()
        After_D_Operation = self.Dispersion_Operator_Idler(D1, D2, D3, D4, Wavelength, Electrid_Field_Frequency_Z)
        After_D_Operation = nfft.fft(After_D_Operation)
        return After_D_Operation

    def Dispersion_Operation_Pump(self, Current_Step, Electrid_Field_Frequency_Z, Wavelength): #Current_Step start from 0
        Dispersion_Coefficient = Dispersion_Coefficients_Calculation_Pump(Wavelength, Crystal_Length, Phase_Matching_Angle)
        D1, D2, D3, D4 = Dispersion_Coefficient.Dispersion_Coefficients_Pump()
        After_D_Operation_Pump = self.Dispersion_Operator_Pump(D1, D2, D3, D4, Wavelength, Electrid_Field_Frequency_Z)
        After_D_Operation_Pump = nfft.fft(After_D_Operation_Pump)
        return After_D_Operation_Pump

    '''Nonlinear effect are calculated in time domain with 4th-ordere Runge-Kutta method. The detail can be seen in the phd thesis of Katalin Mecseki 
    'An ultrafast optical parametric laser for driving high energy density science' '''     

    def Nonlinear_Operation(self, A_p, A_s, A_i, Current_Step):
        Omega_S = 2*np.pi*c0/self.Wavelength_Seed*1e9 # Unit: s^{-1}
        Omega_I = 2*np.pi*c0/self.Wavelength_Idler*1e9
        Omega_P = 2*np.pi*c0/Wavelength_Pump*1e9
        k_S = Refrective_Index_O(self.Wavelength_Seed)*Omega_S/c0
        k_I = Refrective_Index_O(self.Wavelength_Idler)*Omega_I/c0
        k_P = Refrective_Index_Pump(Wavelength_Pump, Phase_Matching_Angle)*Omega_P/c0
        Delta_k = Delta_k_Calculation(self.Wavelength_Seed, Phase_Matching_Angle, Noncollinear_Angle)   

        Cs1 = 1j*self.Delta_Z*2*d_eff/c0/c0*(Omega_S**2)/k_S*A_p*np.conj(A_i)*np.exp(-1j*self.Z_Distance[Current_Step - 1]*Delta_k)
        Ci1 = 1j*self.Delta_Z*2*d_eff/c0/c0*(Omega_I**2)/k_I*A_p*np.conj(A_s)*np.exp(-1j*self.Z_Distance[Current_Step - 1]*Delta_k)
        Cp1 = 1j*self.Delta_Z*2*d_eff/c0/c0*(Omega_P**2)/k_P*A_s*A_i*np.exp(1j*self.Z_Distance[Current_Step - 1]*Delta_k)
       
        Cs2 = 1j*self.Delta_Z*2*d_eff/c0/c0*(Omega_S**2)/k_S*(A_p + Cp1/2)*(np.conj(A_i) + np.conj(Ci1)/2)*np.exp(-1j*(self.Z_Distance[Current_Step - 1] + self.Delta_Z/2)*Delta_k)
        Ci2 = 1j*self.Delta_Z*2*d_eff/c0/c0*(Omega_I**2)/k_I*(A_p + Cp1/2)*(np.conj(A_s) + np.conj(Cs1)/2)*np.exp(-1j*(self.Z_Distance[Current_Step - 1] + self.Delta_Z/2)*Delta_k)
        Cp2 = 1j*self.Delta_Z*2*d_eff/c0/c0*(Omega_P**2)/k_P*(A_s + Cs1/2)*(A_i + Ci1/2)*np.exp(1j*(self.Z_Distance[Current_Step - 1] + self.Delta_Z/2)*Delta_k)     
      
        Cs3 = 1j*self.Delta_Z*2*d_eff/c0/c0*(Omega_S**2)/k_S*(A_p + Cp2/2)*(np.conj(A_i) + np.conj(Ci2)/2)*np.exp(-1j*(self.Z_Distance[Current_Step - 1] + self.Delta_Z/2)*Delta_k)
        Ci3 = 1j*self.Delta_Z*2*d_eff/c0/c0*(Omega_I**2)/k_I*(A_p + Cp2/2)*(np.conj(A_s) + np.conj(Cs2)/2)*np.exp(-1j*(self.Z_Distance[Current_Step - 1] + self.Delta_Z/2)*Delta_k)
        Cp3 = 1j*self.Delta_Z*2*d_eff/c0/c0*(Omega_P**2)/k_P*(A_s + Cs2/2)*(A_i + Ci2/2)*np.exp(1j*(self.Z_Distance[Current_Step - 1] + self.Delta_Z/2)*Delta_k)     
        
        Cs4 = 1j*self.Delta_Z*2*d_eff/c0/c0*(Omega_S**2)/k_S*(A_p + Cp3)*(np.conj(A_i) + np.conj(Ci3))*np.exp(-1j*(self.Z_Distance[Current_Step - 1] + self.Delta_Z)*Delta_k)
        Ci4 = 1j*self.Delta_Z*2*d_eff/c0/c0*(Omega_I**2)/k_I*(A_p + Cp3)*(np.conj(A_s) + np.conj(Cs3))*np.exp(-1j*(self.Z_Distance[Current_Step - 1] + self.Delta_Z)*Delta_k)
        Cp4 = 1j*self.Delta_Z*2*d_eff/c0/c0*(Omega_P**2)/k_P*(A_s + Cs3)*(A_i + Ci3)*np.exp(1j*(self.Z_Distance[Current_Step - 1] + self.Delta_Z)*Delta_k)     
    
        A_s_next = A_s + (Cs1 + 2*(Cs2 + Cs3) + Cs4)/6       
        A_i_next = A_i + (Ci1 + 2*(Ci2 + Ci3) + Ci4)/6       
        A_p_next = A_p + (Cp1 + 2*(Cp2 + Cp3) + Cp4)/6

        return nfft.ifft(A_s_next), nfft.ifft(A_i_next), nfft.ifft(A_p_next)  

    '''4th order Runge-Kutta method solving nonlinear effect in time domain. For a detailed description of Runge-Kutta method, we redirect you to the phd thesis of Xuan Liu
    'SIMULATIONS OF ULTRAFAST PULSE MEASUREMENTS' '''
  
    def Propagation(self):
        for i in range(1, self.Z_Distance.size):
            A_s = self.Dispersion_Operation_Seed(i, self.Electrid_Field_Frequency_Seed_Z[i - 1], self.Wavelength_Seed)
            A_i = self.Dispersion_Operation_Idler(i, self.Electrid_Field_Frequency_Idler_Z[i - 1], self.Wavelength_Idler)
            A_p = self.Dispersion_Operation_Pump(i, self.Electrid_Field_Frequency_Pump_Z[i - 1], Wavelength_Pump)

            Electric_Field_Seed, Electrid_Field_Idler, Electric_Field_Pump = self.Nonlinear_Operation(A_p, A_s, A_i, i)

            A_s2 = self.Dispersion_Operation_Seed(i, Electric_Field_Seed, self.Wavelength_Seed)
            A_i2 = self.Dispersion_Operation_Idler(i, Electrid_Field_Idler, self.Wavelength_Idler)
            A_p2 = self.Dispersion_Operation_Pump(i, Electric_Field_Pump, Wavelength_Pump)
           
            Electric_Field_Seed_After_One_Step = nfft.ifft(A_s2)
            Electric_Field_Idler_After_One_Step = nfft.ifft(A_i2)
            Electric_Field_Pump_After_One_Step = nfft.ifft(A_p2)

            self.Electrid_Field_Frequency_Seed_Z.append(Electric_Field_Seed_After_One_Step)
            self.Electrid_Field_Frequency_Idler_Z.append(Electric_Field_Idler_After_One_Step)
            self.Electrid_Field_Frequency_Pump_Z.append(Electric_Field_Pump_After_One_Step)

    '''The symmetric split step fourier transform method '''   






