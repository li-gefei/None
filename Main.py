'''This is the main function to simulate the broadband OPA process'''

import numpy as np
import matplotlib as plt
import xlwt
import math
from Pulse_Parameter import SeedDict, PumpDict
from Coordination_Matrix import Coordination_Matrix
from Data_Matrix import Data_Matrix, Electric_Field_Matrix
from Propagation_OPA import Runge_Kutta_4th, Chirp_Constant_Pump, Chirp_Constant_Seed
import numpy.fft as nfft
import pandas as pd
import matplotlib.pyplot as plt
from Phase_Matching_Angle_and_Gain import c0, Epsilon, d_eff, Crystal_Length, Wavelength_Pump, Pulse_Duration_Pump, Pulse_Duration_Seed

'''See 'Phase_Matching_Angle_and_Gain' module for Crystal Parameters'''

Time_Window = int(math.pow(2, 16))

Iteration_Number = int(300)

Peak_Intensity_Pump = PumpDict['Peak_Intensity(GW/cm^2)']
Peak_Intensity_Seed = SeedDict['Peak_Intensity(GW/cm^2)']
Central_Wavelength_Seed = SeedDict['Central_Wavelength(nm)']
Wavelength_Idler = 1/(1/Wavelength_Pump-1/Central_Wavelength_Seed)

'''First of first, we need to determine the pulse shape and intensity. We herein assume that the temperol profiles of both pump, seed and idler 
beam are Gaussian shape.'''

def Chirp_Gaussian_Pulse(Wavelength, Peak_Intensity = 0, Pulse_Duration = 1e5, Chirp_Constant = Chirp_Constant_Seed):
        Electric_Field_Peak_Intensity = math.sqrt(Peak_Intensity*1e13*2/Epsilon/c0/1.65)
        Gamma_2 = Chirp_Constant
        Amp = []
        for i in range(-int(Time_Window/2), int(Time_Window/2)):
                Amp.append(Electric_Field_Peak_Intensity*np.exp(-(2*np.log(2)/((Pulse_Duration*1e-12)**2) - 1j*Gamma_2)*((Pulse_Duration_Pump*1e-12*3/Time_Window*i)**2)))   
        return Amp

'''The time resolution is determined by 3*Pump_Pulse_Duration/Time_Window. In our case, the time resolution is about 2 fs'''

Seed_Pulse_Initial = Chirp_Gaussian_Pulse(Central_Wavelength_Seed, Peak_Intensity_Seed,  Pulse_Duration_Seed, Chirp_Constant_Seed)
Pump_Pulse_Initial = Chirp_Gaussian_Pulse(Wavelength_Pump, Peak_Intensity_Pump,  Pulse_Duration_Pump, Chirp_Constant_Pump)
Idler_Pulse_Initial = Chirp_Gaussian_Pulse(Wavelength_Idler)

'''The initial idler pulse has been set to be zero'''

Coor_Matrix = Coordination_Matrix(Pulse_Duration_Pump, Time_Window, Crystal_Length, Iteration_Number)

'''With this coordination matrix, the centrol frequency of each beam equals zero (for frequency domain), which make it easy dealing with dispersion'''

Seed_Matrix = Data_Matrix(Seed_Pulse_Initial)
Pump_Matrix = Data_Matrix(Pump_Pulse_Initial)
Idler_Matrix = Data_Matrix(Idler_Pulse_Initial)

'''Seed/Pump/Idler_Matrix gives the electric field expression both in time domain and frequency domain'''

OPA_Simulation = Runge_Kutta_4th(Central_Wavelength_Seed, Coor_Matrix.Omega_List, Seed_Matrix.Electric_Field_Frequency, Idler_Matrix.Electric_Field_Frequency, Pump_Matrix.Electric_Field_Frequency, Crystal_Length, Iteration_Number,Coor_Matrix.Time_List, 0)
OPA_Simulation.Propagation()

if __name__ == "__main__":      

        p = nfft.fft(OPA_Simulation.Electrid_Field_Frequency_Pump_Z[Iteration_Number - 1])
        plt.plot(Coor_Matrix.Time_List, np.sqrt((Pump_Matrix.Electric_Field * np.conj(Pump_Matrix.Electric_Field))))
        plt.plot(Coor_Matrix.Time_List, np.sqrt((p*np.conj(p))))
        plt.show()


        s = nfft.fft(OPA_Simulation.Electrid_Field_Frequency_Seed_Z[Iteration_Number - 1])
        plt.plot(Coor_Matrix.Time_List, np.sqrt((Seed_Matrix.Electric_Field * np.conj(Seed_Matrix.Electric_Field))))
        plt.plot(Coor_Matrix.Time_List, np.sqrt((s*np.conj(s))))
        plt.show()

        
        id = nfft.fft(OPA_Simulation.Electrid_Field_Frequency_Idler_Z[Iteration_Number - 1])
        plt.plot(Coor_Matrix.Time_List, np.sqrt((Idler_Matrix.Electric_Field * np.conj(Idler_Matrix.Electric_Field))))
        plt.plot(Coor_Matrix.Time_List, np.sqrt((id*np.conj(id))))
        plt.show()

        p2 = OPA_Simulation.Electrid_Field_Frequency_Pump_Z[Iteration_Number - 1]
        plt.plot(2*np.pi*Coor_Matrix.Omega_List, np.sqrt((p2*np.conj(p2))))
        plt.show()

        s2 = OPA_Simulation.Electrid_Field_Frequency_Seed_Z[Iteration_Number - 1]
        plt.plot(2*np.pi*Coor_Matrix.Omega_List, np.sqrt((s2*np.conj(s2))))
        plt.show()

        idler = OPA_Simulation.Electrid_Field_Frequency_Idler_Z[Iteration_Number - 1]
        plt.plot(2*np.pi*Coor_Matrix.Omega_List, np.sqrt((idler*np.conj(idler))))
        plt.show()

        Data_Seed_Electric_Field_Time_Domain = pd.DataFrame()
        Data_Seed_Electric_Field_Frequency_Domain = pd.DataFrame()
        Data_Idler_Electric_Field_Time_Domain = pd.DataFrame()
        Data_Idler_Electric_Field_Frequency_Domain = pd.DataFrame()
        Data_Seed_Electric_Field_Time_Domain_Initial = pd.DataFrame()
        Data_Seed_Electric_Field_Frequency_Domain_Initial = pd.DataFrame()
        Data_Coordination = pd.DataFrame()

        Data_Seed_Electric_Field_Time_Domain['Real'] = s.real
        Data_Seed_Electric_Field_Time_Domain['Imag'] = s.imag
        Data_Seed_Electric_Field_Frequency_Domain['Real'] = s2.real
        Data_Seed_Electric_Field_Frequency_Domain['Imag'] = s2.imag
        Data_Idler_Electric_Field_Time_Domain['Real'] = id.real
        Data_Idler_Electric_Field_Time_Domain['Imag'] = id.imag
        Data_Idler_Electric_Field_Frequency_Domain['Real'] = idler.real
        Data_Idler_Electric_Field_Frequency_Domain['Imag'] = idler.imag
        Data_Seed_Electric_Field_Time_Domain_Initial['Real'] = Seed_Matrix.Electric_Field_Time.real
        Data_Seed_Electric_Field_Time_Domain_Initial['Imag'] = Seed_Matrix.Electric_Field_Frequency.imag
        Data_Seed_Electric_Field_Frequency_Domain_Initial['Real'] = Seed_Matrix.Electric_Field_Frequency.real
        Data_Seed_Electric_Field_Frequency_Domain_Initial['Imag'] = Seed_Matrix.Electric_Field_Frequency.imag  
        Data_Coordination['Time'] = Coor_Matrix.Time_List
        Data_Coordination['Frequency'] = Coor_Matrix.Omega_List

        Data_Seed_Electric_Field_Time_Domain.to_csv('Data_Seed_Electric_Field_Time_Domain.csv',index=None, mode='a', header=False)
        Data_Seed_Electric_Field_Frequency_Domain.to_csv('Data_Seed_Electric_Field_Frequency_Domain.csv',index=None, mode='a', header=False) 
        Data_Idler_Electric_Field_Time_Domain.to_csv('Data_Idler_Electric_Field_Time_Domain.csv',index=None, mode='a', header=False)
        Data_Idler_Electric_Field_Frequency_Domain.to_csv('Data_Idler_Electric_Field_Frequency_Domain.csv',index=None, mode='a', header=False) 
        Data_Seed_Electric_Field_Time_Domain_Initial.to_csv('Data_Seed_Electric_Field_Time_Domain_Initial.csv',index=None, mode='a', header=False) 
        Data_Seed_Electric_Field_Frequency_Domain_Initial.to_csv('Data_Seed_Electric_Field_Frequency_Domain_Initial.csv',index=None, mode='a', header=False)
        Data_Coordination.to_csv('Data_Coordination.csv',index=None, mode='a', header=False) 




