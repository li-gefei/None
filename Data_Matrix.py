'''
In this page, we will discrete the electric fields at different and express them in time domain as well as frequency domain
'''

import numpy as np
import numpy.fft as nfft
import math
import matplotlib.pyplot as plt
from Coordination_Matrix import Coordination_Matrix
'''
Herein, we assume the electric fields of all three types of beams have Guassian shape in time domain.
Especially, the seed beam has a super Guassian shape (m = 20) in frequency domain, i.e. Intensity is constant for all wavelengths
'''

class Data_Matrix:

    def __init__(self, Electric_Field):
        self.Electric_Field = np.asarray(Electric_Field)
        self.Time_Number = self.Electric_Field.size
        self.Electric_Field_Frequency = self.Fourier_Transformation()
        self.Electric_Field_Time = self.Inverse_Fourier_Transformation()
    
    def Fourier_Transformation(self):
        return nfft.ifft(self.Electric_Field)     

    def Inverse_Fourier_Transformation(self):
        return nfft.fft(self.Electric_Field_Frequency)

    def Test_Orthogonality(self):
        assert np.sum(np.real(self.Electric_Field_Time) * np.imag(self.Electric_Field_Time)) < 1e-8
        
    def Test_Recover_Original_Field(self):
        assert np.max(abs(self.Electric_Field - np.real(self.Electric_Field_Time))) < 1e-7

Electric_Field_Matrix = Data_Matrix



if __name__ == "__main__":
    c0 = float(299792458) # Unit SI
    Epsilon = 8.854187817e-12 # Unit SI
    d_eff = 2.02*1e-12  # Unit SI
    n = int(math.pow(2, 16))
    Test_Matrix = Coordination_Matrix(20, n, 5, 150)
    Crystal_Length = float(5)
    Iteration_Number = 150

    def Guassian_Function(n, tau = 20, I0 = 1e-5):
        Guassian = [math.exp(-4*np.log(2)*(i*3*120/n/tau)**2) for i in range(-int(n/2), int((n/2)))]
        Amp = []
        for keys in Guassian:
                Amp.append(math.sqrt(I0*2/Epsilon/c0/1.65)*keys)
        return Amp

    x = Guassian_Function(n)
    Z = Data_Matrix(x)
    Z.Test_Orthogonality()
    Z.Test_Recover_Original_Field()
  
    A_test = np.exp((Crystal_Length*1e-3/Iteration_Number)*1j*1e-9*2*np.pi*c0*1e9/800)*Z.Electric_Field_Frequency

   
    plt.plot(Test_Matrix.Time_List, -nfft.fft(A_test).real)
    plt.plot(Test_Matrix.Time_List, nfft.fft(Z.Electric_Field_Frequency))
    plt.show()

    plt.plot(Test_Matrix.Time_List, nfft.fft(Z.Electric_Field_Frequency) + nfft.fft(A_test).real)
    plt.show()