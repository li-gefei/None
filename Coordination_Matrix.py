'''
This page denotes to build a matrix with discrete data points (both in time domain and frequency domain) to conduct spit step fourier transformation
'''

import numpy as np
import numpy.fft as nfft
import math



'''
Time_Duration: Unit = ps
Crystal_Length: Unit = mm
Omega: Unit = ps^{-1}
'''


class Coordination_Matrix:

    def __init__(self, Time_Duration, Time_Number, Crystal_Length, Iteration_Number):
        self.Time_Duration = Time_Duration*1e-12
        self.Time_Number = Time_Number
        self.Crystal_Length = Crystal_Length*1e-3
        self.Iteration_Number = Iteration_Number
        self.Time_List, self.Delta_Time = self.Set_Time()
        self.Omega_List, self.Delta_Omega = self.Set_Omega()
        self.Z_List, self.Delta_Z = self.Set_Z()


    def Set_Time(self):
        return np.linspace(-self.Time_Duration*3/2, self.Time_Duration*3/2, int(self.Time_Number), endpoint=False, retstep=True)

    def Set_Omega(self):
        Omega = nfft.fftfreq(self.Time_List.size, d = self.Delta_Time)
        Delta_Omega = Omega[1] - Omega[0]
        return Omega, Delta_Omega
    
    def Set_Z(self):
        return np.linspace(0, self.Crystal_Length, self.Iteration_Number + 1, retstep=True)


if __name__ == "__main__":
    Coordination_matrix = Coordination_Matrix(0.12, math.pow(2, 14), 3, 1000)
    print(Coordination_matrix.Omega_List)