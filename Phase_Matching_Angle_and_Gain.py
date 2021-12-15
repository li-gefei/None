from Pulse_Parameter import SeedDict, PumpDict
import numpy as np
import math
import xlwt
import matplotlib.pyplot as plt
import cmath 
import matplotlib.pyplot as mp

'''
In this page, we calculate the best phase matching angle as well as non-collinear angle for broadband OPA process.
In order to find the best angles, we calculated the phase matching factor at each seed beam wavelength, summation and normalization have been conduct afterwards.
The results were compared at different angles to find the best circumstance.

The following calculated result based on beta - BBO (d_eff, Sellmeier equations)
'''

c0 = float(299792458) # Unit SI
Epsilon = 8.854187817e-12 # Unit SI
d_eff = 2.02*1e-12  # Unit SI  
Crystal_Length = float(3) #Unit mm
Interval = 50 
Delta_Phase_Matching_Angle = 0.05 # The calculated interval      Unit  degree
Delta_Noncollinear_Angle = 0.01
Phase_Matching_Angle_Min = 22.5   # The initial value       Unit  degree
Noncollinear_Angle_Min = 2.15
Data_Factor = np.ndarray((Interval, Interval))   # List to save data
Wavelength_Pump = PumpDict['Central_Wavelength(nm)']
Central_Wavelength = SeedDict['Central_Wavelength(nm)']
Chirp_Constant_Seed = SeedDict['Chirp_Constant']
FWHM_Frequency_Seed = SeedDict['FWHM(THz)']
Pulse_Duration_Pump = PumpDict['Pulse_Duration(ps)']
Peak_Intensity_Pump =  PumpDict['Peak_Intensity(GW/cm^2)']

if abs(Chirp_Constant_Seed) > 1e-10:
    Pulse_Duration_Seed = 1e12*2*np.log(2)*np.sqrt(2)/np.sqrt((FWHM_Frequency_Seed*1e12*np.pi)**2 - np.sqrt((FWHM_Frequency_Seed*1e12*np.pi)**4 - 16*Chirp_Constant_Seed*Chirp_Constant_Seed*(np.log(2))**2))
else:
    Pulse_Duration_Seed = 1e12*2*np.log(2)/np.pi/FWHM_Frequency_Seed/1e12

Delta_Omega = 1e12*FWHM_Frequency_Seed*2*np.pi
Omega_Central = 2*np.pi*1e9*c0/Central_Wavelength
Omega_Min = Omega_Central - Delta_Omega/2
Omega_Max = Omega_Central + Delta_Omega/2
Wavelength_Seed_Min = 2*np.pi*1e9*c0/Omega_Max # Unit nm
Wavelength_Seed_Max = 2*np.pi*1e9*c0/Omega_Min
Delta_Wavelength = Wavelength_Seed_Max - Wavelength_Seed_Min # Unit nm

Var_Theta = [Phase_Matching_Angle_Min + i*Delta_Phase_Matching_Angle for i in range(0, Interval, 1)]
Var_Alpha = [Noncollinear_Angle_Min + i*Delta_Noncollinear_Angle for i in range(0, Interval, 1)]
Gain_Curve = [i for i in range(int(Delta_Wavelength))]
Correspond_Wavelength = [i for i in range(int(Delta_Wavelength))]
Delta_k_Optimum = [i for i in range(int(Delta_Wavelength))]


def Refrective_Index_O(Wavelength):       # Calculate the refrective index of ordinary wave
    n_o = math.sqrt(2.7366122+0.0185720/(Wavelength/1000*Wavelength/1000-0.0178746)-0.0143756*Wavelength/1000*Wavelength/1000)
    return n_o

def Refrective_Index_E(Wavelength):      # Calculate the refrective index of extrodinary wave at \theta = 0
    n_e = math.sqrt(2.3698703+0.0128445/(Wavelength/1000*Wavelength/1000-0.0153064)-0.0029129*Wavelength/1000*Wavelength/1000)
    return n_e

def Refrective_Index_Pump(Wavelength, Theta):  # Calculate the refrective index of extrodinary wave at arbitrary \theta
    n_p = math.sqrt(1/(math.cos(Theta/180*math.pi)*math.cos(Theta/180*math.pi)/Refrective_Index_O(Wavelength)/Refrective_Index_O(Wavelength) + math.sin(Theta/180*math.pi)*math.sin(Theta/180*math.pi)/Refrective_Index_E(Wavelength)/Refrective_Index_E(Wavelength)))
    return n_p

'''
Herein, we set the pump beam as e wave and signal and idler beam as o waves, i.e. type I phase matching
'''

def Delta_k_Calculation(Wavelength_Seed, Theta, Alpha):  # Calculate Delta k for monochromatic signal beam 
    Wavelength_Idler = 1/(1/Wavelength_Pump-1/Wavelength_Seed)
    Delta_k = 2*math.pi*1e9*(math.sqrt((Refrective_Index_Pump(Wavelength_Pump, Theta)/Wavelength_Pump)**2   +  (Refrective_Index_O(Wavelength_Seed)/Wavelength_Seed)**2 - 2*math.cos(Alpha/180*math.pi)*Refrective_Index_Pump(Wavelength_Pump, Theta)/Wavelength_Pump*Refrective_Index_O(Wavelength_Seed)/Wavelength_Seed) - Refrective_Index_O(Wavelength_Idler)/Wavelength_Idler)   
    return Delta_k

def Delta_k_Summation(Theta, Alpha):  # Sum Delta k to evaluate th broadband OPA process, the summation result is Delta k factor, sinc^2(delta k/2)
    Sum = 0
    Mesh = np.linspace(Wavelength_Seed_Min, Wavelength_Seed_Max, num = int(Delta_Wavelength), endpoint = False, retstep = False)
    for item in Mesh:
        Sum += np.sinc(Delta_k_Calculation(item, Theta, Alpha)*Crystal_Length/2*1e-3)*np.sinc(Delta_k_Calculation(item, Theta, Alpha)*Crystal_Length/2*1e-3) 
    return Sum

def Normalized(Data):
    if np.max(Data) - np.min(Data) < 1e-8:  # A general normalized function
        return Data/np.max(Data)
    else:
        N_Data = (Data - np.min(Data)) / (np.max(Data) - np.min(Data))
    return N_Data

def Calculate_Factor(Factor, P_M_Angle_Min, Delta_P_M_Angle, N_C_Angle_Min, Delta_N_C_Angle):
    for count_x in range(0, Interval, 1):
        for count_y in range(0, Interval, 1):
            Factor[count_x][count_y] = Delta_k_Summation(P_M_Angle_Min + count_x*Delta_P_M_Angle, N_C_Angle_Min + count_y*Delta_N_C_Angle)
    return Factor

def Gain_Calculation(Wavelength_S):   # The defination of Gain can be seen in "Ultrafast optical parametric amplifiers"
    Wavelength_I = 1/(1/Wavelength_Pump-1/Wavelength_S)
    Delta_k_O = Delta_k_Calculation(Wavelength_S, Phase_Matching_Angle, Noncollinear_Angle)
    Gamma = cmath.sqrt(Peak_Intensity_Pump*1e13*d_eff*d_eff*1e9*1e9*8*math.pi*math.pi/(2*Epsilon*Refrective_Index_Pump(Wavelength_Pump, Phase_Matching_Angle)*Refrective_Index_O(Wavelength_I)*Refrective_Index_O(Wavelength_S)*c0*Wavelength_I*Wavelength_S))
    g = cmath.sqrt(Gamma**2 - (Delta_k_O/2)**2)
    Gain = 1 + (Gamma**2/g**2)*cmath.sinh(g*Crystal_Length*1e-3)*cmath.sinh(g*Crystal_Length*1e-3)
    return Gain, Delta_k_O*Crystal_Length*1e-3

def Gain_Summation():
    count = 0
    Cal_Gain_Curve = [i for i in range(int(Delta_Wavelength))]
    Delta_k_Opt = [i for i in range(int(Delta_Wavelength))]
    Corr_Wave = [i for i in range(int(Delta_Wavelength))]
    Mesh = np.linspace(Wavelength_Seed_Min,  Wavelength_Seed_Max, num = int(Delta_Wavelength), endpoint = False, retstep = False)
    for item in Mesh:
        Corr_Wave[count] = item
        Cal_Gain_Curve[count], Delta_k_Opt[count] = Gain_Calculation(item)
        count = count + 1
    return Corr_Wave, Cal_Gain_Curve, Delta_k_Opt


# Calculate delta k
Data_Factor = Calculate_Factor(Data_Factor, Phase_Matching_Angle_Min, Delta_Phase_Matching_Angle, Noncollinear_Angle_Min, Delta_Noncollinear_Angle)
Normalized_Factor = Normalized(Data_Factor)


# The ideal phase matching angle as well as non-collinear angle for broadband OPA
Phase_Matching_Angle_Index, Noncollinear_Angle_Index,  = np.unravel_index(Normalized_Factor.argmax(), Normalized_Factor.shape)	# 
Phase_Matching_Angle = Phase_Matching_Angle_Index*Delta_Phase_Matching_Angle + Phase_Matching_Angle_Min
Noncollinear_Angle = Noncollinear_Angle_Index*Delta_Noncollinear_Angle + Noncollinear_Angle_Min


# Calculate Gain
Correspond_Wavelength, Gain_Curve, Delta_k_Optimum = Gain_Summation()   
Normalized_Gain_Curve = Normalized(Gain_Curve)



####################################  Plot and Save data  ###############################3
if __name__ == "__main__":
    print(Phase_Matching_Angle, Noncollinear_Angle) 
    Fig_Delta_k_Gain = plt.figure()           #Phase mismatch as different wavelength & the normalized gain curve
    Delta_k_Plot = Fig_Delta_k_Gain.add_subplot(111)
    Delta_k_Plot.plot(Correspond_Wavelength, Delta_k_Optimum, color = "dodgerblue", linestyle = "-",  linewidth = 1, label = "$\Delta$ k")
    Delta_k_Plot.set_ylabel('$\Delta k$ (rad)')
    Gain_Plot = Delta_k_Plot.twinx() # this is the important function
    Gain_Plot.plot(Correspond_Wavelength, Normalized_Gain_Curve, color = "tomato", linestyle = "-",  linewidth = 1, label = "Gain")
    Gain_Plot.set_ylabel('Gain (a.u.)')
    Gain_Plot.set_xlabel('Wavelength (nm)')
    mp.gcf().autofmt_xdate()  
    plt.show()

    plt.rc('font',family='Times New Roman')    #Contour plot for phase matching factor at different theta and alpha.
    Delta_k_Plot = plt.contourf(Var_Alpha, Var_Theta, Normalized_Factor, 5000, cmap ='YlGnBu')
    plt.colorbar(Delta_k_Plot)
    plt.xlabel('Non-collinear Angle (deg)', fontsize = 10)
    plt.ylabel('Phase Matching Angle (deg)', fontsize = 10)
    plt.scatter(Noncollinear_Angle, Phase_Matching_Angle, color ='k', marker = '.',  s = 200)
    plt.annotate(r'($\alpha$, $\theta$) = (%.2f$^\circ$, %.2f$^\circ$)'%(Noncollinear_Angle, Phase_Matching_Angle), xy = (Noncollinear_Angle, Phase_Matching_Angle, ), xytext = (Noncollinear_Angle - Delta_Noncollinear_Angle*10, Phase_Matching_Angle + Delta_Phase_Matching_Angle*10), arrowprops = dict(arrowstyle = '->',
    connectionstyle = 'arc3,rad=0'),  xycoords = 'data', va = "center",  ha = "center")
    plt.show()




    '''
    Following code saves data to .xls type file.
    '''

    Delta_k_and_Gain = xlwt.Workbook()
    Phase_Matching_Factor = Delta_k_and_Gain.add_sheet("Phase_Matching_Factor")
    Gain_Factor = Delta_k_and_Gain.add_sheet("Gain_Factor")

    for i in range(len(Normalized_Factor)):
        for j in range(len(Normalized_Factor[i])):
            Phase_Matching_Factor.write(i, j, Normalized_Factor[i][j])

    for i in range(len(Normalized_Factor)):
            Gain_Factor.write(0, i, Correspond_Wavelength[i])
    for i in range(len(Normalized_Factor)):
            Gain_Factor.write(1, i, np.real(Normalized_Gain_Curve[i]))

    Delta_k_and_Gain.save("Delta_k_and_Gain_at_Different_Phase_Matching_Angle_and_Noncollinear_Angle.xls")


