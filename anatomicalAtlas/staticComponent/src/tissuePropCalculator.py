import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas

"""
Based on http://niremf.ifac.cnr.it/tissprop/

C.Gabriel, S.Gabriel and E.Corthout: The dielectric properties of biological tissues: I. Literature survey, Phys. Med. Biol. 41 (1996), 2231-2249.
S.Gabriel, R.W.Lau and C.Gabriel: The dielectric properties of biological tissues: II. Measurements in the frequency range 10 Hz to 20 GHz, Phys. Med. Biol. 41 (1996), 2251-2269.
S.Gabriel, R.W.Lau and C.Gabriel: The dielectric properties of biological tissues: III. Parametric models for the dielectric spectrum of tissues, Phys. Med. Biol. 41 (1996), 2271-2293.

"""

class TissueCalculator():
    def __init__(self):
        self.csvTissueProp = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'electricalProperties_tissues.csv')
        self.dataFrame = pandas.read_csv(self.csvTissueProp, sep=';', header=0, skiprows=1)

    def getElectricalProp(self, tissueName='Cartilage', frequency_Hz=1000.0,uncertainty_Perc=0.2):
        """
        calculate the dielectric properties of human body tissues in the frequency range
        from 10 Hz to 100 GHz using the parametric model and the parameter values developed by C.Gabriel and collegues

        Parameters
        ----------
        tissueName: string
            Name of the tissue. ATTENTION, this name must match exactly the contents of the csv file!

        frequency_Hz: float
            Frequency in the range  10 Hz to 100 GHz

        uncertainty_Perc: float
            uncertainty percentage between 0.1 and 1.0 . GAbriel's paper suggest using 20%

        Returns
        -------
            properties: dictionary
                each entry of the dictionary is of the form
                     key: [value, stdev]
                    the keys are 'conductivity', 'rel_perm', 'resistivity'

                    conductivity -  unit: S/m
                    rel_permittivity - unit: dimensionless.
                    resistivity - unit Ohm/m

        """
        self.tissueProp = self.dataFrame.loc[self.dataFrame['Tissue'] == tissueName]

        # angular frequency
        omega=2*np.pi*frequency_Hz

        #permittivity of free space  (F/m)
        E0=8.8542e-12

        # ATTENTION: tau_i must be converted to seconds, accordingly with their unit
        # the table can be found (2021) here: http://niremf.ifac.cnr.it/docs/DIELECTRIC/AppendixC.html#C14
        ef = self.tissueProp.iloc[0]['ef']
        del1 = self.tissueProp.iloc[0]['del1']
        tau1 = self.tissueProp.iloc[0]['tau1_(ps)']*1e-12
        alf1 = self.tissueProp.iloc[0]['alf1']
        del2 = self.tissueProp.iloc[0]['del2']
        tau2 = self.tissueProp.iloc[0]['tau2_(ns)']*1e-9
        alf2 = self.tissueProp.iloc[0]['alf2']
        sig = self.tissueProp.iloc[0]['sig']
        del3 = self.tissueProp.iloc[0]['del3']
        tau3 = self.tissueProp.iloc[0]['tau3_(us)']*1e-6
        alf3 = self.tissueProp.iloc[0]['alf3']
        del4 = self.tissueProp.iloc[0]['del4']
        tau4 = self.tissueProp.iloc[0]['tau4_(ms)']*1e-3
        alf4 = self.tissueProp.iloc[0]['alf4']

        delVec=np.array([del1,del2,del3,del4])
        tauVec=np.array([tau1,tau2,tau3,tau4])
        alfVec=np.array([alf1,alf2,alf3,alf4])

        # for more details see GABRIEL-2000 - Dielectric properties of human tissues: definitions, parametric model, computing codes
        # equations 21 and  22
        # document can be found (2021) here: http://niremf.ifac.cnr.it/tissprop/document/tissprop.pdf

        # complex dielectric constant: Ec
        # cole-cole equation
        Ec=ef+sig/(1j*omega*E0)
        for i in range(4):
            term=delVec[i]/(1+(1j*omega*tauVec[i])**(1-alfVec[i]))
            Ec+=term

        # material conductivity:  sigma= - ang_feq * E0 * imag(Ec)
        # material relative permittivity: Er= real(Ec)

        rel_permittivity =np.real(Ec)
        conductivity = -omega*E0*np.imag(Ec)
        resistivity = np.real(1/conductivity)

        #uncertainties
        std_cond=uncertainty_Perc*conductivity
        std_perm=uncertainty_Perc*rel_permittivity

        #Uncertainty propagation of r=1/s   =>  std_r=std_s/s^2
        std_resist=std_cond/(conductivity**2)

        properties = {'conductivity':[conductivity,std_cond],'rel_perm':[rel_permittivity,std_perm],'resistivity': [resistivity,std_resist]}
        return properties

    def getConductivity(self, tissueName='Cartilage', frequency_Hz=1000.0,uncertainty_Perc=0.2):
        # returns a list: [value, stdev]
        propDict = self.getElectricalProp(tissueName,frequency_Hz,uncertainty_Perc)
        return propDict['conductivity']

    def getRelPermittivity(self, tissueName='Cartilage', frequency_Hz=1000.0,uncertainty_Perc=0.2):
        # returns a list: [value, stdev]
        propDict = self.getElectricalProp(tissueName,frequency_Hz,uncertainty_Perc)
        return propDict['rel_perm']

    def getResistivity(self, tissueName='Cartilage', frequency_Hz=1000.0,uncertainty_Perc=0.2):
        # returns a list: [value, stdev]
        propDict = self.getElectricalProp(tissueName,frequency_Hz,uncertainty_Perc)
        return propDict['resistivity']

if __name__ == '__main__':

    if sys.version_info.major == 2:
        sys.stdout.write("Sorry! This program requires Python 3.x\n")
        sys.exit(1)

    tissues = TissueCalculator()
    freqVec=np.logspace(1,11,51)

    Tissue = 'Blood'

    sigma=np.array([tissues.getConductivity(Tissue, frequency_Hz=f,uncertainty_Perc=0.2)[0] for f in freqVec])
    rho=np.array([tissues.getResistivity(Tissue, frequency_Hz=f,uncertainty_Perc=0.2)[0] for f in freqVec])
    perm=np.array([tissues.getRelPermittivity(Tissue, frequency_Hz=f,uncertainty_Perc=0.2)[0] for f in freqVec])

    plt.figure(1)
    plt.plot(freqVec,sigma,'r.-',label='cond (S/m)')
    plt.plot(freqVec,rho,'g.-',label='resist (Ohm/m)')
    plt.plot(freqVec,perm,'b.-',label='rel perm. (dimensionless)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()
    print('Fim!')
