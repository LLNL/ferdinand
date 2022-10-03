##############################################

#                                            #
#    Ferdinand 0.41, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################
import numpy
from fudge.processing.resonances.getCoulombWavefunctions import *
from Whittaker import Whittaker 


def getCoulomb_PSdSW(e_ch,lch, prmax, pMass,tMass,pZ,tZ, fmscal,etacns, shifty):
#
#  Return Penetrability, Shift Function, energy derivative of the Shift Function, W
#
    #print "getCoul:",e_ch,lch, prmax, pMass,tMass,pZ,tZ, fmscal,etacns, shifty
    fmscal = 0.0478450
    etacns = 0.1574855
    hbc =   197.3269788e0

    if pMass<1e-5:  # gamma
        penetrability = 1.0
        shift = 0.0
        dSdE = 0.0
        whit = 1.0
    else:           # massive particles
        redmass = pMass*tMass/(pMass+tMass)
        de = 0.0001
        dSdE = 0.0
        iList = [0]
        if shifty: iList = [-1,0,1]
        for i in iList:
            e = e_ch + i*de

            if redmass > 0:
                k = (fmscal * redmass * abs(e))**0.5
            else: # photon
                k = abs(e)/hbc
            rho = k * prmax
            if pZ*tZ != 0:
                eta  =  etacns * pZ*tZ * (redmass/abs(e))**0.5
            else:
                eta = 0.0
            if abs(e)<1e-10 and abs(eta)<1e-10: 
                return(0.0, 0.0, 0.0, 0.0)
            rho_a = numpy.array([rho])
            eta_a = numpy.array([eta])
            # print('L,rho,eta =',lch,rho,eta)
    
            P = coulombPenetrationFactor(lch,rho_a,eta_a)[0]
            if e > 0.0:
                S = coulombShiftFactor(lch, rho_a, eta_a)[0]
                whit = 0.0
            else:
                W,WD,ie = Whittaker(lch,rho,eta)
                whit = W[lch]
                S = rho*WD[lch]/whit
                P = 0.
            if i==0:
                penetrability = P
                shift = S
            else:
                dSdE += i*S/(de*2)

    return (penetrability,shift,dSdE,whit)

def getCoulombp_PSdS(e_ch,lch, prmax, redmass,ZZ, shifty):

    fmscal = 0.0478450
    etacns = 0.1574855
    #print "getCoul:",e_ch,lch, prmax, redmass,ZZ

    if redmass<1e-5:  # gamma
        penetrability = 1.0
        shift = 0.0
        dSdE = 0.0
    else:           # massive particles
        de = 0.0001
        dSdE = 0.0
        iList = [0]
        if shifty: iList = [-1,0,1]
        for i in iList:
            e = e_ch + i*de

            k = (fmscal * redmass * abs(e))**0.5
            rho = k * prmax
            eta  =  etacns * ZZ * (redmass/abs(e))**0.5
            rho_a = numpy.array([rho])
            eta_a = numpy.array([eta])
            #print 'L,rho,eta =',lch,rho,eta

            P = coulombPenetrationFactor(lch,rho_a,eta_a)[0] # by convention use abs(e) if e<0
            if e > 0.0:
                S = coulombShiftFactor(lch, rho_a, eta_a)[0]
            else:
                W,WD,ie = Whittaker(lch,rho,eta)
                S = rho*WD[lch]/W[lch]
            if i==0:
                penetrability = P
                shift = S
            else:
                dSdE += i*S/(de*2)
    #print "getCoul:",e_ch,lch, prmax, redmass,ZZ,' rho,eta:',rho,eta,'; P,S =',P,S

    return (penetrability,shift,dSdE)


