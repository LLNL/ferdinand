#

##############################################
#                                            #
#    Ferdinand 0.41, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################
# Barker Transformation for ferdinand.py

import numpy
from scipy.linalg import eigh
from numpy.matlib import zeros
from getCoulomb import *

def BarkerTransformation ( gamS, ES, Bc, lchv, colsfrom,pair_of_col, Qval,prmax,redmass,ZZ,lab2cm, J,pi,debug):

    eps = 1e-8   # convergence of energies (MeV)
    maxiter = 1000
    
#  NCH = number of partial waves
#  NLEV = number of levels
#  Arrays input gamS(NLEV,NCH)    gamma  (standard) (MeV**{1/2})
#               Bc (NLEV)         input boundary conditions to fix gamS (independent of energy L)
#               ES(NLEV)          R-matrix pole energy (MeV)
#               lch,prmax,redmass,ZZ        dictionaries for parameters of shift function
#
#       returns gamB(NLEV,NCH)    gamma-tilde (Brune)
#               EB(NLEV)          R-matrix pole energy-tilde (MeV)
#
    NLEV,NCH = gamS.shape
    if debug: print("\nBarkerT: NLEV,NCH =",NLEV,NCH)
    gamB = zeros([NLEV,NCH])
    EB = numpy.zeros(NLEV)
    if debug: Shift = numpy.zeros(NCH)
    if debug: print('Bc:',Bc)
    
    for ie in range (NLEV):
        if debug:
            ief = open('%sp%s-ie%s.txt' % (float(J),pi,ie),'w')
            iem = open('%sp%s-ie%s-m.txt' % (float(J),pi,ie),'w')
        E = ES[ie] 
        if debug: print('Barker  initial %.6f (cm)' % (E*lab2cm))
        for iter in range(maxiter):
            
            EE = zeros([NLEV,NLEV])
            for l in range(NLEV):
                EE[l,l] = ES[l]
                for c in range(NCH):
                    no = colsfrom[c+1]-1
                    ppo = pair_of_col[no]
                    L = int(lchv[c])
                    Ecm = E * lab2cm + Qval[ppo]
                    P,S,derS = getCoulombp_PSdS(Ecm, L, prmax[ppo], redmass[ppo],ZZ[ppo], False)
                    
                    for k in range(NLEV):
                        EE[k,l] -= gamS[l,c]*gamS[k,c] * (S - Bc[c])
                    if debug: Shift[c] = S
            if debug: print(iter,'\n',EE,file=iem)

            eigval,evec = eigh(EE)
            #   options scipy.linalg.eigh(N, M, lower=True, eigvals_only=False, overwrite_a=False, overwrite_b=False, turbo=True, eigvals=None, type=1, check_finite=True)
            err = abs(eigval[ie]-E)
            if iter<maxiter//4: 
                E = eigval[ie]               # to start with
            elif iter<maxiter//3:
                E = (E + eigval[ie])*0.5     # getting desperate
            elif iter<maxiter//2:
                E = (3*E + eigval[ie])*0.25  # getting more desperate
            else:
                E = (9*E + eigval[ie])*0.1   # getting most desperate
            if debug: print(iter,' '.join([str(e*4/3) for e in eigval[:]]),E*4/3,file=ief)
            if debug: print('Barker #',ie,', %i eval %.6f from %.6f (cm), change %.2e' % (iter,E*lab2cm,eigval[ie]*lab2cm,err))
            if err < eps and iter>2: break

        p = '+' if int(pi)>0 else '-'
        if err> eps*10: 
             print("\n*** Barker Transform Failed in J,pi",J,p," from E=%.5f" % ES[ie])
             print("   Last E,delta-E =",E,err,'after',maxiter,'iterations.')
             #if not debug: raise SystemExit
        if debug: print('\nShift:',Shift)
        if debug: print('Barker #%i, eval %.5f from %.5f after %i' % (ie,E,ES[ie],iter))
        # print("Barker convergence in J,pi=%.1f%c from %9.3f to %9.3f took %4i iterations (cm: %9.3f to %9.3f)" % (J,p,ES[ie],E,iter,ES[ie]*lab2cm,E*lab2cm))
        print("Barker convergence in J,pi=%.1f%c from %9.3f to %9.3f took %4i iterations (cm: %12.6f to %12.6f: shift %9.2e = %9.2e rel)" % (J,p,ES[ie],E,iter,ES[ie]*lab2cm,E*lab2cm,(E-ES[ie])*lab2cm,abs(E-ES[ie])/abs(E)) )
        EB[ie] = E

# Transform gamS to gamB

        for k in range(NLEV):
            for c in range(NCH):
                gamB[ie,c] += evec[k,ie] * gamS[k,c]

    if debug:
        print("ES:\n",ES)
        print("EB:\n",EB)
        print("gamS:\n",gamS)
        print("gamB:\n",gamB)
        print('\n')

    return gamB,EB
