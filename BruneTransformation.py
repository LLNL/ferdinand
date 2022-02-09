#

##############################################
#                                            #
#    Ferdinand 0.40, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################
# Brune Transformation for ferdinand.py

import numpy
from scipy.linalg import eigh
from numpy.matlib import zeros #,matmul

def BruneTransformation ( gamB, EB, Bc, Shift, debug,J,pi,lab2cm):
#  NCH = number of partial waves
#  NLEV = number of levels
#  Arrays input gamB(NLEV,NCH)    gamma-tilde  (Brune)
#               Bc (NLEV)         boundary conditions to fix output gamS (independent of energy L)
#               Shift (NLEV,NCH)  S_m(E_l)
#               EB(NLEV)          R-matrix pole energy-tilde 
#
#       returns gamS(NLEV,NCH)    gamma (standard)
#               ES(NLEV)          R-matrix pole energy (standard)
#
# All energies and widths are on the scale of the projectile lab frame
#        
#  From C.R. Brune, Phys. Rev. C66, 044611 (2002), equations (20, 21, 24, 27).
#  
    NLEV,NCH = gamB.shape
    if debug: print("BT: NLEV,NCH =",NLEV,NCH)
    if NLEV*NCH==0: return gamB,EB

    M = zeros([NLEV,NLEV])
    if debug: print('Bc:',Bc)
    if debug: print('Shift:\n',Shift)
    N = M.copy()
    for l in range(NLEV):
        M[l,l] = 1
        N[l,l] = EB[l]
        for c in range(NCH):
            N[l,l] += gamB[l,c]**2 * (Shift[l,c] - Bc[c])

        for k in range(l+1,NLEV):
           for c in range(NCH):
               M[k,l] -= gamB[l,c]*gamB[k,c] * (Shift[l,c]-Shift[k,c])/(EB[l]-EB[k])
               N[k,l] += gamB[l,c]*gamB[k,c] * ( (EB[l]*Shift[k,c] - EB[k]*Shift[l,c]) /(EB[l]-EB[k]) - Bc[c])
           M[l,k] = M[k,l]
           N[l,k] = N[k,l]

    if debug: print('N matrix:\n',N)
    if debug: print('M norm matrix:\n',M)
    try:
        ES,vec = eigh(N, M, lower=True)
    except:
        print("\nERROR in eigh for J/pi=",J,'+' if int(pi)>0 else '-')
        if debug: print('N matrix:\n',N)
        if debug: print('M norm matrix:\n',M)
        ES,vec = eigh(M, lower=True)
        print("\n     Norm eigenvalues:\n",'    ',ES)
        if debug: print("Norm eigenvectors:\n",vec)
        print("\n     FAILED in BruneTransform for",J,'+' if int(pi)>0 else '-',"\n")
        ES = [9.1111 for i in range(NLEV)]
        
# Transform gamB to gamS
    gamS = zeros([NLEV,NCH])
    for c in range(NCH):
        for l in range(NLEV):
            sum = 0
            for k in range(NLEV): sum += vec[k,l] * gamB[k,c]
            gamS[l,c] = sum

    if debug: 
        print("EB:\n",EB)
        print("ES:\n",ES)
        print("gamB:\n",gamB)
        print("gamS:\n",gamS)
        print('\n')

    for l in range(NLEV):
         print("Brune transformation in J,pi=%.1f%c from %9.3f to %9.3f (cm: %9.3f to %9.3f)" % (J,'+' if int(pi)>0 else '-',EB[l],ES[l],EB[l]*lab2cm,ES[l]*lab2cm))

    return gamS,ES
