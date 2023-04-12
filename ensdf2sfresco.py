#! /usr/bin/env python3

##############################################
#                                            #
#    Ferdinand 0.50, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################

import sys, os
from pqu import PQU as PQUModule
from brownies.legacy.converting import endf_endl
from PoPs.chemicalElements.misc import *
import masses
import numpy
from fudge.processing.resonances.getCoulombWavefunctions import *
from getCoulomb import *


fmscal = 0.0478450
etacns = 0.1574855

infile = sys.argv[1]
proj = sys.argv[2]
radius = float(sys.argv[3])

#   Partition specifications
corespin =  0  # float(sys.argv[3]) # core spin (float)
coreparity = +1 # float(sys.argv[4]) # core parity (int)
icch=1  # just elastic for now
iach=1
pspin = 0.5  # nucleon incident
pparity = 1  # of positive parity
projs = {'p':'H1', 'a':'He4'}

levels = open(infile)
lines = levels.readlines()
A = int(lines[0][0:3])    ### This requires r33 convention in files
element = lines[0][3:5].title()+str(A)
#print " Element, A =",element,A
cnZA, MAT = endf_endl.ZAAndMATFromParticleName( element )
half = A > A//2*2

outfile = infile+'.vars'
vars = open(outfile,'w')

#levels.readline()  #header line
projectile = projs[proj]

pZA, MAT = endf_endl.ZAAndMATFromParticleName( projectile )
pMass = masses.getMassFromZA( pZA)
pZ = pZA//1000

tZA = cnZA - pZA
tZ  = tZA//1000
tName = idFromZAndA(tZ,tZA % 1000)
tMass = masses.getMassFromZA( tZA)

print("Target ",tZA," of ",tName)
lab2cm = (tMass+pMass)/tMass
redmass = pMass*tMass/(pMass + tMass)

GammaMin=1e-6 # keV
stepp =0.01
stepmin= 0.0

for data in lines:
    if data[5:9]=='  Q ':  
        sn = float(data[21:29])
        sp = float(data[31:39])
        sa = float(data[41:49])
if proj == 'n': Q = float(sn)
if proj == 'p': Q = float(sp)
if proj == 'a': Q = float(sa)
Q = Q*1e-3
print(" CN threshold for",proj,"scattering is ",Q," MeV (cm)")

term = -1
nvars = 0
for data in lines:
    if data[5:9] != '  L ': continue   # Skip all non-L lines
    ex = data[9:18]
    jpi1 = data[21:38]
    life = data[39:48]
    p = -1 if '-' in jpi1 else 1
    #print  ' %10s %10s %10s ' % (ex,jpi,life)
    jpi = jpi1.replace('(','')
    if ',' in jpi: jpi = jpi.split(',')[0]
    if jpi.isspace():      # empty
        j = 0.5 if half else 0  
    elif half:             # Take first of any list of spins
        j2 = jpi.split('/')[0]  
        #print j2
        try: j = float(j2)*0.5
        except: print('Could not translate spin ',j2,len(jpi))
    else:
        j = int(jpi)
    escat =  float(ex)*1e-3-Q
    elab = escat*lab2cm
    term += 1
    namep = 'E:%s' % str(elab)
    pp = '+1' if p == 1 else '-1'
    string = "&Variable kind=3 name='%8s' term=%i jtot=%s par=%s energy=%.6f /         (lab: %.3f) \n" %(namep[:8],term,j,pp,escat,elab)
    vars.write(string)
    nvars += 1
    
    if life.isspace(): 
        Gamma = GammaMin
    else:
        lifel = life.lower().replace('v','V')
        if 'V' in life:
            Gamma = PQUModule.PQU(lifel).getValueAs('MeV')
        else:   # Try lifetime;q:q
            lifetime = PQUModule.PQU(lifel).getValueAs('s')
            Gamma = 0.693*4.235e-21 / lifetime

    j = float(j)
    p = int(pp)
    smin = abs(corespin-pspin)
    smax = corespin+pspin
    s2min = int(2*smin+0.5)
    s2max = int(2*smax+0.5)
    #print "For J,pi =",j,p,' smin,smax =',smin,smax
    minL = 1000
    for s2 in range(s2min,s2max+1):
        s = s2*0.5
        lmin = int(abs(s-j) +0.5)
        lmax = int(s+j +0.5)
        for l in range(lmin,lmax+1):
            if p != pparity*coreparity*(-1)**l: continue
            minL = min(minL,l)
            if minL == l: minS2 = s2

#   print "     life of ",life," at",elab," gives width ",Gamma, " MeV"
    print("     life of %s at %.3f MeV(lab) gives width %.2e keV for Jpi=%.1f%s using %s" % (life[:10],elab,Gamma*1000.,j,pp[0],jpi1))

    lch = minL

    penetrability,shift,dSdE,W = getCoulomb_PSdSW(
        escat,lch, radius, pMass,tMass,pZ,tZ, fmscal,etacns, True)
    shifty_sum = Gamma *  dSdE / penetrability
    shifty_denom = 1.0/(1.0 - 0.5 * shifty_sum)
    Gammaf = Gamma * shifty_denom

#     k = (fmscal * redmass * abs(escat))**0.5
#     rho = k * radius
#     eta  =  etacns * pZ*tZ * (redmass/abs(escat))**0.5
#     rho_a = numpy.array([rho])
#     eta_a = numpy.array([eta])
#     #print 'L,rho,eta =',lch,rho,eta
#     P = coulombPenetrationFactor(lch,rho_a,eta_a)[0]    
#     print "L,rho,eta ",lch,rho,eta," gives P=",penetrability,'or ',P,P/penetrability
    rwaMain = (abs(Gammaf)/(2.*penetrability))**0.5  #  MeV**{0.5}
    if Gammaf < 0: 
        print("    At ",escat,lch,"G",Gamma," gives Gf",Gammaf," from sum",shifty_sum)


    npole = 0
    for s2 in range(s2min,s2max+1):
        s = s2*0.5
        lmin = int(abs(s-j) +0.5)
        lmax = int(s+j +0.5)
        #print "For J,pi =",j,p,'S =',s,' so lmin,max =',lmin,lmax
        for l in range(lmin,lmax+1):
            #print "Try L=",l,' if ',p,' = ',pparity*coreparity*(-1)**l
            if p != pparity*coreparity*(-1)**l: continue
            rwa = 0.0
            if l==minL and s2==minS2:  rwa = rwaMain

            step = max(rwa*stepp,stepmin)
            npole+=1
            namew = 'w%i:%s' % (npole,str(elab))
            string ="&Variable kind=4 name='%8s' term=%i icch=%i iach=%i lch=%i sch=%.1f width=%.3e step=%.2e / (G:%.3f keV) \n" % (
                namew[:8],term,icch,iach,l,s,rwa,step,Gamma*1e3)
            vars.write(string)
            nvars += 1
    vars.write('\n')

print(" Written file ",outfile," with ",nvars," search variables for R-matrix radius ",radius)
