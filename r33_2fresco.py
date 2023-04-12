#! /usr/bin/env python3

##############################################
#                                            #
#    Ferdinand 0.50, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################
import sys
from brownies.legacy.endl.endl_Z import endl_ZSymbol
from PoPs.chemicalElements.misc import *

__doc__="""Converts r33 nuclear data files to simpler files for SFresco searching.
Makes a summary of file names and parameters of these files for input to SFresco.

usage:
    %s file1.r33 file2.r33 ... *.r33"""


# cross section errors cannot be below these:
dyRelmin = 0.03 # relative
noise = 0.5     # absolute

includeGammas = False
relativeData = False
verbose = True
vverbose = True
vverbose = False

r33File = sys.argv[1] # first file name
SFrescoName = r33File[0:4]+'_sfresco.search'
SFrescoFile = open(SFrescoName,'w')
SFrescoList = []

datasets = 0
foundLevels = {}
residuals = set()
maxlevel = {}
for r33File in sys.argv[1:]:

    x = r33File[5]
    try: ex = int(x)
    except: ex=x
    print("\nReading data",r33File) #, ' to ',ex
    datasets += 1
    r33 = open(r33File,'r').read().splitlines()
    inData = False
    needSetup = True
    rr = False
    for line in r33:
        #if vverbose: print line,inData
        if not inData:
            if 'Masses:' in line:
                pm,tm,em,rm = line[8:].split(',')
                #print 'pm,tm,em,rm  =',pm,tm,em,rm 
                pm=float(pm); tm=float(tm)
                em=float(em); rm=float(rm)
                pA=int(pm+0.5); tA=int(tm+0.5)
                eA=int(em+0.5); rA=int(rm+0.5)
            elif 'Zeds:' in line:
                pz,tz,ez,rz = line[6:].split(',')
                #print 'pz,tz,ez,rz  =',pz,tz,ez,rz 
            elif 'Theta:' in line:
                angle = float(line[7:])
                ia = int(angle+0.5)
                if(abs(angle-ia)<1e-5): angle = ia
            elif 'Enfactors:' in line:
                if vverbose: print('Enfactors:'+line[10:])
                escale = line[10:].split(',')
            elif 'Units:' in line:
                units = line[7:]
                rr = 'rr' in units
                if 'mb' not in units and not rr and not relativeData:
                    print("   Relative cross section data not (yet) converted")
                    break
            elif 'Sigfactors:' in line:
                xscales = line[13:].split(',')
            elif 'Qvalue:' in line:
                qValues = line[8:].split(',')
                Q = float(qValues[0])*1e-3 # MeV from eV
            elif 'Data:' in line and not 'Evaluated' in line:
                inData = True
            elif 'EndData:' in line:
                inData = False
            elif 'Gamma energy' in line and not includeGammas:
                print("   Gamma file not converted")
                break
            else:
                continue # some other keywrod 
        else:  # got some data!
            if 'EndData:' in line:
                inData = False
                break
            if needSetup:
                if vverbose: 
                    print('pm,tm,em,rm  =',pm,tm,em,rm) 
                    print('pz,tz,ez,rz  =',pz,tz,ez,rz ," : ",p,t+str(tA))
                    print('pA,tA,eA,rA  =',pA,tA,eA,rA)
                    print('Theta  =',angle)
                    print('Units  =',units)
                    print('Enfactors  =',escale)
                    print('Sigfactors  =',xscales)
                    print('Qvalues  =',xscales)
                p = endl_ZSymbol(int(pz));  t = endl_ZSymbol(int(tz))
                e = endl_ZSymbol(int(ez));  r = endl_ZSymbol(int(rz))
                pn = '%s%i' % (p,pA)
                tn = '%s%i_e0' % (t,tA)
                en = '%s%i' % (e,eA)
                rn = '%s%i' % (r,rA)
                rnex = '%s%i_e%i' % (r,rA,ex)
                inChannel = '%s + %s' % (pn,tn)
                outChannel = '%s + %s' % (en,rnex)
                if outChannel == inChannel: outChannel = 'el'
                reaction = '%s -> %s %f' % (inChannel,outChannel,Q)
                foundLevels[rnex] = Q
                residuals.add(rn)
                if rn in maxlevel:
                    maxlevel[rn] = max(maxlevel[rn],ex)
                else:
                    maxlevel[rn] = 0
                print('      ',reaction)

                datFileName = r33File+'-a'+str(angle)+'.dat'
                datFile = open(datFileName,'w')
                needSetup = False
            else:
                line2 = line.replace(',',' ')
                data = line2.split()
                if len(data) == 3:
                    x,dx,y = data
                    dy = max(y*dyRelmin,noise)
                else:
                    try:
                        x,dx,y,dy = data
                    except:
                        print(" Error: line<"+line2+'> gives:',data)
                        break
                    dy = max(float(dy),float(y)*dyRelmin)
                    if not rr: dy = max(dy,noise)
                xeV = float(x)*float(escale[0])
                xMeV = xeV*1e-3
                ymb = y
                dymb = dy
    #           y is already in mb
                print(xMeV,ymb,dymb, file=datFile)
    if not needSetup: # already set up
        dtype = 2
        iscale = 2
        idir = 0
        if rr: idir=1
        lab = 'T'
        abserr = 'T'
        ic = 1 if en == pn else 2   # TEMPORARY (revise if more than 2 kinds of ejectiles. Exact order depends on Fresco input file)
        ia = ex+1  
        dataSpec = ' &data type=%i ic=%i ia=%i angle=%d data_file="%s" iscale=%i idir=%i lab=%s abserr=%s /    # %s' % (
             dtype,ic,ia,angle,datFileName,iscale,idir,lab,abserr,outChannel)
        print(dataSpec, file=SFrescoFile)
        SFrescoList += [dataSpec]

print("\nDatasets converted: ",datasets,' in search file',SFrescoName,":")
for s in SFrescoList: print(s)
# for lex in foundLevels.keys(): 
#     lgs,ex = lex.split('_e')
#     lgs = lgs+'_e0'
#     qgs = foundLevels[lgs]
#     print lex,foundLevels[lex],', E*:',qgs-foundLevels[lex]
print("\nExcited states:  level, Q, excitation")
for rn in residuals:
    for ex in range(maxlevel[rn]+1):
        lex = rn+'_e%i'%ex
        if ex==0: qgs = foundLevels[lex]  # Q-value to gs of this residual nuclide
        print(lex,foundLevels[lex],', E*:',qgs-foundLevels[lex])
