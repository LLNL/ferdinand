#! /usr/bin/env python3


# Read  AMUR file and store as python structure
#

import os,sys
import re as reModule
import masses
amu    = 931.494013


def getAMUR(lines,verbose): 

    chref = 0  # Reference partitioon
    dataSets = []
    nuclei = {}
    partitions = []
    channelList = []
    widthList = []
    nl = len(lines)
    term = 0

    for l in range(nl):
        c = lines[l]
        if '<AMUR>' in c: 
            title = c.split(':')[1].strip(' \n')

        header = '<Analysis>' in c or '<Experimental>' in c
        section= "<Section>" in c

        if header or section:
            dataSet = c.split(':')[1].strip(' \n')
            props = {}
            group = None; hasNorm = False; 
            for ll in range(l+1,nl):
                if '=' not in lines[ll]: break
                c = lines[ll].strip()
                var,val =  c.split('= ')[:2]
                var = var.strip()
                value = val.split()[0]
                if var in ['npoints','Lmax']:       value = int(value)
                if var in ['Chisq','normalization','angle','elab']: value = float(value)
                if var in ['elab']: 
                    value /= 1e6
                    if 'Spiger' in dataSet: 
                        value = float("%.4f" % value)
                    else:
                        value = float("%.5f" % value)
                    tag = 'e'
                    group = value
                if var in ['angle']:  
                    tag = 'a'
                    group = value
                if var in ['normalization']:  
                    hasNorm = True
                props[var] = value

            if not hasNorm:  # look for group Norm
                print("Is",groupName,"in",dataSet,"?")
                if groupName in dataSet:
                    normalization =  groupNorm
                    props['normalization'] = normalization
                    print("Yes! Get norm",groupNorm)
                    hasNorm = True
                
            if hasNorm:
                nameparts = dataSet.split('_')
                author = nameparts[0][:-4]
                year = nameparts[0][-4:]
                # print author,year,nameparts
                if len(nameparts)>1:
                    FrescoName = author + '_' + nameparts[1] + '@' + tag + "%s.data" % group
                    props['fresco'] = FrescoName
                else:
                    groupNorm = props['normalization']
                    groupName = dataSet
                    print("Set norm for",groupName,"as",groupNorm)
                    # FrescoName = author + '_' +    "%%"     + '@' + "*"
                # print nameparts,"::",FrescoName
            else:
                FrescoName = None

            dataSets.append([dataSet,props])

        boundaries = {}
        if '<Boundary>' in c:
            l += 1        
            for np in range(1000):
                lp = l+np
                c = lines[lp]
                if c[0:2] != 'Rc': break
                #print reModule.split('[,:;\(\)=]',c)
                lab,zap,jpip,zat,jpit,L,X,Y,Z,RMrad = reModule.split('[,:;\(\)=]',c)
                zap,zat,RMrad = int(zap),int(zat),float(RMrad)
                
                pi = +1 if '+' in jpip else -1
                spin = float(jpip[:-1])
                massp =  masses.getMassFromZA( zap ) 
                nuclei[zap] = spin,pi,massp

                pi = +1 if '+' in jpit else -1
                spin = float(jpit[:-1])
                masst =  masses.getMassFromZA( zat ) 
                nuclei[zat] = spin,pi,masst

                mass =  masses.getMassFromZA( zap+zat )
                nuclei[zap+zat] = 0,0,mass
                Qref = -mass*amu
                Q = -Qref -(massp + masst)*amu
                partitions.append([zap,jpip,zat,jpit,RMrad,Q])

            nParts = lp-l
            if verbose: print("nParts:",nParts)
            l = lp
            nBCO = 0
            for ch in range(1000):
                c = lines[l+ch]
                if c[0:2] != 'Bc': break
                lab,zap,jpip,zat,jpit,L,X,Y,Z,B = reModule.split('[,:;\(\)=]',c)
                zap,zat,L,B = int(zap),int(zat),int(L),float(B)
                #print zap,zat,'for L=',L,'B=',B
                if abs (B+L)>1e-5:  
                     nBCO += 1
                     boundaries['%i,%i,%i' % zap,zat,L] = B

        csh = 0
        if '<Reduced width amplitude' in c:
            l += 1
            jpi = -1
            pole = None
            while lines[l][0:3]=='...':
                c = lines[l]
                J,pi = c[10:14],c[14:15]
                J = float(J)
                parity = +1 if pi=='+' else -1
                if verbose: print("\nJ,pi =",J,parity)
                channel = [[J,parity]]
                poles = [[J,parity]]
                l += 1
                nch = -1
                if l==nl: break
                csh =  lines[l].find('E=')
                if verbose and csh >0: print("Find 'E=' at",csh,"in",lines[l])
                for ll in range(l,nl):
                    if lines[ll][:5]==csh*' ':  lines[ll] = lines[ll][5:]
                    if lines[ll][1:5]=='->  ':  lines[ll] = lines[ll][5:]
                #print lines[l][26:31] 
                while lines[l][26:32] == 'gamma(':
                    pl = lines[l][32:].replace(')=',' ').replace(',',' ').replace(';',' ').replace(':',' ')
                    vals = pl.split()
                    zap,jpip,zat,jpit,L,S,JT,W = vals[0:8]
                    zap,zat,L,S,W = int(zap),int(zat),int(L),float(S),float(W)
                    for np in range(nParts):
                        #print "Compare ", [zap,jpip,zat,jpit],"to", partitions[np][0:4]," get:",[zap,jpip,zat,jpit]==partitions[np][0:4]
                        if [zap,jpip,zat,jpit] == partitions[np][0:4]:   # find partition
                            ip = np
                    if lines[l][0:2] =='E=':     # new pole
                        if pole is not None: poles += [pole]  # dump previous pole info
                        energy = float(lines[l][2:12])
                        pole = [energy*1e-6]
   
                    l += 1
                    ich = -1
                    for jch in range(len(channel[1:])):
                        if [ip,L,S] == (channel[jch+1]):
                            ich = jch
                            break
                    if ich < 0: # not found
                        channel += [[ip,L,S]] 
                        nch += 1
                        ich = nch
                    # add to existing pole
                    pole += [[ich+1,W*1e-3]]   # were in units eV^1/2. Now MeV^1/2
                poles += [pole]
                if verbose: print('poles:',poles)
                pole = None
                #print 'leaving pole:',pole
                channelList += [channel]
                widthList += [poles]

    if verbose: print(widthList)
    
    if verbose: 
        print("\n################ Summary ################")
        print("\nTitle:",title)
        print("\nDataSets:")
        for d in dataSets:
            print(d)
        print("\nParitions",nParts,":",partitions)
        print(len(list(nuclei.keys())),"nuclei: (spin,parity,mass) \n",nuclei)
        if nBCO==0: 
            print('All boundaries B=-L')
        else:
            print("Boundaries B!=-L :",boundaries)
        print("\nChannelList:\n",channelList)
        for jpiset in channelList:
            J,pi = jpiset[0]
            print('J,pi:',J,pi) 
            ich = 0
            for chan in jpiset[1:]:
                ich += 1
                print("     c= %3i" % ich,chan)

        print("\nR-matrix amplitudes:\n", end=' ')
        for jpiset in widthList:
            J,pi = jpiset[0]
            print('J,pi:',J,pi) 
            for poles in jpiset[1:]:
                #print poles
                if poles is not None:
                    print("     Pole E=%13.6f " % poles[0])
                    for width in poles[1:]:
                        print("          c=%3i width=%10.6f" % (width[0],width[1]))

    #print channelList

    return  title, partitions, boundaries, widthList, channelList, dataSets, nuclei
  
    
def putAMUR (title, partitions, boundaries, rmatr, normalizations, nuclei, verbose):
    lines = []
    return lines

if __name__=="__main__":

    amurIn = sys.argv[1]
    lines = open(amurIn).readlines()
    print(" Read in ",len(lines)," lines from file",amurIn)
    DIR = 'Expt5/'
    translate = {'Spiger_aa':'Spiger-A1094004-lab_aa',  'Spiger_ap0':'Spiger-cm_ap0' }
 
    amurSpec = getAMUR(lines, True)
    title, partitions, boundaries, widths, channelList, normalizations, nuclei  = amurSpec
    csv = open(amurIn + '-norms.csv','w')
    print('dataset,norm,shift', file=csv)
    print('str,foat,float', file=csv)
    print(',,MeV', file=csv)
    for d in normalizations:
        props = d[1]
        # print props
        shift = 0.0
        fresco = props.get('fresco',None)
        if fresco is not None: 
            f1,f2 = fresco.split('@')
            if translate.get(f1) is not None: f1 = translate[f1]
            fresco = f1 + '@' + f2
            try:
                f = open(DIR + fresco,'r')
                # print "File",DIR + fresco," exists"
            except:
                print("File",DIR + fresco," does not exist")
 
            print("%s,%s,%s" % (fresco,1.0/props['normalization'],shift), file=csv)
 
    
    # outlines = putAMUR(title, partitions, boundaries, rmatr, normalizations, nuclei, False)
#     
#     amurOut = amurIn + ".echo"
#     
#     print " Wrote file ",amurOut
