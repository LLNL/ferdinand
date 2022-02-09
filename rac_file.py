#! /usr/bin/env python3


# Read  RAC file and store as python structure
#

import os,sys
import re as reModule
import masses
amu    = 931.494013


def getRAC(lines,wzero,verbose): 

    chref = 0  # Reference partitioon
    dataSets = []
    nuclei = {}
    partitions = []
    channelList = []
    widthList = []
    rmQ = []
    nl = len(lines)
    term = 0

    for l in range(nl):
        c = lines[l]
        #print c,'AY' in c
       
        if 'NC.NE' in c:
            vals = lines[l+1].split()[0:2]
            NC,NE = [int(v) for v in vals]
            print('NC,NE:',NC,NE)
        
        if 'ID.CHANN' in c:
            for p in range(NC):
                rm = float(lines[l+p+1][15:37])
                Q  = float(lines[l+p+1][54:63])
                
                #print p,rm,Q
                rmQ.append([rm,Q])
            #if verbose: print 'rmQ:\n',rmQ

        if 'AY' in c:
            #print lines[l+1]
            for part in range(NC):
                vals = lines[l+part+1].split()
                try:
                    tmass,pmass,zt,zp,jt,jp,pt,pp = [float(v) for v in vals]
                except:
                    break
                pp,pt = int(pp),int(pt)
                zp,zt = int(zp),int(zt)
                ap = int(pmass+0.5); at = int(tmass+0.5)
                zap = zp*1000 + ap; zat = zt*1000 + at
                nuclei[zap] = jp,pp,pmass
                nuclei[zat] = jt,pt,tmass
                partitions.append([zap,zat,rmQ[part][0],rmQ[part][1]])
                #partitions[part].append(zap).append(zat)

                #print "pmass,tmass,zp,zt,jp,jt,pp,pt:",pmass,tmass,zp,zt,jp,jt,pp,pt

            if verbose: print('Nuclei:\n',nuclei)

            if verbose: print('Partitions:\n',partitions)


        if 'PAR.LEVE' in c:
            jpi = -1
            nchp = [0 for i in range(NC)]
            for spinSet in range(NE):
                l += 1;  vals = lines[l].split()
                #print vals[0:3+NC]
                pi,J,npoles = vals[0:3]; nchp = vals[3:]
                J = float(J); parity = int(pi); npoles=int(npoles); nchp=[int(v) for v in nchp]
                print("\nJ,pi =",J,'+' if parity>0 else '-')
                channel = [[J,parity]]
                poles = [[J,parity]]

                for ipole in range(npoles):
                    l += 1;  vals = lines[l].split()
                    energy = float(vals[0])
                    #print "Vals:",vals
                    i = 4 if vals[1]=="'" else 3
                    damp = float(vals[i])
                    print("    E: %11.6f , %10.6f*i" % (energy,damp))
                    pole = [[energy,damp]]
            
                    ch = 0
                    for part in range(NC):
                        for ich in range(nchp[part]):
                            l += 1;  vals = lines[l].split()
                            ip,S,L,w = vals[0:4]
                            ip,S,L,w = int(ip),float(S),int(L),float(w)
                            if wzero and abs(w)<1e-10: continue
                            #print "ip,S,L,w:",part+1,S,L,w
                            if ipole==0: channel += [[part+1,S,L]]   # later q.nos are the same as first
                            ch += 1
                            pole += [[ch,w]]   # were in units MeV^1/2
                    poles += [pole]
                #print 'poles:',poles
                #print 'leaving pole:',pole
                channelList += [channel]
                widthList += [poles]
    if verbose:
        print(channelList)
        print(widthList)
    
    if verbose: 
        print("\n################ Summary ################")
        print("\nDataSets:")
        for d in dataSets:
            print(d)
        print("\nParitions",NC,":",partitions)
        print(len(list(nuclei.keys())),"nuclei: (spin,parity,mass) \n",nuclei)
        print('All boundaries B=-L')
 
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
                print("     Pole E=%13.6f " % poles[0])
                for width in poles[1:]:
                    print("          c=%3i width=%10.6f" % (width[0],width[1]))

    #print channelList

    return  partitions, widthList, channelList, dataSets, nuclei
  
    
def putRAC (partitions, widthList, channelList, dataSets, nuclei, verbose):
    lines = []
    return lines

if __name__=="__main__":

    amurIn = sys.argv[1]
    lines = open(amurIn).readlines()
    print(" Read in ",len(lines)," lines from file",amurIn)
 
    amurSpec = getRAC(lines, False, True)
    partitions, widthList, channelList, dataSets, nuclei = amurSpec
    
    # outlines = putRAC(partitions, widthList, channelList, dataSets, nuclei, False)
#     
#     amurOut = amurIn + ".echo"
#     
#     print " Wrote file ",amurOut
