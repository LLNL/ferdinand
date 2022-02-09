#! /usr/bin/env python3


# Read EDA par file and store as python structure
#

import os,sys,numpy
amu    = 931.494013


def getEDA(lines,covlines,verbose): 

    print('covlines is:', covlines)
    chref = 0  # Reference partitioon
    isomax = 3.0  # all isospins must have abs values less than this, to be isospins
     
    nl = len(lines)
     
    partitions = []
    l = 0
    for l in range(nl):
        c = lines[l]
        if "'" not in c: break
        p = [c[2:7].replace("'","").strip(), c[10:15].replace("'","").strip(),int(c[17:21]),int(c[21:26])]
        partitions.append(p)
    l0 = l
    if verbose: print(l,"partitions\n",partitions)
    comment = lines[l0]
    l0 += 1

    boundaries = []
    radii = set()
    for l in range(l0,nl):
        c = lines[l]
        try: r = float(c[:15])
        except: break
        rf = c[15]=='f'
        bc = float(c[18:35])
        bcf = c[35]=='f'
        a  = float(c[36:52])
        radii.add(r)
        boundaries.append([r,rf,bc,bcf,a])
        #print r,rf,bc,bcf,a
    if verbose: print(len(boundaries),"boundaries\n",boundaries)
    if verbose: print("Distinct radii:",radii)
    l0 = l

    rmatr = []
    pVarying = []
    ipar = 0
    for jpi in range(1000):
        if lines[l]=='    0': break
        try:     nle,rpower,a,b,c = [int(x) for x in lines[l].split()[:5]]
        except: break
        try:     Jpi = lines[l].split()[5]
        except: Jpi = ''
        #print nle,rpower,a,b,c 
        if verbose: print("\n########### jpi set ",jpi," with ",nle," levels",Jpi," ###########")
        iso = []
        try:
            iso = [float(x) for x in lines[l+1].split()[:nle]]
            if max(iso)<=isomax:
                l += 1
                #print iso
            else:
                iso = []
        except:
            pass
        Jpars = [iso,Jpi]
        for chan in range(1000):
            nels = 4
            nc = (nle-1)//4 + 1
            ie = 0
            try: x = float(lines[l+1][0:15]) 
            except: break
            if lines[l+1][:5]=='    0': break
            pars = []
            parf = []
            for ic in range(nc):
                l += 1; #print "line ",l
                c = lines[l]

                if ic==nc-1: nels = nle-nels*(nc-1)
                for iel in range(nels):
                    col = iel*18
                    re = float(c[col:col+15])
                    ref = c[col+15]=='f'
                    ie += 1
                    ipar += 1
                    if not ref: pVarying.append([ipar,jpi,chan,re])
                    pars.append(re)
                    parf.append(ref)
                    #print jpi,chan,ie,re,ref
            Jpars.append([pars,parf])
        if verbose: print(" pars:\n",Jpars)
        rmatr.append(Jpars)
        l += 1
        #print 'Next:',l,lines[l]
    if verbose: print('\nR-matrix parameters:\n',rmatr)
    if verbose: print('\nVariable parameters: %s\n' % len(pVarying),pVarying)
    l0=l+1

    normalizations = []
    nnorms = int(lines[l0][:6])
    if verbose: print(nnorms,"normalization variables")
    for n in range(nnorms):
        l = l0+n+1
        c = lines[l]
        #print c
        tag = c[2:10]
        norm = float(c[11:25])
        fixed = c[25]=='f'
        rest = [float(x) for x in c[26:].split()]
        normalizations.append([tag,norm,fixed,rest[0],rest[1]])
    l = l0+nnorms
    if verbose: print(normalizations)

    l0 = l+1
    nconstraints = int(lines[l0].split()[3])
    print("Skip",nconstraints,"constraints")
    l0 = l0 + nconstraints

    l0 = l0+1
    npartitions = int(lines[l0][:6])
    print("# Partitions:",npartitions)
    l0 += 1
    lmax = []
    nuclei = {}
    for np in range(npartitions):
        l = l0 + np*3 
        c = lines[l]
        p,t,lm,e = [c[col*5:col*5+5] for col in range(4)]
        lmax.append(int(lm))
        parity,spin,charge,mass,isospin = [float(x) for x in lines[l+1].split()]
        # print('np:p,J,Z,m,T::',np,parity,spin,charge,mass,isospin)
        #nuclei.append([parity,spin,charge,mass,isospin])
        nuclei[p.strip()] = int(parity),spin,charge,mass,isospin
        parity,spin,charge,mass,isospin = [float(x) for x in lines[l+2].split()]
        #nuclei.append([parity,spin,charge,mass,isospin])
        nuclei[t.strip()] = int(parity),spin,charge,mass,isospin

    if verbose: print(len(list(nuclei.keys())),"nuclei: ",nuclei)
    if verbose: print(len(lmax),"lmax: ",lmax)
    
    
    JpiList = []
    pref = partitions[chref]
    Lmax_ref = pref[2]
    if verbose: print('Reference partition:',pref, pref[0], pref[1])
    
#     for part in partitions:
    if True:
        part = pref 
        
        print(part)
        print('Partition:',part, nuclei[part[0]], nuclei[part[1]], nuclei[part[0]][0], nuclei[part[1]][0])
        for Lref in range(Lmax_ref+1):
            pi = (-1)**Lref * nuclei[part[0]][0] * nuclei[part[1]][0]
            Smax = nuclei[part[0]][1] + nuclei[part[1]][1]
            # print('\nL, pi, Smax =',Lref,pi,Smax)
            for ism in range(int(Smax)+1):
                sch = Smax - ism
                JTmax2 = int(2*(Lref+sch)+0.5)
                JTmin2 = int(2*abs(Lref-sch)+0.5)
                # print('sch: ',sch, 'JTmin, JTmax:',JTmin2*0.5,JTmax2*0.5)
                for JT2 in range(JTmax2,JTmin2-1,-2):
                    JT = JT2*0.5
                    Jpi = JT,pi
                    if Jpi not in JpiList: JpiList.append(Jpi)

    if verbose: print("   List of Jpi",JpiList)
    NJS = len(JpiList)

    spinGroups = []
    partialWave = 0  # index of LANL channel order
    channelList = []
    Radii = {}
   
    for spinGroupIndex in range(NJS):
        JJ,pi = JpiList [ spinGroupIndex ]
        if verbose: print('\n ##### Spinset #',spinGroupIndex,': J,pi =',JJ,pi)
        pw1 = partialWave+1
        channels = [[JJ,pi]]
        for npart in range(npartitions):
            p,t = partitions[npart][0:2]
            pt,jp,chargep,massp,isospinp = nuclei[p]
            tt,jt,charget,masst,isospint = nuclei[t]
            Lmax = partitions[npart][2]
            smin = abs(jt-jp)
            smax = jt+jp
            s2min = int(2*smin+0.5)
            s2max = int(2*smax+0.5)
            for s2 in range(s2max,s2min-1,-2):
                sch = s2*0.5
                lmin = int(abs(sch-JJ)+0.5)
                if massp<1e-10 and jp<0.01: lmin=max(lmin,1)    # no E0: only E1, E2
                lmax = min(int(sch+JJ +0.5),Lmax)
                for lch in range(lmin,lmax+1):
                    if pi != pt*tt*(-1)**lch: continue
                    partialWave += 1

                    BV = boundaries[partialWave-1][2]
                    b = BV
                    if abs(BV+lch)<1e-5 : BV = None

                    RM = boundaries[partialWave-1][0]
                    p_t = p+'+'+t
                    if p_t not in list(Radii.keys()): Radii[p_t] = set(); 
                    Radii[p_t].add(RM)
                    
                    channels.append((npart,p,t,lch,sch,BV,partialWave))
                    print(' Channel # %2i ' % partialWave,":: L=",lch,"S=",sch,'B=%s' % str(BV),'R=%s' % str(RM),p,t,JJ,pi,b,abs(b+lch))
                    
        if verbose: print("channels:",channels)
        
        channelList.append(channels)
     
    if verbose: print('    channelList =',channelList)

    if verbose: print("\n Read in ",partialWave," EDA partial waves")
    if verbose: print("\n R-matrix radii: ",Radii)
    if partialWave != len(boundaries):
        print('\n ERROR:  %5i channels enumerated, but boundaries given for %5i  !!\n\n' % (partialWave,len(boundaries)))

    
    p,t = pref[0:2]
    print('Reference p,t=',p,t)
    pt,jp,chargep,massp_ref,isospinp = nuclei[p]
    tt,jt,charget,masst_ref,isospint = nuclei[t]
    Qref = -(massp_ref + masst_ref)*amu
    for npart in range(npartitions):
            p,t = partitions[npart][0:2]
            p_t = p+'+'+t
            if len(Radii[p_t])>1: 
                print("TOO MANY RADII for partition",p_t,':',Radii[p_t])
            partitions[npart].append(list(Radii[p_t])[0])
            pt,jp,chargep,massp,isospinp = nuclei[p]
            tt,jt,charget,masst,isospint = nuclei[t]
            Q = -Qref -(massp + masst)*amu
            partitions[npart].append(Q)
            if verbose: print('Partition',npart,'has Q=',Q)

    if verbose: print(partitions)

    if covlines is not None:
        varying = [int(i) for i in covlines[1].split()]
        nv = len(varying)
        covMatrix = numpy.zeros([nv,nv])
        print('\nCovariance matrix:')
        for i in range(nv):
            line = 2+i
            vals = [float(x) for x in covlines[line].split()[1:]]
            print(i+1,pVarying[i][0],":",vals)
            covMatrix[i,:i+1] = numpy.array(vals)
            if i>0: covMatrix[:i,i] = numpy.array(vals[:-1])
        #print(covMatrix)
        covariances = pVarying,covMatrix
    else:
        covariances = None,None

    return  comment, partitions, boundaries, rmatr, channelList, normalizations, nuclei, covariances
    
    
def putEDA (comment, partitions, boundaries, rmatr, normalizations, nuclei, covariances, verbose):

    if verbose: print("\nOUTPUT:")
    lines = []
    for p in partitions:
        s = " '%-5s' '%-5s'%5i %4i\n" % (p[0][0:5],p[1],p[2],p[3])
        lines.append(s)
        
    lines.append(comment)
    
    for b in boundaries:
        rf=' '; bf=' '
        if b[1]: rf='f'
        if b[3]: bf='f'
        s = "%15.8e%c  %15.8e%c  %15.8e\n" % (b[0],rf,b[2],bf,b[4])
        lines.append(s.replace('e','E'))
    
    for jpi in rmatr:
        if verbose: print("/n######", jpi)
        iso = jpi[0]
        Jpi = jpi[1]
        rpower = 1
        nle = len(jpi[2][1])
        s = " %4i %4i %4i %4i %4i %s  %5s %2i\n" % (nle,rpower,0,0,0,44*' ',Jpi,4)
        lines.append(s)
        if len(iso)>0:
            s = ''
            for t in iso: s += '%5.1f' % t
            lines.append(s+"\n")
        for chan in jpi[2:]:
            val = chan[0]
            fix = chan[1]
            fc = ['f' if f else ' ' for f in fix]
            s = ""
            for p in range(len(val)):
                s += "%15.8e%c  " % (val[p],fc[p])
            
            nl = (len(s)-1)//72 + 1
            for il in range(nl):
                ss = s[il*72:(il+1)*72]
                lines.append(ss.replace('e','E')+"\n")
    lines.append('    0\n')
    
   
    normalizations = []   # decide to not print those for now
    lines.append('%5i\n' % len(normalizations))
    for n in normalizations:
        s1 = '  %-8s' % n[0]
        fc = 'f' if n[2] else ' '
        s2 = '%15.8e%c    %15.8e  %8.3f     \n' % (n[1],fc,n[3],n[4])
        s = s1 + s2.replace('e','E')
        lines.append(s)

    
    lines.append('    0    0    0    0\n')
    lines.append('%5i\n' % len(partitions))
    for p in partitions:
        proj = p[0]
        targ = p[1]
        lm = p[2]
        s = '%-5s%-5s %4i %4i\n' % (proj[:5],targ[:5],lm,p[3])
        lines.append(s)
        q = nuclei[proj]
        s = '%5i %4.1f %4.1f %14.10f %4.1f\n' % (q[0],q[1],q[2],q[3],q[4])
        q = nuclei[targ]
        lines.append(s)
        s = '%5i %4.1f %4.1f %14.10f %4.1f\n' % (q[0],q[1],q[2],q[3],q[4])
        lines.append(s)

    #if covariances is not None:
    print('cov:', covariances)
    if covariances is not None and covariances[1]:
        pVarying,pMatrix = covariances
        nv = len(pVarying)
        lines.append('Covariance matrix %s :\n' % nv)
        lines.append(6*' ' +(' '.join(['%12i' % (i+1) for i in range(nv)])+'\n'))
        for i in range(nv):
            l = '%6i %6i' % (i+1,pVarying[i][0])
            for j in range(i+1):
                l += '%13.5e' % pMatrix[i,j]
            lines.append(l+'\n')

    return lines

if __name__=="__main__":

    edaIn = sys.argv[1]
    print("\ngetEDA\n")
    lines = open(edaIn).readlines()
    print(" Read in ",len(lines)," lines from file",edaIn)

    if len(sys.argv[1:])>1:
        edacov= sys.argv[2]
        covlines = open(edacov).readlines()
        print(" Read in ",len(covlines)," lines from file",edacov)
    else:
        covlines = None

    edaSpec = getEDA(lines, covlines, True)
    comment, partitions, boundaries, rmatr, channelList, normalizations, nuclei,covariances  = edaSpec
    
    print("\nputEDA\n")
    outlines = putEDA(comment, partitions, boundaries, rmatr, normalizations, nuclei, covariances, False)
    
    edaOut = edaIn + ".echo"
    open(edaOut,'w').writelines(outlines)
    print(" Wrote file ",edaOut)
