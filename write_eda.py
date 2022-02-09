##############################################

#                                            #
#    Ferdinand 0.40, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,eda,hyrma         #
#                                            #
##############################################

import math,numpy
from pqu import PQU as PQUModule
from fudge import documentation as documentationModule
import fudge.resonances.resolved as resolvedResonanceModule
from getCoulomb import *
from eda_parfile import putEDA
import os,pwd,time

##############################################  write_eda

def write_eda(gnd,outFile,Lvals,verbose,debug):
    
    PoPs = gnd.PoPs
    rrr = gnd.resonances.resolved
    Rm_Radius = gnd.resonances.scatteringRadius
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
#    emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs('MeV')
#    emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs('MeV')

    BC = RMatrix.boundaryCondition
    BV = RMatrix.boundaryConditionValue
    IFG = RMatrix.reducedWidthAmplitudes  
    fmscal = 0.0478450
    etacns = 0.1574855
    amu    = 931.494013
    getiso = 0
    
    # print BC,type(BC),isinstance(BC,float),float(BC)
    if BC is None or BC==resolvedResonanceModule.BoundaryCondition.EliminateShiftFunction or BC==resolvedResonanceModule.BoundaryCondition.Brune:
        print(" BoundaryCondition =", BC, " cannot be converted to EDA")
        raise SystemExit
    elif BC==resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum:
        btype = 'L'
    elif BC==resolvedResonanceModule.BoundaryCondition.Given:
        btype = 'B'
    else:
        print("Boundary condition BC <%s> not recognized" % BC, " in write_eda.")
        raise SystemExit

    proj,targ = gnd.projectile,gnd.target
    elasticChannel = ('%s + %s' % (proj,targ))
    if verbose: print('Elastic is <%s>\n' % elasticChannel) 

    partitions = []
    rmatr = []
    nuclei = {}
    itc = 0; elasticPartition = None
    for rreac in RMatrix.resonanceReactions:
        label = rreac.label
        #p,t = rreac.ejectile,rreac.residual
        p,t = rreac.ejectile,rreac.residual
        if rreac.scatteringRadius is not None:
            prmax =  rreac.scatteringRadius.getValueAs('fm')
        else:
            prmax = Rm_global

        if Lvals is not None:
            Lmax =  int(Lvals[itc])  # index based on size of partitions list
        else:
            Lmax = 0
            for Jpi in RMatrix.spinGroups:
                for ch in Jpi.channels:
                    if ch.resonanceReaction == label: Lmax = max(Lmax,ch.L)
        
        projectile = PoPs[p];
        target     = PoPs[t]; 
        pMass = projectile.getMass('amu');   
        tMass = target.getMass('amu')
        if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
        if hasattr(target, 'nucleus'):     target = target.nucleus
        if debug: print('For nuclides:',projectile,target)

        pZ = projectile.charge[0].value; tZ =  target.charge[0].value
        jp,pt,ep = projectile.spin[0].float('hbar'), projectile.parity[0].value, 0.0
        try: 
            jt,tt,et = target.spin[0].float('hbar'),     target.parity[0].value, 0.0
        except: 
            jt,tt,et = 0.0,1,0.0   #  Placeholder properties for nucleus after Reich-Moore capture
        nuclei[p.strip()] = pt,jp,pZ,pMass,0.0
        nuclei[t.strip()] = tt,jt,tZ,tMass,0.0
        
        if label == elasticChannel:
            lab2cm = tMass/(tMass + pMass)
            elasticPartition = 0 # now in 'partitions
            partitions.insert(elasticPartition,[p,t,Lmax,0,prmax])
        else:
            partitions.append([p,t,Lmax,0,prmax])
        itc += 1
    if verbose: print(len(list(nuclei.keys())),"nuclei: ",nuclei)

    comment = 'From GNDS: %s\n' % time.ctime()

    if Lvals is not None: print("Remake channels in each partition for L up to ",Lvals,'for the order',[rreac.label for rreac in RMatrix.resonanceReactions])

# any variable or data namelists in the documentation?

    try:
        ddoc = gnd.styles.getEvaluatedStyle().documentation.get('Fitted_data')
        lines = ddoc.getLines()
        nnorms = int(lines[0].split()[0])
        if verbose: print(nnorms,"normalization variables")
    except:
        nnorms = 0
    normalizations = []
    for n in range(nnorms):
        c = ' ' + lines[n+1]
        #if verbose: print 'c['+c+']'
        tag = c[2:10]
        try: 
            norm = float(c[11:25])
        except:
            norm = 1.0
        fixed = c[25]=='f'
        try: 
            rest = [float(x) for x in c[26:].split()]
        except:
            rest = [1.,1.]
        normalizations.append([tag,norm,fixed,rest[0],rest[1]])

#                  Find JpiList in the correct order for EDA 
    npart = elasticPartition
    p,t = partitions[npart][:2]
    pt,jp,pZ,pMass,isospinp = nuclei[p]
    tt,jt,tZ,tMass,isospint = nuclei[t]
    Lmax = partitions[npart][2]
    if verbose: print("Elastic partition: #,p,t,Lmax:",npart,p,t,Lmax)
    JpiList = []
    for Lref in range(Lmax+1):
        pi = (-1)**Lref * pt * tt
        Smax = jp + jt
        # if verbose: print '\nL, pi, Smax =',Lref,pi,Smax
        for ism in range(int(Smax)+1):
            sch = Smax - ism
            JTmax2 = int(2*(Lref+sch)+0.5)
            JTmin2 = int(2*abs(Lref-sch)+0.5)
            #print 'sch: ',sch, 'JTmin, JTmax:',JTmin2*0.5,JTmax2*0.5
            for JT2 in range(JTmax2,JTmin2-1,-2):
                JT = JT2*0.5
                JpiEDA = JT,pi
                if JpiEDA not in JpiList: JpiList.append(JpiEDA)
    print("   EDA list for Jpi",JpiList)
    NJS = len(JpiList)
    print("   spinGroups:",[(float(Jpi.spin),int(Jpi.parity)) for Jpi in RMatrix.spinGroups])

    nwidths = 0
    boundaries = []
    nchans = 0
    for JpiEDA in JpiList:
        for Jpi in RMatrix.spinGroups:
            jtot = float(Jpi.spin);  parity = int(Jpi.parity)
            if JpiEDA != (jtot,parity): continue

            pi = '+' if parity>0 else '-'
            if verbose: print("\n#",jtot,pi)
            R = Jpi.resonanceParameters.table
            poleEnergies = R.getColumn('energy','MeV')
            widths = [R.getColumn( col.name, 'MeV' ) for col in R.columns if col.name != 'energy']
    
            existingChs = {}
            for ch in Jpi.channels:
                rr = ch.resonanceReaction
                lch = ch.L
                sch = float(ch.channelSpin)
                existingChs[(rr,lch,int(sch*2))] = ch
                if verbose: print("Existing chs:",rr,lch,sch) #existingChs
                # print "Existing chs:",rr,lch,sch," ch#",ch.columnIndex
    
            jpi = '%2i/2%1s' % (int(2*jtot),pi)
            Jpars = [ [], jpi ]  # no isospins here
            
            pars = [];  parf = []
            for i in range(len(poleEnergies)):
                e = poleEnergies[i] * lab2cm
                pars.append(e)
                parf.append(False)  
                          
            if len(poleEnergies)==0: 
                eD = 200.; widthD = 0.0
                if verbose: print("For",jtot,pi,"Add dummy pole at",eD,", width",widthD)
                pars.append(eD)
                parf.append(False)  
                
            Jpars.append([pars,parf])
            if verbose: print('1 trace Jpars',Jpars)
    
    #         add in any missing partial wave channels, as EDA needs them all spelled out
    #         all with default BC
    
            for npart in range(len(partitions)):
                p,t = partitions[npart][:2]
                rr = '%s + %s' % (p,t)
                pt,jp,pZ,pMass,isospinp = nuclei[p]
                tt,jt,tZ,tMass,isospint = nuclei[t]
                Lmax = partitions[npart][2]
                prmax = partitions[npart][4]
                smin = abs(jt-jp)
                smax = jt+jp
                s2min = int(2*smin+0.5)
                s2max = int(2*smax+0.5)
                for s2 in range(s2max,s2min-1,-2):
                    sch = s2*0.5
                    lmin = int(abs(sch-jtot)+0.5)
                    if pMass<1e-10 and jp<0.01: lmin=max(lmin,1)    # no E0: only E1, E2
                    lmax = min(int(sch+jtot +0.5),Lmax)
                    for lch in range(lmin,lmax+1):
                        if parity != pt*tt*(-1)**lch: continue
                        nchans += 1
                        # print "\npartition#,L,S:",npart,lch,sch
                        pars = [];  parf = []
                        if (rr,lch,s2) in list(existingChs.keys()) and len(poleEnergies)>0: 
                            ch = existingChs[(rr,lch,s2)]
         
                            n = ch.columnIndex
                            # rr = ch.resonanceReaction
                            rreac = RMatrix.resonanceReactions[rr]
                
                            if btype == 'L':
                                bndx = -lch
                            else:              # btype='B'
                                bndx = BV
                            if ch.boundaryConditionValue is not None: 
                                bndx = float(ch.boundaryConditionValue)
                            if verbose: print(p,t,"existing",lch,sch,", R,B=",prmax,bndx,'@',nchans,' poles:',len(poleEnergies))
    
                            if rreac.Q is not None:
                                Q_MeV = rreac.Q.getConstantAs('MeV')
                            else:
                                reaction = rreac.reactionLink.link
                                Q_MeV = reaction.getQ('MeV') 
                
                            for i in range(len(poleEnergies)):
                                e = poleEnergies[i] * lab2cm + Q_MeV
                                width = widths[n-1][i]
                                width *= lab2cm**0.5 if IFG else lab2cm
                
                                if not IFG: 
                                    penetrability,shift,dSdE,W = getCoulomb_PSdSW(e,lch, prmax, pMass,tMass,pZ,tZ, fmscal,etacns, False)
                                    rwa = (abs(width) /(2. * penetrability))**0.5
                                    if width<0.: rwa = -rwa
                                    width = rwa
                                pars.append(width)
                                parf.append(False)
                                nwidths += 1                            
                                print('%4.1f%s %3s %3s LS,' % (jtot,pi,p,t),lch,sch,n,i,'rwa=',width,'from',widths[n-1][i])
                                # print 'LS,ch#,pole#',lch,sch,n,i,'c pars:',pars
                        else: # put in dummy widthD = 0.         
                            pars.append(widthD)
                            parf.append(False)
                            bndx = -lch
                            if verbose: print(p,t,"added",lch,sch,", R,B=",prmax,bndx,'@',nchans)
                            print(p,t,"added",lch,sch,", R,B=",prmax,bndx,'@',nchans)
                            # print 'd pars:',pars
                            
                        boundaries.append([prmax,True,bndx,False,0.]) 
                           
                        Jpars.append([pars,parf])
                        if verbose: print(npart,'2 trace Jpars',Jpars)
    
        rmatr.append(Jpars)               
        if verbose: print('R-matrix for ',jtot,pi,':\n',Jpars)
    #if verbose: print 'R-matrix parameters:\n',rmatr

    covariances = None
    outlines = putEDA(comment, partitions, boundaries, rmatr, normalizations, nuclei, covariances, False)
    open(outFile,'w').writelines(outlines)

    print("\nEDA file: %i partitions, %i,%i channels, %i widths, %i normalizations" % (len(partitions),len(boundaries),nchans,nwidths,nnorms))
