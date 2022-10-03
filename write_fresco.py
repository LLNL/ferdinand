##############################################
#                                            #
#    Ferdinand 0.41, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################

import math,numpy,os,sys
import fudge.resonances.resolved as resolvedResonanceModule
from pqu import PQU as PQUModule
from fudge.processing.resonances.getCoulombWavefunctions import *
from fudge import documentation as documentationModule
from PoPs.groups.misc import *

def nuclIDs (nucl):
    datas = chemicalElementALevelIDsAndAnti(nucl)
    if datas[1] is not None:
        return datas[1]+str(datas[2]),datas[3] 
    else:
        return datas[0],0

##############################################  write_fresco

def write_fresco(gnd,outFile,verbose,debug,nozero,background,bound,one,egrid,anglegrid):

    name_frin = outFile+'.frin'
    if one: name_frin = '='
    name_frout = outFile+'.frout'  # name only
    name_pars = outFile+'.pars'
    name_srch = outFile+'.search'
    if one: name_srch = outFile
    grid = egrid>0.
    angles = anglegrid is not None

    if not grid: 
        search = open(name_srch ,'w')
    if not one:
        frin = open(name_frin ,'w')
        if grid:
            pars = frin
            print('\nWriting temporary grid file %s'  % name_frin,' dE=',egrid,'MeV, angles=',angles,anglegrid)
        else:
            pars = open(name_pars ,'w+')
            print('\nWriting files %s, %s and %s'  % (name_frin,name_srch,name_pars))
    else:
        frin = search
        pars = search
        #print '\nWriting only file %s'  % name_srch
    if verbose and background is not None: print("Background poles are above ",background)
    if verbose and bound is not None: print("Bound poles are below ",bound)

    PoPs = gnd.PoPs
    rrr = gnd.resonances.resolved
    Rm_Radius = gnd.resonances.scatteringRadius
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
    emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs('MeV')
    emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs('MeV')

    BC = RMatrix.boundaryCondition
    BV = RMatrix.boundaryConditionValue
    print("    BC,BV=",BC,BV)
    IFG = RMatrix.reducedWidthAmplitudes  # pass through to SFRESCO 'RWA'
    KRL = RMatrix.relativisticKinematics  
    RWA = 'T' if IFG else 'F'
    fmscal = 0.0478450
    etacns = 0.1574855

#  Input parameters for FRESCO:
    hcm = 0.1; rmatch = Rm_global; pel = 0; exl = 0

    proj,targ = gnd.projectile,gnd.target
    elasticChannel = '%s + %s' % (proj,targ)
    if verbose: print('Elastic is <%s>\n' % elasticChannel , ' Rm=',Rm_global)
    targNucl = nuclIDs(targ)[0]  # to go in first partition in Fresco

    tLevels = {}; reac = {}; incident = {} 
    for rreac in RMatrix.resonanceReactions:  # to equal #resonanceReaction in channels.
        if not rreac.eliminated:
            p,t = rreac.ejectile,rreac.residual
            tnucl,tlevel = nuclIDs(t)
            if not tnucl in list(tLevels.keys()): 
                tLevels[tnucl] = set()
            tLevels[tnucl].add(tlevel)
            reac[t] = rreac
            incident[t] = p
            if rreac.scatteringRadius is not None:
                prmax =  rreac.scatteringRadius.getValueAs('fm')
            else:
                prmax = Rm_global
            #rmatch = max(prmax,rmatch)

#   print 'tLevels:',tLevels, '\nreac:',reac
    #targ = targ if '_' in targ else targ+'_e0'    
    nex_t = [None]; iaval = {}; icval = {}; rr2it = {}; tLev = {} ; pNucl = {}  # it,ic,ia start from 1
    partitions = []
    ic = 0; it = 0
    targs = list(tLevels.keys()); targs.remove(targNucl)
    for tnucl in [targNucl] + targs:   # put elastic channel in first partition
        ic += 1
        ia = 0
        for tlevel in sorted(tLevels[tnucl]):
            ia += 1
            it += 1
            icval[it] = ic
            iaval[it] = ia
            pNucl[ic] = tnucl
            tLev[(tnucl,ia)] = tlevel
            t = nuclideIDFromIsotopeSymbolAndIndex(tnucl,tlevel)
            p = incident[t]
            if p==proj and t==targ: 
                pel = ic
                exl = ia
            rr2it[reac[t].label] = it             
            partitions.append(reac[t].label)
        nex_t.append(ia)
    itcm = it        
    #print "nex_t:",nex_t[1:],"\ntLev:",tLev,"\npNucl:",pNucl,"\nrr2it:",rr2it
#     for it in range(1,itcm+1): print it,": icval:",icval[it]," iaval:",iaval[it]

    #if debug: print itcm, ' reaction pairs: ',rr2it

    if pel<1: 
        print(" Elastic channel not found in particle rreacs!")
        print(" Looked for <%s + %s>" % (proj , targ))
        print("        among ",[rreac.label for rreac in RMatrix.resonanceReactions])
        print(" *** Setting pel=1")
        pel = 1

    sbndx = ''
    if BC is None:
        btype = 'S'
    elif BC==resolvedResonanceModule.BoundaryCondition.EliminateShiftFunction:
        btype = 'S'
    elif BC==resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum:
        btype = 'L'
    elif BC=='Brune':
#       print "Should have used Brune transformation first"
#       print "  but ignore that, and give BC=resolvedResonanceModule.BoundaryCondition.EliminateShiftFunction "
#       btype = resolvedResonanceModule.BoundaryCondition.EliminateShiftFunction
        btype = 'A'
    elif BC==resolvedResonanceModule.BoundaryCondition.Given:
        btype = 'B'
        bndx = BV
        sbndx = 'bndx=%s %s' % (bndx,bndx)
    else:
        print("Boundary condition BC <%s> not recognized" % BC,"in write_fresco")
        raise SystemExit
    print("Write fresco btype = ",btype,sbndx," from BC,BV=",BC,BV)

# also need vars=nparameters !!!
    vars = 0;
    jtmax = 0.0
    for Jpi in RMatrix.spinGroups:
        jtmax = max(jtmax,Jpi.spin)
        R = Jpi.resonanceParameters.table
        for i in range(0,len(R)):
            found = False            
            dampingfound = False
            for ch in Jpi.channels:
                n = ch.columnIndex
                rr = ch.resonanceReaction
                rreac = RMatrix.resonanceReactions[rr]
                if not rreac.eliminated:
                    rwa = R[i][n]
                    if abs(rwa)>1e-10 or not nozero:
                        vars += 1  #  kind=4
                        found = True
                else:
                    dampingfound = True
            if found:        vars += 1   #  kind=3
            if dampingfound: vars += 1   #  kind=7

# any variable or data namelists in the documentation?
    docData = []; docVars = []
    try:
        ddoc = RMatrix.documentation.computerCodes['R-matrix fit'].inputDecks[-1]   # last fit
        for line in ddoc.body.split('\n'):
            if '&data'     in line.lower() :  docData += [line]
            if '&variable' in line.lower() :  docVars += [line]
    except:
        pass

    datas = len(docData)
    dvars = len(docVars)
#     print('datas:',docData)
#     print('dvars:',docVars)

    if not grid:          #  Write search file for SFRESCO:
        vars += dvars
        string = " '%s' '%s' \n" % (name_frin,name_frout)
        search.write(string)
        search.write(" %i %i \n" % (vars,datas) )   # nparameters, ndatasets=0
        if not one: search.write(" &variable param_file ='%s' /\n" % name_pars )
        nparameters = 0
        nlab = 1
        thmin,thinc = 30,50
        dist = ''
    else:                 #  Write input file for FRESCO grid reconstruction
        nparameters = vars
        emin = max(emin,egrid)
        nlab = (emax-emin)/egrid + 1
        datas = 0; dvars = 0
        if angles:
            thinc = anglegrid[1]
            thmin = 0.0
            dist = 'dist0 = %s' % anglegrid[0]
        else:
            thmin,thinc = 30,50
            dist = ''

    if one: frin.write('\n')  # separator line
    frin.write("R-matrix calculation " + outFile + '\n')
    frin.write('NAMELIST \n')
    frin.write('&FRESCO \n hcm=%f rmatch=%f rintp=%f \n jtmin=0.0 jtmax=%.1f absend=-1.0 \n' % (hcm,rmatch,0.5,jtmax))
    frin.write(' thmin=%s thmax=-180 thinc=%s %s smats=0 weak=0 \n' % (thmin,thinc,dist))
    if KRL: frin.write(" rela='h' relref=%i \n" % KRL)
    frin.write(' iter = 0 nrbases=1 nparameters=%i btype="%s" %s boldplot=F\n pel=%i exl=%i elab(1:2) = %e %e nlab(1)=%i /\n ' % (nparameters,btype,sbndx,pel,exl,emin,emax,nlab))
    frin.write('\n')

    further = []
    lab2cm = 0
    p1p2 = {}
    for it in range(1,itcm+1):
        ic = icval[it]
        ia = iaval[it]
        tnucl = pNucl[ic]
        tlevel = tLev[(tnucl,ia)]
        # t = tnucl + '_e%i' % tlevel
        t = nuclideIDFromIsotopeSymbolAndIndex(tnucl,tlevel)
        rreac = reac[t]
        rr = rreac.label
        reaction = rreac.reactionLink.link
        cpot = ic

        p,t = rreac.ejectile,rreac.residual
        #t = t if '_' in t else t+'_e0'
        if debug: print("\nExamine rreac rr=",rr,' of ',p,' + ',t, 'it,ic =',it,ic)

        projectile = PoPs[p];
        target     = PoPs[t]; 
        pMass = projectile.getMass('amu');   
        tMass = target.getMass('amu')
        if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
        if hasattr(target, 'nucleus'):     target = target.nucleus
        if debug: print('For nuclides:',projectile,target)

        pZ = projectile.charge[0].value; tZ =  target.charge[0].value

        if rreac.Q is not None:
            Q_MeV = rreac.Q.getConstantAs('MeV')
        else:
            Q_MeV = reaction.getQ('MeV')
        if rreac.label == elasticChannel: lab2cm = tMass / (pMass + tMass)

        pgs,plevel = nuclIDs(p)
        # pgs,plevel = p.split('_e') if '_e' in p else [p,0]
        tgs,tlevel = nuclIDs(t)
#  Assume only targets have excited states
        clevel = int(tlevel)

        jp,pt,ep = projectile.spin[0].float('hbar'), projectile.parity[0].value, 0.0
        try:
            jt,tt,et =     target.spin[0].float('hbar'), target.parity[0].value,     target.energy[0].pqu('MeV').value
        except:
            jt,tt,et = 0.0,1,0.0
        if debug: print(cpot,'target ',t,' spin: ',jt,' parity: ',tt,' E_inel =',et)
        p1 = (1-pt)//2; p2 = (1-tt)//2  #  0 or 1 for + or - parities
        p1p2[rreac.label] = (p1,p2)       # save for later access from ic, and ia

        if rreac.scatteringRadius is not None:
            prmax =  rreac.scatteringRadius.getValueAs('fm')
        else:
            prmax = Rm_global
        if debug: print("partition ",cpot,' Q=',Q_MeV,' MeV,  prmax =',prmax,'\n')

        nex = nex_t[ic]
        if clevel==0:
            if debug: print(p, pMass, pZ, nex, t,tMass,tZ, Q_MeV, prmax)
            string = " &PARTITION namep='%s' massp=%f zp=%i nex=%i namet='%s' masst=%s zt=%i qval=%f prmax=%s /\n" % ( pgs, pMass, pZ, nex, tgs,tMass,tZ, Q_MeV, prmax) 
            further.append(string)
            string = ' &STATES  cpot =%i jp=%s ptyp=%i ep=%f  jt=%s ptyt=%i et=%f /\n' % (cpot,str(jp),pt,ep, str(jt),tt,et)
        else:
            string = ' &STATES  cpot =%i                            jt=%s ptyt=%i et=%f /\n' % (cpot,               str(jt),tt,et) 
        further.append(string)

    if lab2cm <1e-5:
        print("Elastic channel not found")
        raise SystemExit

    further.append(' &PARTITION /\n\n')     
    cpot = 0
    for rreac in RMatrix.resonanceReactions: 
        if not rreac.eliminated:
            cpot += 1
            further.append(' &pot kp=%i  type=0 p(1:3) = 0 0 1.0 /\n' % cpot )

    further.append(' &pot /\n\n')     
    further.append(' &overlap /\n\n')     
    further.append(' &coupling /\n\n')    
    further.append('EOF\n')

    if not grid: 
        for l in further:
            frin.write(l)
        pars.write('\n\n')
        
    term = 0; vars = 0; backgrounds = []; bounds=[]
    for Jpi in RMatrix.spinGroups:
        jtot = Jpi.spin;  parity = int(Jpi.parity)
        pi = '+' if parity>0 else '-'
        # print('parity',parity,'is',type(parity))
        x = (1-parity)//2
        if verbose: print("\nWF: J,pi =",jtot,pi,'(x=',x,')')
        R = Jpi.resonanceParameters.table
        poleEnergies = R.getColumn('energy','MeV')
        widths = [R.getColumn( col.name, 'MeV' ) for col in R.columns if col.name != 'energy']

        for i in range(len(poleEnergies)):
            term += 1
            e = poleEnergies[i] * lab2cm
            nam='J%.1f%s:E' % (jtot,pi) + str(e)
            if bound is not None and e < bound:
                nam = 'BS:%.1f%s' % (jtot,pi)
                bounds += [[jtot,pi,e]]
            if background is not None and e > background:
                nam = 'BG:%.1f%s' % (jtot,pi)
                backgrounds += [[jtot,pi,e]]
            if debug: print(" R-matrix pole at ",e,"MeV: ",nam," (term =",term,")")
            if debug: print(" L values:",[ch.L for ch in Jpi.channels])

            found = False
            damp = None
            for ch in Jpi.channels:
                n = ch.columnIndex
                rr = ch.resonanceReaction
                rreac = RMatrix.resonanceReactions[rr]
                computePenetrability = rreac.computePenetrability
                if not computePenetrability:
                    print('Option computePenetrability=F not yet implemented for Fresco')
                    sys.exit(1)
                lch = ch.L
                # bndx = None if ch.boundaryConditionValue is None else float(ch.boundaryConditionValue)
                bndx = ch.boundaryConditionValue 
                sch = abs(float(ch.channelSpin))   #  abs in case any old format still present
                width = widths[n-1][i]
                width *= lab2cm**0.5 if IFG else lab2cm

                if rreac.eliminated:
                    if IFG: width =  2*width**2  # P=1 for gammas and fission
                    if damp is None: damp = 0.0
                    damp += width
                else:
                    p1,p2 = p1p2[rreac.label] 
                    phaz =  p1 + p2 + lch - x
                    if phaz % 2 == 1:    # should be even. If not: stop
                        print('Error: ic,ia=',ic,ia,' with parities',p1,p2,' L=',lch,'x,ph:',x,phaz)
                        sys.exit(1)
#                   else:
#                       print 'Phases: ic,ia=',ic,ia,' with parities',p1,p2,' L=',lch,'x,ph:',x,phaz
                    phase = (-1)**(phaz//2)
                    width *= phase    # convert to phase for Fresco, which has explicit i^L Huby phases.

                    it = rr2it[rr]
                    wnam = 'w'+str(it)+','+ (nam[1:] if nam[0]=='J' else nam)
                    icch = icval[it]
                    iach = iaval[it]
                    if debug: print('%s E= %.3f MeV, damp=%s width=%.4e l=%i S=%.1f B=%s p1,p2,phaz,s=%i,%i,%i: %i %i' % (p, e, damp, width,lch,sch,bndx,p1,p2,x,phaz,phase))
                    if abs(width)>1e-10  or not nozero:
                        vars += 1
                        step = min(abs(width)/100.,99.) if width != 0. else 0.01 
                        if bndx is None:
                            string = "&Variable kind=4 name='%s' term=%i icch=%i iach=%i lch=%i sch=%.1f width=%s rwa=%s step=%.1e ivar=%i/\n" % (wnam[0:15], term,icch,iach,lch,sch,width,RWA,step,vars)
                        else:
                            string = "&Variable kind=4 name='%s' term=%i icch=%i iach=%i lch=%i sch=%.1f width=%s rwa=%s B=%s step=%.1e ivar=%i/\n" % (wnam[0:15], term,icch,iach,lch,sch,width,RWA,bndx,step,vars)
                        pars.write(string)
                        found = True
            if found :
                vars += 1
                estep = 0.1
                if background is not None and e > background:  estep = 0   #   background poles have fixed energies
                if bound is not None      and e < bound:       estep = 0   #   bound poles have fixed energies

                string = "&Variable kind=3 name='%s'    term=%i jtot=%.1f par=%s energy=%s step=%s ivar=%i/ ! from %s\n" % (nam[0:15], term,jtot,parity, e,estep,vars,R.columns[0].unit)
                pars.write(string)

                if damp is not None:
                    vars += 1
                    maxdamp = background if background is not None else 20.
                    maxdamp = max(maxdamp,damp*1.1)
                    string = "&Variable kind=7 name='%s' term=%i damp=%s step=%s valmin=0.0 valmax=%s ivar=%i/ ! from %s\n\n" % ('D:'+nam[0:13], term,damp,estep,maxdamp,vars,R.columns[0].unit)
                    pars.write(string)
           
    if not grid:
        pars.write('\n\n')
        pars.write('\n'.join( docVars ))
        pars.write('\n\n')
        pars.write('\n'.join( docData ))
        print("\nFresco input: %i+%i=%i &Variable and %i &Data lines written" % (vars,dvars,vars+dvars,datas))
        if len(bounds)>0:
            print(" with fixed bound poles", end=' ')
            for j,pi,e in sorted(bounds):
                print("%s%s %s," % (j,pi,e), end=' ')
            print(' ')
        if len(backgrounds)>0:
            print(" with fixed background poles", end=' ')
            for j,pi,e in sorted(backgrounds):
                print("%s%s %s," % (j,pi,e), end=' ')
    else:
        for l in further:
            frin.write(l)
        frin.close()
        # print "\nFresco input: %i &Variables written for grid reconstruction:\n" %  vars

    return partitions

