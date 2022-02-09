

##############################################
#                                            #
#    Ferdinand 0.40, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################

from fudge import reactionSuite as reactionSuiteModule
from fudge import styles as stylesModule
from fudge import documentation as documentationModule

import fudge.resonances.resonances as resonancesModule
import fudge.resonances.scatteringRadius as scatteringRadiusModule
import fudge.resonances.resolved as resolvedResonanceModule
import fudge.resonances.common as commonResonanceModule
import fudge.resonances.scatteringRadius as scatteringRadiusModule

import fudge.covariances.modelParameters as covarianceModelParametersModule
import fudge.covariances.covarianceSection as covarianceSectionModule
import fudge.covariances.covarianceSuite as covarianceSuiteModule

from PoPs import database as databaseModule
from PoPs import misc as miscModule
from PoPs.families import gaugeBoson as gaugeBosonModule
from PoPs.families import baryon as baryonModule
from PoPs.families import nucleus as nucleusModule
from PoPs.families import nuclide as nuclideModule
from PoPs.quantities import spin as spinModule
from PoPs.groups.misc import *

from pqu import PQU as PQUModule
from fudge import physicalQuantity as physicalQuantityModule
import numpy

from xData import table as tableModule
import xData.constant as constantModule
from xData import link as linkModule
import xData.xDataArray as arrayModule

from zeroReaction import *
from getCoulomb import *
import os,pwd,time
fmscal = 0.0478450
etacns = 0.1574855
amu = 931.494013
spinUnit = spinModule.baseUnit

##############################################  read_azure

def read_azure(inFile,covFile,elastic,RWA,noCov, emin,emax, verbose,debug,amplitudes):

    azr = open(inFile,'r')
    UPvalues = []
    RWAupdate = covFile is not None and not noCov
    if RWAupdate: 
       covlines = open(covFile).readlines()
       nFitted = 0
       nRp = 0
       nLevels = 0
       for l,line in enumerate(covlines):
           # print(line[25:26])
           if line[25:26]==')':
               i = int(line[22:25])
               val = float(line[30:46])
               # print('fit:',line[47:50])
               kind = 0 if line[17:20]=='rwa' else 1 if line[14:20]=='energy' else 2
               if kind<2:
                   fitted = line[47:50] == 'Fit'
                   UPvalues.append([val,kind])
                   if fitted: nFitted += 1
                   if kind<=1: nRp += 1
                   if kind==1: nLevels += 1
           if line[:17]=='Covariance Matrix':
               header = covlines[l+2]
               R2icor = [int(i) for i in header.split()]
               nv = len(R2icor)
               norms = nv - nFitted
               print('Updated values ',len(UPvalues),'of which',nFitted,'of',nRp,'R-matrix parameters in',nLevels,'levels, with',norms,'fitted norms')
               if debug: print('R2icor:',R2icor)
               print('R2icor for R-matrix::',R2icor[:nFitted])
               pMatrix = numpy.zeros([nv,nv])
               for i in range(nv):
                   ll = l+3+i
                   vals =[float(x) for x in covlines[ll].split()]
                   # print(len(vals),'vals:',vals)
                   pMatrix[i,:] = numpy.asarray(vals[1:])
       if debug: print('Covariance matrix:\n',pMatrix)

    line = azr.readline(); 
    while (not '<config>' in line): line = azr.readline()
    config = [' ' for i in range(11)]
    for i in range(11):
        config[i] = azr.readline().split()[0]
    line = azr.readline(); 
    while (not '</config>' in line): line = azr.readline()
    while (not '<levels>' in line): line = azr.readline()
    line = azr.readline(); 
    maxvar = 10000
    vars = [' ' for i in range(maxvar)]
    JpiList = [];  pairList = set(); tLevels = {};

    nogamma = True
    i = 0
    while (not '</levels>' in line): 
        while (len(line)<5): 
            line = azr.readline()
            #if debug: print 'read in line=',line
            if line =='': 
                 print("Premature EOF \nStop")
                 raise SystemExit
            
        if '</levels>' in line: break

        lJ,lPi,E,lFix,aa,ir,s,l,ID,Active,cFix,gamma,j1,pi1,j2,pi2,e2,m1,m2,z1,z2,eSepE,sepE,j3,pi3,e3,pType,chRad,g1,g2,mask = line.split()
        Q = float(eSepE) - float(sepE)
        E = str(float(E) - float(eSepE))
        #if debug: print i,'::',lJ,lPi,E,lFix,aa,ir,s,l,ID,Active,cFix,gamma,j1,pi1,j2,pi2,e2,m1,m2,z1,z2, Q #eSepE,sepE #,j3,pi3,e3,pType,chRad,g1,g2,mask 
	
        A1= int(float(m1)+0.5); A2= int(float(m2)+0.5)
        Z1= int(z1); Z2= int(z2)
        if Z1==0 and i==1: nogamma = False

        p = idFromZAndA(Z1,A1) if A1!=0 else 'photon'
        t = idFromZAndA(Z2,A2)
        if (lJ,lPi) not in JpiList: 
            JpiList.append((lJ,lPi))

        if not t in list(tLevels.keys()): tLevels[t] = set()
        tLevels[t].add(e2)

        pairList.add((p,t,e2,Z2,m2,j2,pi2,Q,Z1,m1,j1,pi1,chRad,i))

        vars[i] = (lJ,lPi,E,ir,s,l,ID,gamma,p,t,e2,chRad)  # all strings
        #print 'vars[',i,'] =',(lJ,lPi,E,ir,s,l,ID,gamma,p,t,e2,chRad)  # all strings
        line = azr.readline()
        i += 1

    pairList = sorted(pairList)
    nvars = i # #elements+1
    if debug and False:
        print(nvars,' R-matrix parameters')
        print('pairList:',pairList)
        print('tLevels:',tLevels)
    if verbose: print('tLevels:',tLevels)

    domain = stylesModule.projectileEnergyDomain(emin,emax,'MeV')
    style = stylesModule.evaluated( 'eval', '', physicalQuantityModule.temperature( 300., 'K' ), domain, 'from '+inFile , '0.1.0' )
    PoPs_data = databaseModule.database( 'azure', '1.0.0' )
    resonanceReactions = commonResonanceModule.resonanceReactions()
    computePenetrability = True
    MTchannels = {}

    LRU = 1      # resolved resonances
    LRF = 7      # R-Matrix limited
    KRM = 4 if nogamma else 3
    SHF = 0  # native input 
    KRL = False  # not relativistic
    IFG = amplitudes  # True for RWA in ev**{1/2} for gnd intermediate file, False for Formal Widths
    LRP = 2  # do not reconstruct resonances pointwise
    BC = resolvedResonanceModule.BoundaryCondition.Brune
    if RWA: BC = resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum
    
    if KRM==3:
        approximation = 'Reich_Moore'
    elif KRM==4:
        approximation = 'Full R-Matrix'

    #if debug: print '\nUnsorted target excited states given by tLevels:',tLevels
    for t in tLevels:
        tLevels[t] = sorted(tLevels[t])
    if verbose: print('   Order of target excited states given by tLevels:',tLevels)

    ZAdict = {}
    cm2lab = 0
#    print 'pairList:',[str((p,t,et))+';' for p,t,et,zt,tMass,jt,pit,Q,zp,pMass,jp,pip,chRad,i in pairList]
#    print 'sorted(pairList):',[str((p,t,et))+';' for p,t,et,zt,tMass,jt,pit,Q,zp,pMass,jp,pip,chRad,i in sorted(pairList)]
    Rm_global = None
    for p,t,et,zt,tMass,jt,pit,Q,zp,pMass,jp,pip,chRad,i in pairList:
        pp = pip
        pt = pit
        jps = str(int(float(jp))) if float(jp).is_integer() else '%i/2' % int(2*float(jp))
        jts = str(int(float(jt))) if float(jt).is_integer() else '%i/2' % int(2*float(jt))
        QI = Q - float(et)

        ia = tLevels[t].index(et)
        pMass = float(pMass); tMass = float(tMass); pip=int(pip); pit=int(pit); et=float(et)
        tex = nuclideIDFromIsotopeSymbolAndIndex(t,ia)

        MT = 5
        if p=='gamma':
           MT = 102
        elif p=='n' :
           MT = 50+ia
        elif p=='H1' :
           MT = 600+ia
        elif p=='H2' :
           MT = 650+ia
        elif p=='H3' :
           MT = 700+ia
        elif p=='He3' :
           MT = 750+ia
        elif p=='He4' :
           MT = 800+ia
        
        
        #if debug: print "Build PoPs for projectile ",p,jps,pp,zp,pMass
        if zp==0 and pMass == 0 :   # g
            projectile = miscModule.buildParticleFromRawData( gaugeBosonModule.particle, p, mass = ( 0, 'amu' ), spin = (jps,spinUnit ),  parity = (pip,'' ), charge = (0,'e') )
        elif zp<2 and pMass < 1.5 and p != 'H1' :  # n or p
            projectile = miscModule.buildParticleFromRawData( baryonModule.particle, p, mass = (pMass,'amu' ), spin = (jps,spinUnit ),  parity = (pip,'' ), charge = (zp,'e') )
        else: # nucleus in its gs
            nucleus = miscModule.buildParticleFromRawData( nucleusModule.particle, p, index = 0, energy = ( 0.0, 'MeV' ) , spin=(jps,spinUnit), parity=(pip,''), charge=(zp,'e'))
            projectile = miscModule.buildParticleFromRawData( nuclideModule.particle, p, nucleus = nucleus,  mass=(pMass,'amu'))
        PoPs_data.add( projectile )

        # Some state of target at energy 'et':
        #if debug: print "Build PoPs for target ",tex,jts,pt,zt,tMass,ia,et
        nucleus = miscModule.buildParticleFromRawData( nucleusModule.particle, tex, index = ia, energy = (et,'MeV' ) , spin=(jts,spinUnit), parity=(pit,''), charge=(zt,'e') )
        target = miscModule.buildParticleFromRawData( nuclideModule.particle, tex, nucleus = nucleus, mass=(tMass,'amu')) 
        PoPs_data.add( target )
        #if debug: print '      level',tex,' to ',target.name,jts,pt,' having ',[target.levels[iaa].energy for iaa in range(ia+1)]

        #if debug: print '               projectile=',p,pMass,zp,jp,pt, ', target=',tex,tMass,zt,jt,pp, ' Q,QI=',Q,QI,' ia=',ia,' MT=',MT
        rr = '%s + %s' % (p,tex)
        ZAdict[ (p,tex) ] = (pMass,tMass,zp,zt,QI)

        if  rr == elastic or (elastic==None and i==0):   #  elastic channel
            elastics = (p,tex)
            print("   Elastic: ",elastics,'( E* =',et,') since elastic=',elastic)
            MT = 2
            cm2lab = (float(tMass) + float(pMass))/float(tMass)

# store MT #s for all reactions that need to include resonance data:
        eminc = max(emin,-QI*cm2lab)
        channelName = "%s + %s" % (p,tex)
        #if debug: print rr," channelName: ",channelName
        prmax = float(chRad)
        MTchannels[rr] = zeroReaction(rr,MT, QI, [projectile,target], None, eminc,emax,'MeV', debug), channelName,prmax,p
        if Rm_global is None:  Rm_global = prmax

    p,tex = elastics    
    gnd = reactionSuiteModule.reactionSuite( p, tex, 'azure R-matrix fit', PoPs = PoPs_data, style = style, interaction='nuclear') 

    if verbose: 
        # if debug: print PoPs_data.toXML( )
        projectile,target = gnd.projectile,gnd.target
        print('\n   GND Projectile,Target: ',projectile,target)
        #print '    GND p,t spins:',gnd.particles[projectile.name].getSpin().value,gnd.particles[target.name].getSpin().value
        print("   ZAdict: ",ZAdict)
    if cm2lab<1e-5:
        print("Missed elastic channel for cm2lab factor!")

#  After making all the channels, and gnd is generated for the elastic channel, now add them to gnd
#
    for rr in MTchannels: 
        reaction,channelName,prmax,p = MTchannels[rr]
        gnd.reactions.add(reaction)
        reactionLink = linkModule.link(reaction)
        computeShiftFactor = True
        rreac = commonResonanceModule.resonanceReaction ( label=rr, reactionLink=reactionLink, ejectile=p, computePenetrability=True,
                                                     computeShiftFactor=computeShiftFactor, Q=None, eliminated=False  ) 
        if prmax is not None and prmax != Rm_global:
            rreac.scatteringRadius = scatteringRadiusModule.scatteringRadius(      
                constantModule.constant1d(prmax, domainMin=emin, domainMax=emax,
                    axes=axesModule.axes(labelsUnits={1: ('energy_in', 'MeV'), 0: ('radius', 'fm')})) )
        resonanceReactions.add(rreac)

#  Now read and collate the reduced channel partial waves and their reduced width amplitudes

# First, update if necessary:
    if RWAupdate: 
        nRp = 0
        IDList=[]
        for i in range(nvars):
            lJ,lPi,E,ir,s,l,ID,gamma,p,t,e2,chRad = vars[i]
            if (ID,E) not in IDList: 
                IDList.append((ID,E))
                v = float(E); kk=1
                vup,kup = UPvalues[nRp]
                vup -= float(eSepE)
                diff = vup - v 
                print('At',nRp,'for J,pi=',lJ,lPi)
                if kk!=kup: print("Error in updating!")
                print('   E = %11.5f  but update to %11.5f : diff =%11.5f' % (v,vup,diff))
                EE = str(vup)
                vars[i] = lJ,lPi,EE,ir,s,l,ID,gamma,p,t,e2,chRad 
                nRp += 1

            v = float(gamma); kk=0
            vup,kup = UPvalues[nRp]
            diff = vup - v 
            if kk!=kup: print("Error in updating!")
            if abs(diff)>1e-10 or not RWA: 
                print('  rwa= %11.5f  but update to %11.5f ; diff =%11.5f ' % (v,vup,diff))
            gamma = str(vup)
            vars[i] = lJ,lPi,EE,ir,s,l,ID,gamma,p,t,e2,chRad 
            nRp += 1

# next we have NJS spin groups, each containing channels and resonances
    spinGroups = resolvedResonanceModule.spinGroups()
    spinGroupIndex = 0
    if debug: print('   JpiList =',JpiList)
    for J,pi in JpiList:
        if verbose: print('J,pi =',J,pi)
        channelList = []
        for i in range(nvars):
            lJ,lPi,E,ir,s,l,ID,gamma,p,t,e2,chRad = vars[i]
            if lJ == J and lPi==pi: 
                ia = tLevels[t].index(e2)
                tex = nuclideIDFromIsotopeSymbolAndIndex(t,ia)

                if (p,tex,s,l,chRad) not in channelList: channelList.append((p,tex,s,l,chRad))
 
        NCH = len(channelList)
        if verbose: print('    Channel list =',channelList)

        idx = 0 # row index in table
        columnHeaders = [ tableModule.columnHeader(0, name="energy", unit="MeV") ]
        channels = resolvedResonanceModule.channels()    
        channelNames = []
        for chidx in range(NCH):
            idx += 1  # row index in table
            p,tex,s,l,chRad = channelList[chidx]
            if debug: print('    p,tex,s,l,chRad =',p,tex,float(s)//2,int(l)//2,chRad)
            sch = float(s)*0.5
            #sch = str(int(sch) if float(sch).is_integer() else '%i/2' % int(2*float(sch)))
            lch = int(l)//2
            rr = '%s + %s' % (p,tex)
            thisChannel = resonanceReactions[rr]
            channelName = "%s width" % thisChannel.label
            jdx = 2
            while True:
                if channelName not in channelNames:
                    channelNames.append( channelName ); break
                channelName = '%s width_%d' % (thisChannel.label, jdx)
                jdx += 1
            width_units = 'MeV*{1/2}' if IFG==1 else 'MeV'
            sch = resolvedResonanceModule.spin( sch )
            columnHeaders.append( tableModule.columnHeader(chidx+1, name=channelName, unit= width_units) )
            channels.add( resolvedResonanceModule.channel(str(chidx), rr, columnIndex=chidx+1, L=lch,  channelSpin=sch) )

        IDList=[]
        for i in range(nvars):
            #print 'lJ,lPi,E,ir,s,l,ID,gamma,p,t,e2,chRad', vars[i]
            lJ,lPi,E,ir,s,l,ID,gamma,p,t,e2,chRad = vars[i]
            if lJ == J and lPi==pi: 
                if (ID,E) not in IDList: IDList.append((ID,E))
        if debug: print('        ID list =',IDList)

        resonances = []
        for term,e in IDList:
            
            values = [float(e)*cm2lab]   # to MeV
#
            if not ((RWA or RWAupdate) and IFG):
#         Scan to sum shifty_denom over all channels for given (term,e,J,pi)'
                shifty_sum = 0.0
                for i in range(nvars):
                    lJ,lPi,E,ir,sv,lv,ID,gamma,pv,tv,e2,chRad = vars[i]
                    try: ia = tLevels[tv].index(e2)
                    except: print("Level at ",e2," not found for ",tv)
                    texv = nuclideIDFromIsotopeSymbolAndIndex(tv,ia)
                    if J==lJ and pi==lPi and ID==term:
                        chidx = -1
                        for p,tex,s,l,prmax in channelList:
                            lch = int(l)//2
                            chidx += 1
                            if s==sv and l==lv and p==pv and tex==texv:
                                energy = float(E)   # from Azure in MeV
                                pMass,tMass,pZ,tZ,QI = ZAdict[ (p,tex) ]
                                pMass,tMass,pZ,tZ,QI = float(pMass),float(tMass),float(pZ),float(tZ),float(QI)
                                redmass = pMass*tMass/(pMass+tMass)
                                e_ch = energy + QI
                                penetrability,shift,dSdE,W = getCoulomb_PSdSW(
                                    e_ch,lch, float(chRad), pMass,tMass,pZ,tZ, fmscal,etacns, True)
                                # if debug: print  "i,lch,e_ch,QI:",chidx,lch,e_ch,QI," PSdS =",penetrability,shift,dSdE
                                if e_ch>0.0:
                                    GObs = abs(float(gamma)) * 1e-6   # convert to MeV since dSdE is per MeV
                                    shifty_sum += GObs *  dSdE / penetrability
                                    if verbose: print(10*' ',"e_ch,GObs",e_ch,GObs,'P,dSdE:',penetrability,dSdE, 'contrib:',0.5*GObs *  dSdE / penetrability)
                                else:
                                    ANC = float(gamma)           # in units of fm^{-1/2}
                                    if abs(ANC)>1e-20:
                                        if redmass<1e-10: print("Gamma bs problem. ANC,e_ch,W:",ANC,e_ch,W)
                                        T = 2.0 * W**2 / (fmscal * redmass * float(chRad))
                                        shifty_sum += ANC**2 * T *  dSdE
                                    if verbose: print(10*' ',"e_ch,ANC",e_ch,ANC,'W,P-eff,W^2*P,dSdE:',W,1./T,W*W/T,dSdE, 'contrib:',0.5*ANC**2 * T *  dSdE) 
                                    # if verbose: print "e_ch,ANC,dSdE,sum",e_ch,ANC,dSdE,shifty_sum, dSdE*shifty_sum,ANC**2 * T *  dSdE/2.
                                #if debug and abs(GObs)>1e-20: print 'Ssum=',shifty_sum,' from ', GObs ,  dSdE , penetrability,' at e_ch=',e_ch
                shifty_denom = 1.0/(1.0 - 0.5 * shifty_sum)
                if debug: print("Pole in Jpi =",J,pi," at ",e," has SD =",shifty_denom, ' from SS',shifty_sum,'\n')
            chidx = -1
            for p,tex,s,l,prmax in channelList:
                chidx += 1
                found = False
                for i in range(nvars):
                    lJ,lPi,E,ir,sv,lv,ID,gamma,pv,tv,e2,chRad = vars[i]
                    gamma = float(gamma)
                    try: ia = tLevels[tv].index(e2)
                    except: print("Level at ",e2," not found for ",tv)
                    texv = nuclideIDFromIsotopeSymbolAndIndex(tv,ia)
                    lch = int(l)//2

                    if J==lJ and pi==lPi and ID==term and s==sv and l==lv and p==pv and tex==texv:
                        energy = float(E)   # from Azure in MeV
                        pMass,tMass,pZ,tZ,QI = ZAdict[ (p,tex) ]
                        pMass,tMass,pZ,tZ,QI = float(pMass),float(tMass),float(pZ),float(tZ),float(QI)
                        e_ch = energy + QI
                        penetrability,shift,dSdE,W = getCoulomb_PSdSW(
                            e_ch,lch, float(chRad), pMass,tMass,pZ,tZ, fmscal,etacns, False)   # CWF at abs(e_ch) 
                        #if debug: print 'p,t =',p,tex,': call coulombPenetrationFactor(L=',lch,'rho=',rho,'eta=',eta,') =',penetrability,dSdE,W 
                        #   find gamma or Gamma_formal from the G_obs in the AZR input
                        #    Gamma_formal = G_obs * shifty_denom
                        #    gamma =  sqrt(Gamma_formal/(2*P))
                        if not RWA and not RWAupdate:
                            if e_ch > 0.0:
                                GObs = gamma * 1e-6 # G_obs in the AZR input in eV
                                Gammaf = GObs * shifty_denom   # units of eV from Azure
                                if IFG:         # GND to have rwa
                                    width = ( abs(Gammaf) /(2. * penetrability) ) **0.5  # units of eV**{1/2}
                                    if Gammaf < 0: width = -width
                                else:           # GND to have G_formal
                                    width = Gammaf                      #    in eV already
                            else:  # bound states
                                if debug: print('p,t =',p,tex,': P(L=',lch,'e=',e_ch,') =',penetrability,dSdE,W) 
                                if abs(gamma)>1e-20:
                                    redmass = pMass*tMass/(pMass+tMass)
                                    if shifty_denom<0.0: print(" gamma,denom<0:",gamma,shifty_denom,' for line ',i,' & redmass=',redmass)
                                    anc = gamma  * shifty_denom**0.5
                                    rwa = anc * W / (fmscal * redmass * float(chRad))**0.5 
                                    if debug: print(J,pi,": gamma",gamma,"becomes ANC",anc,"becomes rwa",rwa,'(IFG=',IFG,')')
                                else:
                                    rwa = 0.0

                                if IFG: 
                                    width = rwa
                                else:
                                    width = 2.0 * rwa**2 * penetrability    # Calculating formal width for energy |e_ch|                              
                                    if debug: print('rwa',gamma,'becomes Gamma',width)
                                    if gamma<0: width = -width
                                    #print 'Gobs,rwa, width =',rwa,width
                        else:  # RWA or RWAupdate: input already as rwa.
                            if IFG:         # GND to have rwa
                                width = gamma                      #    in MeV already                        
                            else:           # GND to have  G_formal
                                width = 2. * gamma**2 * penetrability  # gamma units of MeV**{1/2}
                                if gamma < 0: width = -width
                        w = width
                        width *= cm2lab**0.5 if IFG else cm2lab
                        values.append(width)
                        if debug: print("For p,t,s,l =",p,texv,s,l,": IFG,w,width",IFG,w,width,'(cm2lab=',cm2lab,')')
                        found = True
                if not found: values.append(0)
                #if debug: print values
            resonances.append(values)
        #if debug: print '   channelData :',channelData
        #if debug: print '   Resonance data :',resonances
        table = tableModule.table( columns=columnHeaders, data=resonances )
        J = resolvedResonanceModule.spin( J )
        pi= resolvedResonanceModule.spin( pi)
        spinGroups.add(    resolvedResonanceModule.spinGroup(str(spinGroupIndex), J, pi, channels,
                           resolvedResonanceModule.resonanceParameters(table)) )
        spinGroupIndex += 1

    RMatrix = resolvedResonanceModule.RMatrix( 'eval', resonanceReactions, spinGroups, approximation=approximation, boundaryCondition=BC,
                relativisticKinematics=KRL, reducedWidthAmplitudes=bool(IFG), 
                supportsAngularReconstruction=True, calculateChannelRadius=False )

    resolved = resolvedResonanceModule.resolved( emin,emax,'MeV' )
    resolved.add( RMatrix )

    Rm_radius = scatteringRadiusModule.scatteringRadius(
        constantModule.constant1d(Rm_global, domainMin=emin, domainMax=emax,
            axes=axesModule.axes(labelsUnits={1: ('energy_in', 'MeV'), 0: ('radius', 'fm')})) )
    scatteringRadius = Rm_radius
    unresolved = None
    resonances = resonancesModule.resonances( scatteringRadius, resolved, unresolved )
    gnd.resonances = resonances

    updated = '   updated from %s\n' % covFile if RWAupdate else ' '
    docLines = [' ','Converted from AZURE search file','   '+inFile,updated,time.ctime(),pwd.getpwuid(os.getuid())[4],' ',' ']
    doc = documentationModule.documentation( 'ENDL', '\n'.join( docLines ) )
    # gnd.styles[0].documentation['ENDL'] = doc

    if len(UPvalues)>0:
        npars = nvars + nLevels
        print("\nCovariance matrix for",nFitted,"varied out of ",npars)
        matrix = numpy.zeros([npars,npars])
        for i in range(nFitted):
            for j in range(nFitted):
                matrix[R2icor[i],R2icor[j]] = pMatrix[i,j]

     # store into GNDS (need links to each spinGroup)
        parameters = covarianceModelParametersModule.parameters()
        startIndex = 0
        for spinGroup in resonances.resolved.evaluated:
            nParams = spinGroup.resonanceParameters.table.nColumns * spinGroup.resonanceParameters.table.nRows
            if nParams == 0: continue
            parameters.add( covarianceModelParametersModule.parameterLink(
                label = spinGroup.label, link = spinGroup.resonanceParameters.table, root="$reactions",
                matrixStartIndex=startIndex, nParameters=nParams
            ))
            startIndex += nParams
        if debug: 
            print(parameters.toXML(),'\n')
            print(type(matrix))
            print('matrix:\n',matrix)
        if True:
            correlation = numpy.zeros([npars,npars])
            if debug: print("Cov shape",matrix.shape,", Corr shape",correlation.shape)
            # print "\nCov diagonals:",[matrix[i,i] for i in range(npars)]
            # print "\nCov diagonals:\n",numpy.array_repr(numpy.diagonal(matrix),max_line_width=100,precision=3)
            print("Diagonal uncertainties:\n",numpy.array_repr(numpy.sqrt(numpy.diagonal(matrix)),max_line_width=100,precision=4))
            for i in range(npars):
                for j in range(npars): 
                    t = matrix[i,i]*matrix[j,j]
                    if t !=0: correlation[i,j] = matrix[i,j] / t**0.5

            from scipy.linalg import eigh
            eigval,evec = eigh(matrix)
            if debug:
                print("  Covariance eigenvalue     Vector")
                for kk in range(npars):
                    k = npars-kk - 1
                    print(k,"%11.3e " % eigval[k] , numpy.array_repr(evec[:,k],max_line_width=200,precision=3, suppress_small=True))
            else:
                print("Covariance eivenvalues:\n",numpy.array_repr(numpy.flip(eigval[:]),max_line_width=100,precision=3))


            if False: print('correlation matrix:\n',correlation)
            eigval,evec = eigh(correlation)
            if debug:
                print("  Correlation eigenvalue     Vector")
                for kk in range(npars):
                    k = npars-kk - 1
                    print(k,"%11.3e " % eigval[k] , numpy.array_repr(evec[:,k],max_line_width=200,precision=3, suppress_small=True))
            else:
                print("Correlation eivenvalues:\n",numpy.array_repr(numpy.flip(eigval[:]),max_line_width=100,precision=3))

        GNDSmatrix = arrayModule.flattened.fromNumpyArray(matrix, symmetry=arrayModule.symmetryLowerToken)
        # print GNDSmatrix.toXML()
        Type="absoluteCovariance"
        covmatrix = covarianceModelParametersModule.parameterCovarianceMatrix('eval', GNDSmatrix,
            parameters, type=Type )
        if verbose: print(covmatrix.toXML())
        rowData = covarianceSectionModule.rowData(gnd.resonances.resolved.evaluated,
                root='')
        parameterSection = covarianceModelParametersModule.parameterCovariance("resolved resonances", rowData)
        parameterSection.add(covmatrix)

        p,tex = elastics
        covarianceSuite = covarianceSuiteModule.covarianceSuite( p, tex, 'EDA R-matrix covariances' )
        covarianceSuite.parameterCovariances.add(parameterSection)

        if debug: print(covarianceSuite.toXMLList())
        if verbose: covarianceSuite.saveToFile('CovariancesSuite.xml')

    else:
        if noCov: print("     Covariance data ignored")
        else:     print("     No covariance data found")
        covarianceSuite = None

    return gnd,covarianceSuite
