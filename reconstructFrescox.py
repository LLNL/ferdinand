##############################################

#                                            #
#    Ferdinand 0.50, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################

import os 
import math
from write_fresco import write_fresco
import fudge.sums as sumsModule
import fudge.styles as stylesModule
import fudge.reactionData.crossSection as crossSectionModule
import fudge.productData.distributions as distributionsModule

##############################################  write_fresco

def reconstructFrescox(gnd,base,verbose,debug,egrid,angles,thin,reconstyle):

    projectile = gnd.PoPs[gnd.projectile]
    target     = gnd.PoPs[gnd.target]
    if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
    if hasattr(target, 'nucleus'):     target = target.nucleus
    pZ = projectile.charge[0].value; tZ =  target.charge[0].value
    charged =  pZ*tZ != 0
    elasticChannel = gnd.getReaction('elastic')
    identicalParticles = gnd.projectile == gnd.target
    rStyle = reconstyle.label
    if debug: print("Charged-particle elastic:",charged,",  identical:",identicalParticles,' rStyle:',rStyle)

    if charged and angles is not None:
        from fudge.reactionData.doubleDifferentialCrossSection.chargedParticleElastic import CoulombPlusNuclearElastic as  CoulombPlusNuclearElasticModule
        from fudge.reactionData.doubleDifferentialCrossSection.chargedParticleElastic import nuclearPlusInterference as nuclearPlusInterferenceModule
#        from fudge.reactionData.doubleDifferentialCrossSection.chargedParticleElastic import RutherfordScattering as RutherfordScatteringModule
        from fudge.productData.distributions import reference as referenceModule

        thmin = angles[0]
        pi = 3.1415826536
        muCutoff = math.cos(thmin*pi/180.)

    fresco_base = base + '.fresco_recon'
    channels = write_fresco(gnd,fresco_base,verbose,debug,True,None,None,False,egrid,angles)
    name_frin = fresco_base + '.frin'   # must be same as in write_fresco
    name_frout= fresco_base + '.frout'  
    accuracy = None
    
    cmd = "frescox < "+name_frin+" > "+name_frout
    print(cmd)
    
    os.system(cmd)   # Run FRESCO

    f239 = open('fort.239','r')
    egrid = []
    totalxs = []; elasticxs = []; fissionxs = []; absorbtionxs = []
    chanxs =[]; 
    # lastzero = [ 0 for i in range(len(channels))]
    for rreac in gnd.resonances.resolved.evaluated.resonanceReactions:
        if not rreac.eliminated:
            chanxs.append([])
    if len(channels) != len(chanxs):
        print("Only getting",channels," data channels, not",len(chanxs))
        exit()
    if debug: print("Fresco channel order:",channels)

    mb = 1e-3
    for line  in f239:
        if 'NaN' not in line:
            data = line.split()
            try:
                elab,absorbtion,reaction,total,elastic = [float(d) for d in data[:5]]
                sigr =  [float(d) for d in data[5:]]
                #print elab,absorbtion,reaction,total,elastic,sigr
                egrid.append(elab)
                totalxs.append(total*mb)
                elasticxs.append(elastic*mb)
                fissionxs.append(0.0)
                absorbtionxs.append(absorbtion*mb)
                for c in range(len(channels)):
                    chanxs[c].append(sigr[c]*mb)
                    # if sigr[c]== 0.: lastzero[c] = elab
            except:
                pass

    crossSectionAxes = crossSectionModule.defaultAxes( 'MeV' )
    total = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, totalxs), dataForm="XsAndYs" )
    elastic = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, elasticxs), dataForm="XsAndYs" )
    fission = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, fissionxs), dataForm="XsAndYs" )
    absorbtion = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, absorbtionxs), dataForm="XsAndYs" )

    if not isinstance( reconstyle, stylesModule.CrossSectionReconstructed ):
        raise TypeError("style must be an instance of crossSectionReconstructed, not %s" % type(reconstyle))

    haveEliminated = False
    for rreac in gnd.resonances.resolved.evaluated.resonanceReactions:
        reaction = rreac.link.link
        haveEliminated = haveEliminated or rreac.eliminated
        #                  elastic or capture 
        if reaction == gnd.getReaction('capture'): rreac.tag = 'capture'
        elif reaction == gnd.getReaction('elastic'): rreac.tag = 'elastic'
        elif 'fission' in rreac.label: rreac.tag = rreac.label
        else: rreac.tag = 'competitive'
                
    xsecs = {'total':total, 'elastic':elastic, 'fission':fission, 'nonelastic':absorbtion}
    for c in range(len(channels)):  # skip elastic 
        if channels[c] != elasticChannel.label:     # skip elastic 
            xsecs[channels[c]] = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, chanxs[c]), dataForm="XsAndYs" )

    if haveEliminated:
        eliminatedReaction = [rr for rr in gnd.resonances.resolved.evaluated.resonanceReactions if rr.eliminated]
        if len(eliminatedReaction) != 1:
            raise TypeError("Only 1 reaction can be eliminated in Reich-Moore approximation!")
        xsecs[eliminatedReaction[0].tag] = absorbtion - fission
                
    epsilon = 1e-8  # for joining multiple regions together

    # for each reaction, add tabulated pointwise data (ENDF MF=3) to reconstructed resonances:
    possibleChannels = { 'elastic' : True, 'capture' : True, 'fission' : True, 'total' : False, 'nonelastic' : False }
    elasticChannel = gnd.getReaction('elastic')
    derivedFromLabel = ''
    for reaction in gnd :
        if isinstance( reaction, sumsModule.MultiplicitySum ): continue
        iselastic = reaction is elasticChannel

        evaluatedCrossSection = reaction.crossSection.evaluated
        if not isinstance( evaluatedCrossSection, crossSectionModule.ResonancesWithBackground ):
            continue
        # which reconstructed cross section corresponds to this reaction?
        if( derivedFromLabel == '' ) : derivedFromLabel = evaluatedCrossSection.label
        if( derivedFromLabel != evaluatedCrossSection.label ) :
            print(('WARNING derivedFromLabel = "%s" != "%s"' % (derivedFromLabel, evaluatedCrossSection.label)))
        RRxsec = None
        if str( reaction ) in xsecs:
            RRxsec = xsecs[ str( reaction ) ]
            # print 'Assign to ',str(reaction),'\n',RRxsec.toString()
        else :
            for possibleChannel in possibleChannels :
                if( possibleChannels[possibleChannel] ) :
                    if( possibleChannel in str( reaction ) ) : 
                        RRxsec = xsecs[possibleChannel]
                        # print 'Assign to ',str(reaction),'\n',RRxsec.toString()
                if( RRxsec is None ) :
                    if( reaction is gnd.getReaction( possibleChannel ) ) : 
                        RRxsec = xsecs[possibleChannel]
                        # print 'Assign to ',str(reaction),'\n',RRxsec.toString()
                if( RRxsec is not None ) : break
        if( RRxsec is None ) :
            if True:
                print(( "Warning: couldn't find appropriate reconstructed cross section to add to reaction %s" % reaction ))
            continue

        background = evaluatedCrossSection.background
        background = background.toPointwise_withLinearXYs( accuracy = 1e-3, lowerEps = epsilon, upperEps = epsilon )
        RRxsec = RRxsec.toPointwise_withLinearXYs( accuracy = 1e-3, lowerEps = epsilon, upperEps = epsilon )
        RRxsec.convertUnits( {RRxsec.domainUnit: background.domainUnit,  RRxsec.rangeUnit: background.rangeUnit } )

        background, RRxsec = background.mutualify(0,0,0, RRxsec, -epsilon,epsilon,True)
        RRxsec = background + RRxsec    # result is a crossSection.XYs1d instance
        if thin:
            RRx = RRxsec.thin( accuracy or .001 )
        else:
            RRx = RRxsec
        RRx.label = rStyle

        reaction.crossSection.add( RRx )
       
        # print "Channels ",reaction.label,iselastic,":\n",RRxsec.toString(),"\n&\n",RRx.toString()
        if iselastic:
            effXsc = RRxsec

    gnd.styles.add( reconstyle )
#     print "Last energies of zero cross section:",lastzero

    if angles is None: return

    f241 = open('fort.241','r')
    sigdd = {}
    for rr in channels: sigdd[rr] = []
    
    for line  in f241:
        if '# Elab =' in line:
            elab,ich = float(line[9:9+15]),int(line[9+15:9+15+4])-1    # Elab = 1.00000000E-06   1
            line1 = line
            dist = []
        elif "&" in line:
            rr = channels[ich]
            sigdd[rr].append([elab,dist])
            # if elab<1.0001: print '\n',ich,rr,sigdd[rr]
        elif "NaN" in line:
            continue
        else:
            mu,p = line.split()
            try:
                mu,p = float(mu),float(p)
                dist.insert(0,p)
                dist.insert(0,mu)
            except:
                pass

    angularAxes = distributionsModule.angular.defaultAxes( 'MeV' )
    
    for rreac in gnd.resonances.resolved.evaluated.resonanceReactions:
        if not rreac.eliminated:
            productName = rreac.ejectile
            residName   = rreac.residual
            elastic = productName == gnd.projectile and residName == gnd.target
            print("Add angular distribution for",productName," in",rreac.label,"channel (elastic=",elastic,")")

            reaction = rreac.link.link
            firstProduct = reaction.outputChannel.getProductWithName(productName)

            effDist = distributionsModule.angular.XYs2d( axes = angularAxes )

            elab_max = 0.; elab_min = 1e10; nangles=0
            ne = 0
            for elab,dist in sigdd[rreac.label]:
                if debug: print('E=',elab,'has',len(dist),' angles')
                angdist = distributionsModule.angular.XYs1d( data = dist, outerDomainValue = elab, axes = angularAxes, dataForm = 'list' ) 
                if thin:
                    angdist = angdist.thin( accuracy or .001 )
                norm = angdist.integrate()
                if norm != 0.0:
                    if debug: print(rreac.label,elab,norm)
                    effDist.append( angdist ) 
                elab_max = max(elab,elab_max); elab_min = min(elab,elab_min); nangles = max(len(dist),nangles)
                ne += 1
            print("   Angles reconstructed at %i energies from %s to %s MeV with up to %i angles at each energy" % (ne,elab_min,elab_max,nangles))

            newForm = distributionsModule.angular.TwoBody( label = reconstyle.label,
                productFrame = firstProduct.distribution.evaluated.productFrame, angularSubform = effDist )
            firstProduct.distribution.add( newForm )

            if elastic and charged:   #    dCrossSection_dOmega for charged-particle elastics:
   
                NCPI = nuclearPlusInterferenceModule.NuclearPlusInterference( muCutoff=muCutoff,
                        crossSection=nuclearPlusInterferenceModule.CrossSection( effXsc),
                        distribution=nuclearPlusInterferenceModule.Distribution( effDist)
                        )
#                Rutherford = RutherfordScatteringModule.RutherfordScattering()

                CoulombElastic = CoulombPlusNuclearElasticModule.Form( gnd.projectile, rStyle, nuclearPlusInterference = NCPI, identicalParticles=identicalParticles )
                reaction.doubleDifferentialCrossSection.add( CoulombElastic )
    
                reaction.crossSection.remove( rStyle )
                reaction.crossSection.add( crossSectionModule.CoulombPlusNuclearElastic( link = reaction.doubleDifferentialCrossSection[rStyle],
                    label = rStyle, relative = True ) )
                firstProduct.distribution.remove( rStyle )
                firstProduct.distribution.add( referenceModule.CoulombPlusNuclearElastic( link = reaction.doubleDifferentialCrossSection[rStyle],
                    label = rStyle, relative = True ) )

            secondProduct = reaction.outputChannel[1]
            # secondProduct.distribution[rStyle].angularSubform.link = firstProduct.distribution[rStyle]    ## Fails
            # give 'recoil' distribution!
    return 
