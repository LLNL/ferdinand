#
##############################################
#                                            #
#    Ferdinand 0.40, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################
from fudge import product as productModule
from fudge import outputChannel as channelsModule
from fudge.reactions import reaction as reactionModule
from fudge.reactionData import crossSection as crossSectionModule
from fudge.productData import multiplicity as multiplicityModule
from fudge.productData.distributions import unspecified as unspecifiedModule

from brownies.legacy.converting import toGNDSMisc
from pqu import PQU as PQUModule
import xData.standards as standardsModule
import xData.axes as axesModule
import xData.XYs as XYsModule

def zeroReaction(it,MT,QI, productList, process,emin,emax,energyUnit, v):

# make a zero background cross section for given MT channel over energy range [emin,emax]
    ENDF_Accuracy = 1e-3
    debug = v
    regionData = [ [emin,0.0], [emax,0.0]]   # Zero from emin to emax
  
    multiplicityAxes = multiplicityModule.defaultAxes( energyUnit )
    
    background = crossSectionModule.regions1d( axes = crossSectionModule.defaultAxes( energyUnit=energyUnit ))
    background.append( crossSectionModule.XYs1d( data=regionData,   axes=background.axes ) )
    # background.append( crossSection.XYs1d( data=fastRegionData,   axes=background.axes ) )  # not needed here.
    RRBack = crossSectionModule.resolvedRegion( background )
    background_ = crossSectionModule.background( RRBack, None, None )

    crossSection =  crossSectionModule.resonancesWithBackground( 'eval', 
#       crossSectionModule.resonanceLink(link = resonances),
        crossSectionModule.resonanceLink(path = "/reactionSuite/resonances"),
        background_ )
    multiplicity = multiplicityModule.constant1d( 1, domainMin=emin, domainMax=emax, axes=multiplicityAxes, label='eval' )

    Q = PQUModule.PQU( QI, energyUnit )
    reaction = reactionModule.reaction( channelsModule.Genre.twoBody, ENDF_MT = MT)
    outputChannel = reaction.outputChannel
    if process is not None: outputChannel.process = process
    outputChannel.Q.add( toGNDSMisc.returnConstantQ( 'eval', QI , crossSection) )

    frame = standardsModule.frames.centerOfMassToken
    form = unspecifiedModule.form( 'eval', productFrame = frame )

    for particle in productList:
        product = productModule.product( particle.id )
        product.multiplicity.add( multiplicity )
        product.distribution.add( form )
        outputChannel.products.add( outputChannel.products.uniqueLabel( product ) )

    reaction.crossSection.add( crossSection )

    return reaction
