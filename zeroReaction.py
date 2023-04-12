#
##############################################
#                                            #
#    Ferdinand 0.50, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################
from fudge import product as productModule
from fudge import outputChannel as outputChannelModule
from fudge import enums as enumsModule
from fudge.reactions import reaction as reactionModule
from fudge.reactionData import crossSection as crossSectionModule
from fudge.productData import multiplicity as multiplicityModule
from fudge.productData.distributions import unspecified as unspecifiedModule

from brownies.legacy.converting import toGNDSMisc
from pqu import PQU as PQUModule
import xData.standards as standardsModule
import xData.axes as axesModule
import xData.XYs1d as XYsModule
from xData import enums as xDataEnumsModule

def zeroReaction(it,MT,QI, productList, process,emin,emax,energyUnit, v):

# make a zero background cross section for given MT channel over energy range [emin,emax]
    ENDF_Accuracy = 1e-3
    regionData = [ [emin,0.0], [emax,0.0]]   # Zero from emin to emax
    styleLabel = 'eval'
  
    multiplicityAxes = multiplicityModule.defaultAxes( energyUnit )
    
    background = crossSectionModule.Regions1d( axes = crossSectionModule.defaultAxes( energyUnit=energyUnit ))
    background.append( crossSectionModule.XYs1d( data=regionData,   axes=background.axes ) )
    # background.append( crossSection.XYs1d( data=fastRegionData,   axes=background.axes ) )  # not needed here.
    RRBack = crossSectionModule.ResolvedRegion( background )
    background_ = crossSectionModule.Background( RRBack, None, None )

    crossSection =  crossSectionModule.ResonancesWithBackground( styleLabel, 
#       crossSectionModule.ResonanceLink(link = resonances),
        crossSectionModule.ResonanceLink(path = "/reactionSuite/resonances"),
        background_ )
    multiplicity = multiplicityModule.Constant1d( 1, domainMin=emin, domainMax=emax, axes=multiplicityAxes, label=styleLabel )

    Q = PQUModule.PQU( QI, energyUnit )
    reaction = reactionModule.Reaction( None, enumsModule.Genre.twoBody,  MT)
    outputChannel = reaction.outputChannel
    if process is not None: outputChannel.process = process
    outputChannel.Q.add( toGNDSMisc.returnConstantQ( styleLabel, QI , crossSection) )

    frame = xDataEnumsModule.Frame.centerOfMass
    form = unspecifiedModule.Form( styleLabel, productFrame = frame )

    for particle in productList:
        product = productModule.Product( particle.id, particle.id )
        product.multiplicity.add( multiplicity )
        product.distribution.add( form )
        outputChannel.products.add( outputChannel.products.uniqueLabel( product ) )

    reaction.crossSection.add( crossSection )

    return reaction
