# Ferdinand: Translate R-matrix Evaluations 
   Version 0.40

   Release: LLNL-CODE-831538

####  Ian Thompson

   Email: thompson97@llnl.gov


## Needed libraries

Users to download:  
fudge version 5.0 from [github.com/LLNL/fudge](https://github.com/LLNL/fudge) 

Already included:  
f90nml modified from  [github.com/marshallward/f90nml](https://github.com/marshallward/f90nml) 
  
For reconstructTensorFlow.py and reconstructxs_TF.py, users to download:  
tensorflow from [www.tensorflow.org/install](https://www.tensorflow.org/install).  
For macs see [developer.apple.com/metal/tensorflow-plugin/](https://developer.apple.com/metal/tensorflow-plugin/) 

## Ferdinand

```
usage: ferdinand.py [-h] [-c COVFILE] [--noCov] [-i in-form] [-o outFile] [-v]
                    [-d] [-w] [-W] [-L LVALS [LVALS ...]] [-e ELASTIC]
                    [-l Emin] [-u Emax] [-D Edist] [-B Ebound] [-b B] [-a]
                    [-G] [-E new_label] [-Q] [-g] [-R REICHMOORE] [-r]
                    [-f FILTER] [--nocm] [-A Eadjust] [-F Efile] [-z]
                    [-n NONZERO] [-V Eunit] [-6] [--lineNumbers] [-p dE]
                    [-P Ang Ang] [-t dE] [-M] [-C] [-S] [--CN CN CN]
                    inFile finalformat

```
## Translate R-matrix Evaluations

#### positional arguments:
```
  inFile                The input file you want to translate. Formats: fresco,
                        sfresco, eda, amur, rac, endf, azure, gnd=gnds=xml, ..
  finalformat           Output source format: fresco, sfresco, eda, hyrma,
                        endf, azure, gnd=gnds=xml, tex.
```

#### optional arguments:
```
  -h, --help            show this help message and exit
  -c COVFILE, --covFile COVFILE
                        Input file with covariance matrix
  --noCov               Ignore input covariance matrices
  -i in-form, --initial in-form
                        Input source format: endf, gnd=gnds=xml=gnds.xml,
                        fresco, eda, amur, apar, rac, sfresco, sfrescoed,
                        hyrma, azure, ... This is expected suffix of input
                        file
  -o outFile            Specify the output file. Otherwise use ``inFile`` with
                        expected suffix removed if present.
  -v, --verbose         Verbose output
  -d, --debug           Debugging output (more than verbose)
  -w, --rwa             Reading first makes GNDS with reducedWidthAmplitudes
  -W, --RWA             When reading azure files, amplitudes are already as
                        reduced width amplitudes, and B=-L.
  -L LVALS [LVALS ...], --Lvals LVALS [LVALS ...]
                        When reading fresco files, or writing EDA files, set
                        partial waves up to this list value in each pair.
  -e ELASTIC, --elastic ELASTIC
                        ResonanceReaction label of elastic particle-pair in
                        input file
  -l Emin, --lower Emin
                        Lower energy of R-matrix evaluation
  -u Emax, --upper Emax
                        Upper energy of R-matrix evaluation
  -D Edist, --Distant Edist
                        Pole energy above which are all distant poles, to help
                        with labeling. Fixed in sfresco searches.
  -B Ebound, --Bound Ebound
                        Pole energy below which are all bound poles, to help
                        with labeling. Fixed in sfresco searches.
  -b B, --boundary B    Boundary condition in output: 'Brune'; '-L' or 'L' for
                        B=-L; or 'X' for B=float(X).
  -a, --amplitudes      Convert intermediate gnd file stores to reduced width
                        amplitudes, not widths. If not -a or -G, leave
                        unchanged.
  -G, --Gammas          Convert intermediate gnd file stores to formal widths,
                        not reduced width amplitudes. Overrides -a.
  -E new_label, --Elastic new_label
                        ResonanceReaction label of new elastic particle-pair
                        after transforming input.
  -Q, --Q               Allow elastic Q values to be non-zero.
  -g, --nogamma         Omit gamma channels
  -R REICHMOORE, --ReichMoore REICHMOORE
                        Add a Reich-Moore gamma channel with this value
  -r, --noreac          Omit all nonelastic (reaction) channels
  -f FILTER, --filter FILTER
                        Filter of csv list of particle-pair-labels to include.
                        Overrides -g,-r options
  --nocm                No incoming transformations of cm-lab pole energies:
                        for old mistaken files
  -A Eadjust, --Adjust Eadjust
                        Adjust pole energies: give arithmetic function of E,
                        such as 'E+5000 if E>2e6 else E'. Applied after any
                        Barker but before any Brune transformations
  -F Efile, --File Efile
                        Data file for reading R-matrix data
  -z, --zero            Omit zero widths
  -n NONZERO, --nonzero NONZERO
                        Replace zero widths by this value.
  -V Eunit, --Volts Eunit
                        Energy units for conversion after making gnds, before
                        output conversions. Not checked.
  -6, --p6              Limit energies and widths to ENDF6 precision.
  --lineNumbers         Add line numbers to ENDF6 format files
  -p dE, --pointwise dE
                        Reconstruct angle-integrated cross sections using
                        Fresco for given E step
  -P Ang Ang, --Pointwise Ang Ang
                        Reconstruct also angle-dependent cross sections, given
                        thmin, thinc (in deg)
  -t dE, --tf dE        Reconstruct angle-integrated cross sections using
                        TensorFlow for given E step
  -M, --Ecm             Print poles in latex table in CM energy scale of
                        elastic channel.
  -C, --Comp            Print poles in latex table in CM energy scale of
                        compound nucleus.
  -S, --Squeeze         Squeeze table of printed poles in latex
  --CN CN CN            Spin and parity of compound nucleus, if needed
```

## Standalone codes


```
usage: reconstructxs_TF.py [-h] [-M] [-D DIAGONALONLY] [-S SCALE [SCALE ...]]
                           [--dE DE] [-E EMAX] [-s STRIDE] [-L LMAX] [-H] [-w]
                           [-t] [-G] [-X] [-A AVERAGING] [-C CONVOLUTE] [-v]
                           [-d]
                           inFiles [inFiles ...]
```

Pointwise reconstruction of R-matrix excitation functions on a grid started using resonance positions. 
No angular distributions.
Method using tensorflow for GPUs (otherwise on CPUs)

#### positional arguments:

```
inFiles               The input file you want to pointwise expand.
```

#### optional arguments:

```
  -h, --help            show this help message and exit
  -M, --MatrixL         Use level matrix method if not already Brune basis
  -D DIAGONALONLY, --DiagonalOnly DIAGONALONLY
                        Model S(SLBW) or M(MLBW) for diagonal-only level matrix
  -S SCALE [SCALE ...], --Scale SCALE [SCALE ...]
                        Scale all amplitudes by factor
  --dE DE               Energy step for uniform energy grid, in MeV
  -E EMAX, --EMAX EMAX  Maximum Energy (MeV)
  -s STRIDE, --stride STRIDE
                        Stride for accessing non-uniform grid template
  -L LMAX, --LMAX LMAX  Max partial wave L
  -H, --HardSphere      Without hard-sphere phase shift
  -w, --write           Write cross-sections in GNDS file
  -t, --thin            Thin distributions in GNDS form
  -G, --Global          print output excitation functions in cm energy of GNDS
                        projectile
  -X, --Xcited          calculate and print output for excited initial states
  -A AVERAGING, --Averaging AVERAGING
                        Averaging width to all scattering: imaginary =
                        Average/2.
  -C CONVOLUTE, --Convolute CONVOLUTE
                        Gaussian width to convolute excitation functions
  -v, --verbose         Verbose output
  -d, --debug           Debugging output (more than verbose)
```

