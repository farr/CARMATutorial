# A Tutorial on Fitting Time-Series With CARMA Models

A few prerequesites: you will need some code and libraries in order to follow along.

 * [Julia](http://julialang.org): an innovative Matlab-like language with C-like speeds, developed at MIT.
 * [Ensemble](https://github.com/farr/Ensemble.jl): a set of Julia libraries for sampling from probability distributions, loosely organised around the "affine invariant" algorithm underlying [emcee](http://dan.iel.fm/emcee/current/user/install/) and described by [Goodman and Weare](http://www.cims.nyu.edu/~weare/papers/d13.pdf).  `Ensemble` is not yet a part of the standard Julia package set, so you will have to install it directly from GitHub: `Pkg.clone("https://github.com/farr/Ensemble.jl.git")`.
 * [CARMA](https://github.com/farr/CARMA.jl): a set of Julia libraries for fitting CARMA models.  See [Kelly et al. (2014)](https://ui.adsabs.harvard.edu/#abs/2014ApJ...788...33K/abstract) for the methods.
 * The Julia packages `PyPlot` and `PyCall` and the Python package `seaborn` to plot the results of your fits.
 * [IJulia](https://github.com/JuliaLang/IJulia.jl): to allow the use of [Juypter](http://jupyter.org) (formerly iPython) notebooks with Julia.  `Pkg.add("IJulia")`.

## More Information

For more information on CARMA models, read [Kelly, et al (2014)](http://arxiv.org/abs/1402.5978) or the forthcoming Barrett & Farr (in prep).  There is a similar package, [carma_pack](https://github.com/brandonckelly/carma_pack), associated with Kelly, et al (2014), but we have found it to be sometimes buggy.
