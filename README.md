# A Tutorial on Fitting Time-Series With CARMA Models

A few prerequesites: you will need some code and libraries in order to follow along.

 * [Julia](http://julialang.org): an innovative Matlab-like language with C-like speeds, developed at MIT.
 * [Gadfly](http://dcjones.github.io/Gadfly.jl/): a Julia plotting library.  Once you have Julia, you can install gadfly using the built-in package manager.  Open julia, and type `Pkg.add("gadfly").
 * [Ensemble](https://github.com/farr/Ensemble.jl): a set of Julia libraries for sampling from probability distributions, loosely organised around the "affine invariant" algorithm underlying [emcee](http://dan.iel.fm/emcee/current/user/install/) and described by [Goodman and Weare](http://www.cims.nyu.edu/~weare/papers/d13.pdf).  `Ensemble` is not yet a part of the standard Julia package set, so you will have to install it directly from GitHub: `Pkg.clone("https://github.com/farr/Ensemble.jl.git")`.
 * [IJulia](https://github.com/JuliaLang/IJulia.jl): to allow the use of [Juypter](http://jupyter.org) (formerly iPython) notebooks with Julia.
