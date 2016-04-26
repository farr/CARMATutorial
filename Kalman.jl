module Kalman

using Base.LinAlg: PosDefException

export AR1KalmanFilter, reset!, advance!, observe!, generate, whiten, log_likelihood

type AR1KalmanFilter
    ypred::Float64
    vypred::Float64
    mu::Float64
    var::Float64
    tau::Float64
end

""" Produce a Kalman filter for an AR(1) process with the given mean,
variance, and timescale. """
function AR1KalmanFilter(mu, var, tau)
    AR1KalmanFilter(mu, var, mu, var, tau)
end

""" Reset the given filter to its initial state.  """
function reset!(filt::AR1KalmanFilter)
    filt.ypred = filt.mu
    filt.vypred = filt.var
end

""" Advance the given filter by a step `dt`.  """
function advance!(filt::AR1KalmanFilter, dt::Float64)
    lfac = exp(-dt/filt.tau)

    filt.ypred = filt.mu + (filt.ypred - filt.mu)*lfac
    filt.vypred = lfac*(filt.vypred - filt.var)*lfac + filt.var
end

""" Incorporate the given observation with measurement uncertainty (1
s.d.) into the filter estimate. """
function observe!(filt::AR1KalmanFilter, y::Float64, dy::Float64)
    vy = dy*dy + filt.vypred

    gain = filt.vypred / vy

    filt.ypred = filt.ypred + (y - filt.ypred)*gain
    filt.vypred = filt.vypred - gain*gain*vy
end

""" Generate the GP represented by the given filter, sampled at the
given times with the given observational uncertainties.  """
function generate(filt::AR1KalmanFilter, ts::Array{Float64,1}, dys::Array{Float64,1})
    reset!(filt)
    
    n = size(ts, 1)
    ys = zeros(n)

    ytrue = sqrt(filt.var)*randn() + filt.mu
    ys[1] = ytrue + dys[1]*randn()
    
    for i in 2:n
        observe!(filt, ytrue, 0.0)
        advance!(filt, ts[i]-ts[i-1])
        ytrue = filt.ypred + sqrt(filt.vypred)*randn()
        ys[i] = ytrue + dys[i]*randn()
    end

    ys
end

""" Whiten the observations assuming they are drawn from the GP
represented by the given filter.  The outputs will be independent
N(0,1) distributed. """
function whiten(filt::AR1KalmanFilter, ts::Array{Float64,1}, ys::Array{Float64, 1}, dys::Array{Float64,1})
    n = size(ts, 1)

    xs = zeros(n)

    reset!(filt)

    xs[1] = (ys[1] - filt.ypred)/sqrt(filt.vypred + dys[1]*dys[1])
    for i in 2:n
        observe!(filt, ys[i-1], dys[i-1])
        advance!(filt, ts[i]-ts[i-1])
        xs[i] = (ys[i] - filt.ypred) / sqrt(filt.vypred + dys[i]*dys[i])
    end

    xs
end

square(x) = x*x

function log_likelihood_term(filt, y, dy)
    var = filt.vypred + dy*dy
    -0.5*log(2.0*pi) - 0.5*log(var) - 0.5*square(y-filt.ypred)/var
end

""" The log-likelihood function for the given data assuming that it
comes from the GP represented by the filter. """
function log_likelihood(filt::AR1KalmanFilter, ts::Array{Float64,1}, ys::Array{Float64,1}, dys::Array{Float64,1})
    n = size(ts, 1)

    ll = zero(ys[1])
    
    reset!(filt)

    ll += log_likelihood_term(filt, ys[1], dys[1])
    for i in 2:n
        observe!(filt, ys[i-1], dys[i-1])
        advance!(filt, ts[i]-ts[i-1])
        ll += log_likelihood_term(filt, ys[i], dys[i])
    end

    ll
end

function psd(filt::AR1KalmanFilter, fs::Array{Float64, 1})
    4.0*filt.tau ./ (1.0 + (2.0*pi*filt.tau*fs).^2)
end

type CARMAKalmanFilter
    mu::Float64
    sig::Float64
    x::Array{Complex128,1}
    vx::Array{Complex128, 2}
    v::Array{Complex128, 2}
    arroots::Array{Complex128, 1}
    b::Array{Complex128, 2}
    vxtemp::Array{Complex128, 2}
    xtemp::Array{Complex128, 1}
    lambda::Array{Complex128, 1}
end

function reset!(filt::CARMAKalmanFilter)
    p = size(filt.x, 1)

    filt.x = zeros(Complex128, p)
    filt.vx = copy(filt.v)

    filt
end

"""Construct a polynomial from the given roots.  Returns an array of
coefficients, `c`, represting the polynomial as `p(x) =
sum(c[i]*x^(i-1))`.

"""
function poly{T <: Number}(roots::Array{T,1})
    n = size(roots, 1) + 1

    if n == 1
        return T[one(T)]
    else
        poly = zeros(T, n)

        poly[2] = one(T)
        poly[1] = -roots[1]

        for i in 2:size(roots,1)
            r = roots[i]
            for j in n:-1:2
                poly[j] = poly[j-1] - r*poly[j]
            end
            poly[1] = -poly[1]*r
        end

        poly
    end
end

function polyeval{T <: Number}(roots::Array{T, 1}, x::T)
    p = one(x)

    for i in 1:size(roots,1)
        p = p*(x - roots[i])
    end

    p
end

function CARMAKalmanFilter(mu::Float64, sigma::Float64, arroots::Array{Complex128, 1}, maroots::Array{Complex128, 1})
    p = size(arroots, 1)
    q = size(maroots, 1)

    @assert q < p "q must be less than p: q = $q, p = $p"
    @assert all(real(arroots) .< 0.0) "AR roots must have negative real part: $arroots"
    @assert all(real(maroots) .< 0.0) "MA roots must have negative real part: $maroots"

    beta = poly(maroots)
    beta /= beta[1]
    b = cat(1, beta, zeros(p-q-1))
    b = b'

    U = zeros(Complex128, (p,p))
    for j in 1:p
        for i in 1:p
            U[i,j] = arroots[j]^(i-1)
        end
    end

    # Rotated observation vector
    b = b*U

    e = zeros(Complex128, p)
    e[end] = one(Complex128)

    J = U \ e

    V = zeros(Complex128, (p,p))
    for j in 1:p
        for i in 1:p
            V[i,j] = -J[i]*conj(J[j])/(arroots[i] + conj(arroots[j]))
        end
    end

    s2 = sigma*sigma/(b*V*b')[1]
    V = V*s2

    sig = sqrt(real(s2))

    CARMAKalmanFilter(mu, sig, zeros(Complex128, p), V, copy(V), copy(arroots), b, zeros(Complex128, (p,p)), zeros(Complex128, p), zeros(Complex128, p))
end

function advance!(filt::CARMAKalmanFilter, dt::Float64)
    p = size(filt.x, 1)

    for i in 1:p
        filt.lambda[i] = exp(filt.arroots[i]*dt)
    end
    lam = filt.lambda

    for i in 1:p
        filt.xtemp[i] = lam[i]*filt.x[i]
    end

    for j in 1:p
        for i in 1:p
            filt.vxtemp[i,j] = lam[i]*conj(lam[j])*filt.vx[i,j] + filt.v[i,j]*(one(lam[i])-lam[i]*conj(lam[j]))
        end
    end

    for j in 1:p
        filt.x[j] = filt.xtemp[j]
        for i in 1:p
            filt.vx[i,j] = filt.vxtemp[i,j]
        end
    end
    
    filt
end

function observe!(filt::CARMAKalmanFilter, y::Float64, dy::Float64)
    p = size(filt.x, 1)

    y = y - filt.mu

    dy2 = dy*dy

    # The complicated series below reproduces the Morrison-Woodbury
    # formula for the update step for V:
    #
    # V -> (V^{-1} + b^* b / dy2)^{-1}
    #
    # SMW:
    # V -> V - V b^* (dy2 + b V b^*)^{-1} b V

    c = complex(dy2)
    for i in 1:p
        for j in 1:p
            c += filt.b[1,i]*filt.vx[i,j]*conj(filt.b[1,j])
        end
    end
    c = one(c)/c

    for i in 1:p
        filt.lambda[i] = zero(filt.lambda[i])
        filt.xtemp[i] = zero(filt.xtemp[i])
        for j in 1:p
            filt.lambda[i] += filt.vx[i,j]*conj(filt.b[1,j])
            filt.xtemp[i] += filt.b[1,j]*filt.vx[j,i]
        end
    end
    
    for i in 1:p
        for j in 1:p
            filt.vxtemp[i,j] = filt.vx[i,j] - filt.lambda[i]*filt.xtemp[j]*c
        end
    end

    # Now we work out the algebra for
    #
    # x -> Vnew b* y/dy2 + Vnew V^{-1} x
    #
    # or
    #
    # x -> Vnew b* y/dy2 + x - V b* b x / c

    bdx = zero(filt.b[1,1])
    for i in 1:p
        bdx += filt.b[1,i]*filt.x[i]
    end

    s = y/dy2
    for i in 1:p
        filt.x[i] = filt.x[i] - filt.lambda[i]*bdx * c
        for j in 1:p
            filt.x[i] += filt.vxtemp[i,j]*conj(filt.b[1,j])*s
            filt.vx[i,j] = filt.vxtemp[i,j]
        end
    end

    filt
end

function predict(filt::CARMAKalmanFilter)
    p = size(filt.x,1)
    
    yp = filt.mu
    for i in 1:p
        yp += real(filt.b[1,i]*filt.x[i])
    end

    vyp = 0.0
    for i in 1:p
        for j in 1:p
            vyp += real(filt.b[1,i]*filt.vx[i,j]*conj(filt.b[1,j]))
        end
    end

    yp, vyp
end

function whiten(filt::CARMAKalmanFilter, ts, ys, dys)
    n = size(ts, 1)

    reset!(filt)
    zs = zeros(n)

    for i in 1:n
        yp, vyp = predict(filt)

        zs[i] = (ys[i] - yp)/sqrt(vyp + dys[i]*dys[i])

        observe!(filt, ys[i], dys[i])

        if i < n
            advance!(filt, ts[i+1]-ts[i])
        end
    end

    zs
end

function generate(filt::CARMAKalmanFilter, ts, dys)
    n = size(ts, 1)
    nd = size(filt.x, 1)

    ys = zeros(n)

    reset!(filt)
    
    for i in 1:n
        # Draw a new state
        L = chol(filt.vx, Val{:L})
        filt.x = filt.x + L*randn(nd)
        filt.vx = zeros(Complex128, (nd, nd))

        y, _ = predict(filt)

        ys[i] = y + dys[i]*randn()

        if i < n
            advance!(filt, ts[i+1]-ts[i])
        end
    end

    ys
end

function log_likelihood(filt, ts, ys, dys)
    n = size(ts, 1)

    ll = -0.5*n*log(2.0*pi)

    reset!(filt)
    for i in 1:n
        yp, vyp = predict(filt)

        if vyp < 0.0
            warn("Kalman filter has gone unstable!")
            return -Inf
        end
        
        dy = ys[i] - yp
        vy = vyp + dys[i]*dys[i]

        ll -= 0.5*log(vy)
        ll -= 0.5*dy*dy/vy

        observe!(filt, ys[i], dys[i])

        if i < n
            advance!(filt, ts[i+1]-ts[i])
        end
    end

    ll
end

function carmacovariance(ts, sigma, arroots, maroots)
    n = size(ts, 1)
    p = size(arroots, 1)
    q = size(maroots, 1)

    @assert q < p

    beta = poly(maroots)
    beta /= beta[1]

    cm = zeros((n,n))

    for i in 1:n
        for j in i:n
            for k in 1:p
                r = arroots[k]
                br = zero(beta[1])
                bmr = zero(beta[1])
                rprod = one(arroots[1])
                for l in 1:q+1
                    br += beta[l]*r^(l-1)
                    bmr += beta[l]*(-r)^(l-1)
                end
                for l in 1:p
                    if l != k
                        rprod *= (arroots[l] - r)*(conj(arroots[l]) + r)
                    end
                end

                cm[i,j] += real(br*bmr*exp(r*abs(ts[j]-ts[i]))/(-2.0*real(r)*rprod))
            end
            cm[j,i] = cm[i,j]
        end
    end

    sfact = sigma*sigma/cm[1,1]
    for i in 1:n
        for j in 1:n
            cm[i,j] *= sfact
        end
    end

    cm
end

function carmagenerate(ts, dys, mu, sigma, arroots, maroots)
    n = size(ts, 1)

    cm = carmacovariance(ts, sigma, arroots, maroots)
    for i in 1:n
        cm[i,i] += dys[i]*dys[i]
    end

    L = chol(cm, Val{:L})

    ys = mu + L*randn(n)

    ys
end

function raw_carma_log_likelihood(ts, ys, dys, mu, sigma, arroots, maroots)
    n = size(ts, 1)

    cm = carmacovariance(ts, sigma, arroots, maroots)
    for i in 1:n
        cm[i,i] += dys[i]*dys[i]
    end

    zs = ys - mu

    F = cholfact(cm)
    L = F[:L]

    logdet = 0.0
    for i in 1:n
        logdet += log(L[i,i])
    end

    -0.5*n*log(2*pi) - logdet - 0.5*dot(zs, F \ zs)
end

type CARMASqrtKalmanFilter
    mu::Float64
    sig::Float64
    x::Array{Complex128, 1}
    sx::Array{Complex128, 2}
    v::Array{Complex128, 2}
    arroots::Array{Complex128, 1}
    b::Array{Complex128, 2}
end

function CARMASqrtKalmanFilter(mu::Float64, sigma::Float64, arroots::Array{Complex128, 1}, maroots::Array{Complex128, 1})
    filt = CARMAKalmanFilter(mu, sigma, arroots, maroots)

    CARMASqrtKalmanFilter(mu, filt.sig, filt.x, chol(filt.vx, Val{:L}), filt.v, filt.arroots, filt.b)
end

function reset!(filt::CARMASqrtKalmanFilter)
    p = size(filt.x, 1)
    filt.x = zeros(Complex128, p)
    filt.sx = chol(filt.v, Val{:L})

    filt
end

function predict(filt::CARMASqrtKalmanFilter)
    yp = real(filt.b*filt.x) + filt.mu
    vyp = real(filt.b*filt.sx*filt.sx'*filt.b')

    yp[1], vyp[1,1]
end

# The following comes from
#
# Verhaegen, M., & Van Dooren, P. (1986). Numerical aspects of
# different Kalman filter implementations. IEEE Transactions on
# Automatic Control, 31(10),
# 907–917. http://doi.org/10.1109/TAC.1986.1104128
#
# which describes a combined observe and advance step in terms of the
# a triangularisation of a single matrix involving the measurement
# uncertainty, the evolution matrix, and the error term.
function observe_advance!(filt::CARMASqrtKalmanFilter, dt, y, dy)
    p = size(filt.x, 1)

    A = zeros(Complex128, p)
    for i in 1:p
        A[i] = exp(dt*filt.arroots[i])
    end

    B = copy(filt.v)
    for j in 1:p
        for i in 1:p
            B[i,j] = filt.v[i,j] - A[i]*filt.v[i,j]*conj(A[j])
        end
    end
    Q = B 
    try
        Q = chol(B, Val{:L})
    catch ex
        if isa(ex, Base.LinAlg.PosDefException)
            x = maximum(abs(diag(B)))
            for i in 1:p
                B[i,i] += 1e-8*x
            end
            Q = chol(B, Val{:L})
        else
            throw(ex)
        end
    end

    M = zeros(Complex128, (p+1, 2*p+1))
    M[1,1] = dy
    # M[1,2:p+1] = filt.b*filt.sx
    for j in 1:p
        for i in 1:p
            M[1,j+1] += filt.b[i]*filt.sx[i,j]
        end
    end
    # M[2:p+1, 2:p+1] = A*filt.sx
    for j in 1:p
        for i in 1:p
            M[i+1, j+1] = A[i]*filt.sx[i,j]
        end
    end
    M[2:p+1, p+2:2*p+1] = Q

    M = M'
    q, r = qr(M)
    r = r'

    #Re = r[1,1]
    #g = r[2:end, 1]
    #sx = r[2:end, 2:p+1]

    yp = filt.mu
    for i in 1:p
        yp += filt.b[i]*filt.x[i]
    end
    # x = A*x + g/Re*(y - yp)
    for i in 1:p
        filt.x[i] = A[i]*filt.x[i] + r[i+1,1]/r[1,1]*(y - yp)
    end
    filt.sx = r[2:end, 2:p+1]

    filt
end

function whiten(filt::CARMASqrtKalmanFilter, ts, ys, dys)
    n = size(ts, 1)

    wys = zeros(n)

    reset!(filt)
    
    for i in 1:n
        yp, vyp = predict(filt)

        wys[i] = (ys[i] - yp)/sqrt(vyp)

        if i < n
            observe_advance!(filt, ts[i+1]-ts[i], ys[i], dys[i])
        end
    end

    wys
end

function log_likelihood(filt::CARMASqrtKalmanFilter, ts, ys, dys)
    n = size(ts, 1)

    reset!(filt)

    ll = -0.5*n*log(2*pi)

    for i in 1:n
        yp, dyp = predict(filt)

        s2 = dyp + dys[i]*dys[i]
        
        ll -= 0.5*log(s2)
        ll -= 0.5*(ys[i]-yp)*(ys[i]-yp)/s2

        if i < n
            observe_advance!(filt, ts[i+1]-ts[i], ys[i], dys[i])
        end
    end

    ll
end

type CARMAKalmanPosterior
    ts::Array{Float64, 1}
    ys::Array{Float64, 1}
    dys::Array{Float64,1}
    p::Int
    q::Int
end

function nparams(post::CARMAKalmanPosterior)
    post.p + post.q + 3
end

type CARMAPosteriorParams
    mu::Float64
    sigma::Float64
    nu::Float64
    arroots::Array{Complex128, 1}
    maroots::Array{Complex128, 1}
end

function to_roots(x::Array{Float64, 1})
    n = size(x, 1)
    r = zeros(Complex128, n)

    if n == 0
        r    
    elseif n == 1
        r[1] = -exp(x[1])
        r
    else
        for i in 1:2:n-1
            logb = x[i]
            logc = x[i+1]

            b = exp(logb)
            c = exp(logc)

            d = b*b - 4*c
            if d < 0.0
                sd = sqrt(-d)
                r[i] = -0.5*(b + sd*1im)
                r[i+1] = -0.5*(b - sd*1im)
            else
                sd = sqrt(d)
                r[i] = -0.5*(b + sd)
                r[i+1] = -0.5*(b - sd)
            end
        end

        if n % 2 == 1
            r[end] = -exp(x[end])
        end

        r
    end
end

function to_rparams(x::Array{Complex128, 1})
    n = size(x, 1)
    rp = zeros(n)

    if n == 0
        rp
    elseif n == 1
        rp = log(-real(r[1]))
    else
        for i in 1:2:n-1
            r1 = x[i]
            r2 = x[i+1]

            b = real(-(r1 + r2))
            c = real(r1*r2)

            rp[i] = log(b)
            rp[i+1] = log(c)
        end

        if n % 2 == 1
            rp[end] = log(-real(x[end]))
        end

        sort_root_params(rp)
    end
end

function root_params_ordered(rp::Array{Float64, 1})
    n = size(rp, 1)

    if n == 0
        true
    elseif n == 1
        true
    else
        x = -Inf
        for i in 1:2:n-1
            if rp[i] > x
                x = rp[i]
            else
                return false
            end
        end
        true
    end
end

function sort_root_params(rp::Array{Float64, 1})
    n = size(rp, 1)

    if n == 0
        rp
    elseif n == 1
        rp
    else
        logbs = rp[1:2:n-1]
        logcs = rp[2:2:n]
        inds = sortperm(logbs)

        rps = zeros(n)
        for i in 1:2:n-1
            rps[i] = logbs[inds[div(i,2)+1]]
            rps[i+1] = logcs[inds[div(i,2)+1]]
        end

        if n % 2 == 1
            rps[end] = rp[end]
        end

        rps
    end
end

function to_params(post::CARMAKalmanPosterior, x::Array{Float64,1})
    @assert size(x,1)==nparams(post)

    CARMAPosteriorParams(x[1], x[2], x[3], to_roots(x[4:3+post.p]), to_roots(x[4+post.p:end]))
end

function to_array(post, p)
    x = zeros(nparams(post))

    x[1] = p.mu
    x[2] = p.sigma
    x[3] = p.nu
    x[4:3+post.p] = to_rparams(p.arroots)
    x[4+post.p:end] = to_rparams(p.maroots)

    x
end

function rmin_rmax(post::CARMAKalmanPosterior)
    dt = minimum(diff(post.ts))
    T = post.ts[end] - post.ts[1]

    min_r = 1.0/T
    max_r = 1.0/dt

    min_r, max_r
end

function log_prior(post::CARMAKalmanPosterior, x::Array{Float64, 1})
    if !root_params_ordered(x[4:3+post.p]) || !root_params_ordered(x[4+post.p:end])
        return -Inf
    end
    log_prior(post, to_params(post, x))
end

function log_prior(post::CARMAKalmanPosterior, x::CARMAPosteriorParams)
    min_r, max_r = rmin_rmax(post)
    
    mu = mean(post.ys)
    sig = std(post.ys)

    if x.mu < mu - 10*sig || x.mu > mu + 10*sig
        return -Inf
    end

    if x.sigma < sig / 10.0 || x.sigma > sig * 10.0
        return -Inf
    end

    if x.nu < 0.1 || x.nu > 10.0
        return -Inf
    end

    for i in 1:post.p
        r = x.arroots[i]
        if real(r) < -max_r || real(r) > -min_r
            return -Inf
        end

        if imag(r) < -max_r || imag(r) > max_r
            return -Inf
        end
    end

    for i in 1:post.q
        r = x.maroots[i]
        if real(r) < -max_r || real(r) > -min_r
            return -Inf
        end

        if imag(r) < -max_r || imag(r) > max_r
            return -Inf
        end
    end

    -log(x.sigma) - log(x.nu)
end

function CARMAKalmanFilter(post::CARMAKalmanPosterior, x::Array{Float64, 1})
    CARMAKalmanFilter(post, to_params(post, x))
end

function CARMAKalmanFilter(post::CARMAKalmanPosterior, p::CARMAPosteriorParams)
    CARMAKalmanFilter(p.mu, p.sigma, p.arroots, p.maroots)
end

function log_likelihood(post::CARMAKalmanPosterior, x::Array{Float64, 1})
    log_likelihood(post, to_params(post, x))
end

function log_likelihood(post::CARMAKalmanPosterior, x::CARMAPosteriorParams)
    filt = CARMAKalmanFilter(post, x)

    dys = post.dys * x.nu

    log_likelihood(filt, post.ts, post.ys, dys)
end

function log_posterior(post::CARMAKalmanPosterior, x::Array{Float64, 1})
    log_posterior(post, to_params(post, x))
end

function log_posterior(post::CARMAKalmanPosterior, x::CARMAPosteriorParams)
    lp = log_prior(post, x)

    if lp == -Inf
        lp
    else
        lp + log_likelihood(post, x)
    end
end

function randroots(rmin, rmax, n)
    p = zeros(n)

    bmin = 2*rmin
    bmax = 2*rmax

    cmin = rmin*rmin
    cmax = 2.0*rmax*rmax

    logbmin = log(bmin)
    logbmax = log(bmax)

    logcmin = log(cmin)
    logcmax = log(cmax)
    
    for i in 1:2:n-1
        logb = 0.0
        logc = 0.0
        while true
            logb = logbmin + (logbmax-logbmin)*rand()
            logc = logcmin + (logcmax-logcmin)*rand()

            rs = to_roots(Float64[logb, logc])
            
            r1 = rs[1]
            r2 = rs[2]

            if real(r1) < -rmin && real(r1) > -rmax && imag(r1) > -rmax && imag(r1) < rmax && real(r2) < -rmin && real(r2) > -rmax && imag(r2) > -rmax && imag(r2) < rmax
                break
            end
        end

        p[i] = logb
        p[i+1] = logc
    end

    if n % 2 == 1
        p[end] = log(rmin) + (log(rmax)-log(rmin))*rand()
    end

    to_roots(p)
end

function init(post, n)
    rmin, rmax = rmin_rmax(post)

    mu0 = mean(post.ys)
    sig0 = std(post.ys)
    
    xs = zeros((nparams(post), n))

    for i in 1:n
        mu = mu0-10*sig0 + 20*sig0*rand()
        sig = exp(log(0.1*sig0) + rand()*(log(10.0*sig0) - log(0.1*sig0)))
        nu = exp(log(0.1) + rand()*(log(10.0)-log(0.1)))

        arroots = randroots(rmin, rmax, post.p)
        maroots = randroots(rmin, rmax, post.q)

        p = CARMAPosteriorParams(mu, sig, nu, arroots, maroots)
        xs[:,i] = to_array(post, p)
    end

    xs
end

function whiten(post::CARMAKalmanPosterior, p::Array{Float64, 1})
    whiten(post, to_params(post, p))
end

function whiten(post::CARMAKalmanPosterior, p::CARMAPosteriorParams)
    filt = CARMAKalmanFilter(post, p)

    whiten(filt, post.ts, post.ys, post.dys*p.nu)
end

function residuals(post::CARMAKalmanPosterior, p::Array{Float64, 1})
    residuals(post, to_params(post, p))
end

function residuals(post::CARMAKalmanPosterior, p::CARMAPosteriorParams)
    filt = CARMAKalmanFilter(post, p)

    residuals(filt, post.ts, post.ys, post.dys*p.nu)
end

function residuals(filt::CARMAKalmanFilter, ts::Array{Float64, 1}, ys::Array{Float64, 1}, dys::Array{Float64, 1})
    reset!(filt)

    rs = zeros(size(ts, 1))
    drs = zeros(size(ts, 1))

    for i in 1:size(ts, 1)
        y, vy = predict(filt)

        rs[i] = ys[i] - y
        drs[i] = sqrt(vy + dys[i]*dys[i])

        observe!(filt, ys[i], dys[i])

        if i < size(ts, 1)
            advance!(filt, ts[i+1]-ts[i])
        end
    end

    rs, drs
end

function psd(post::CARMAKalmanPosterior, x::Array{Float64, 1}, fs::Array{Float64, 1})
    psd(post, to_params(post, x), fs)
end

function psd(post::CARMAKalmanPosterior, p::CARMAPosteriorParams, fs::Array{Float64, 1})
    psd = zeros(size(fs, 1))

    filt = CARMAKalmanFilter(post, p)
    
    for i in 1:size(fs, 1)
        f = fs[i]
        tpif = 2.0*pi*1.0im*f

        numer = polyeval(p.maroots, tpif) / polyeval(p.maroots, 0.0+0.0*1im)
        denom = polyeval(p.arroots, tpif)

        psd[i] = filt.sig*filt.sig*abs2(numer)/abs2(denom)
    end

    psd
end

end
