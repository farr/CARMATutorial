module Transit

""" Produces a box transit lightcurve at the given times, with the
given period, starting time, duration, and (fractional) depth. """
function transit_lightcurve(ts, P, T0, Tdur, d)
    ts = ts - T0
    tsince = mod(ts, P)
    tsince[tsince .< 0.0] += P

    tr = ones(size(ts,1))
    tr[tsince .< Tdur] -= d
    tr
end

""" Compute the SNR of a matched-filter transit search in the data.
The SNR is actually taken to be `sqrt(log(likelihood_ratio))` for a
search with a constant lightcurve compared to the transit lightcurve
given.  The `sigmas` argument is either an array giving the estimated
measurement uncertainty at each point or a scalar giving the
constant-in-time measurement uncertainty. """
function snr(flux, transit, sigmas)
    n = size(flux,1)
    ons = ones(n)

    nflux = flux ./ sigmas
    ntr = transit ./ sigmas
    nons = ons ./ sigmas
    
    rho2 = dot(nflux, ntr)^2/dot(ntr,ntr) - dot(nflux, nons)^2/dot(nons, nons)

    if rho2 > 0.0
        sqrt(rho2)
    else
        0.0
    end
end

function snr(flux, transit, sigma::Number)
    snr(flux, transit, sigma*ones(size(flux,1)))
end

function bls_snr(flux, transit)
    intr = transit .< 1.0
    outtr = ~intr

    muin = mean(flux[intr])
    muout = mean(flux[outtr])

    sigin = std(flux[intr])/sqrt(sum(intr))
    sigout = std(flux[outtr])/sqrt(sum(outtr))
    
    (muout - muin)/sqrt(sigin^2 + sigout^2)
end

function new_snr(flux, trans)
    intr = trans .< 1.0
    outtr = ~intr

    flux = flux - mean(flux[outtr])

    tr = zeros(size(trans,1))
    tr[intr] = -1.0

    dot(flux/std(flux), tr)/sqrt(dot(tr,tr))
end

end
