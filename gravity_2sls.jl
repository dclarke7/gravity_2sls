# load packages
#import Pkg
#Pkg.activate(".")
#Pkg.add("LinearAlgebra")
#Pkg.add("Distributions")
#Pkg.add("Distances")
#Pkg.add("Random")
#Pkg.add("Plots")
#Pkg.add("LaTeXStrings")
#Pkg.add("KernelDensity")
using LinearAlgebra, Distributions, Distances, Random, Plots, LaTeXStrings, KernelDensity
print("packages installed.","\n")

###############################################################################
# functions

# Two-stage Least Squares estimator
function twosls(x,y,z)

    N = size(x,1)

    # add constant
    x = hcat(ones(N),x)
    z = hcat(ones(N),z)

    P = z*inv(z'z)z'                       # projection matrix
    @assert isapprox(z,P*z;rtol=1e-10)     # oblique projection
    @assert isapprox(P,P*P;rtol=1e-10)     # idempotent
    @assert isapprox(P,P'P*P';rtol=1e-10)  # idempotent
    β_hat_iv = inv(x'P*x)x'P*y             # 2SLS beta (Cameron & Trivedi eq. 4.54)
    ε_hat_iv = y - x*β_hat_iv              # residuals
    K = size(z,2)                          # number of instruments (incl. constant)
    s=zeros(K,K)
    for i in 1:N
        s[:,:] += (ε_hat_iv[i].^2) .* z[i,:] * z[i,:]'
    end
    s=s/(N-K) # small sample adjustment from Cameron & Trivedi
    # VCV matrix (Cameron & Trivedi eq. 4.55)
    Σ = N * inv(x'P*x) * ( x'z*inv(z'z) * s * inv(z'z)z'x ) * inv(x'P*x)

    result = hcat(β_hat_iv[2,1],  # point estimate
                  sqrt.(Σ[2,2]))  # standard error
    return result
end

# Compute coverage probabilities from a matrix of (b,se)
function coverage_probability(x,β)
    ci = hcat(x[:,1] .+ 1.96*x[:,2],
              x[:,1] .- 1.96*x[:,2])
    cp = (ci[:,2] .< β) + (ci[:,1] .> β)
    cp = filter(b->b==2,cp)
    cp = length(cp)/size(x,1)
end

# Computes R² statistic (coefficient of determination)
function r²(y,x)
    N = size(y,1)
    x = hcat(ones(N),x)
    βhat = inv(x'x)x'y
    R² = y'x * βhat * inv(y'y)
    return R²
end

# Computes first-stage F statistic
function fstat(y,x,k::Int64=1)
    N = size(y,1)
    if k==1; @assert size(x,2)==1; end                                          # ensure number of instruments is entered
    x = hcat(ones(N),x)                                                         # add constant
    π_hat = inv(x'x)x'y                                                         # first stage regression coefficient
    yhat = x*π_hat                                                              # fitted values
    μ_hat = y - yhat                                                            # first stage errors
    ydm = yhat .- sum(y)/length(y)
    msr = ydm'ydm / k
    mse = μ_hat'μ_hat / (N-k-1)
    F = msr/mse
    return F
end

# Computes K distances for N observations
# (Credit to Jonathan Dingel for this function)
function generate_distances(N,K)
    dist_log = zeros(N*(N-1),K) # initialize matrix
    for k in 1:K
        Random.seed!(k)
        coords =  hcat(rand(Uniform(-90,90),N),    # lat
                       rand(Uniform(-180,180),N))' # long
        distance_log = log.(pairwise(Euclidean(),coords,dims=2))        # N x N distance matrix
	distance_log_vec = dropdims(reshape(distance_log,N^2,1),dims=2) # N*N x 1
        dist_log[:,k] = filter(d->d!=-Inf,distance_log_vec)             # N*(N-1) x 1
    end
    return dist_log
end


# run simulation?
simulate = true


if simulate==1

###############################################################################
# params

const β = 2   # true coefficient on trade
const N = 500 # number of countries

# generate gammas for distance instruments
γ_coeffs = [1,-1]                                    # instrument coefficients ("distance elasticities")
N_γ = 2                                              # expands the number of instruments
const K = size(γ_coeffs,1)*N_γ                       # number of instruments
γ = zeros(K+1)
γ[1] = 1                                             # constant
Z_size = 0.5                                         # strength of instruments
γ[2:end] = repeat(γ_coeffs,inner=1,outer=N_γ)*Z_size # gammas


###############################################################################
# Gravity Instrument Simulation

const iter = Int(1e3)                                                           # number of iterations
ols_results = zeros(iter,2);                                                    # storage for estimates
iv_results  = zeros(iter,3);                                                    # storage for estimates
rhos = [0,0.1,0.5]                                                              # ρ = correlation coefficient between errors, or level of endogeneity
fplot = Plots.plot()                                                            # plot for distribution of F statistics
colors = ["black","red","green"]
for (ρ,c) in zip(rhos,colors)
    for r in 1:iter
        # set seed
        Random.seed!(r+1000);

        εi  = rand(Normal(0,1),N)                                               # output equation error
        μi  = rand(Normal(0,1),N)                                               # auxiliary error for contamination
        ηij = (ρ.*repeat(εi,inner=(N-1),outer=1) +
                sqrt.(1-ρ.^2).*repeat(μi,inner=(N-1),outer=1))                  # ηij = ρ*ε_i + √(1-ρ^2) * μ_i
        #rand(Normal(0,1),N*(N-1)))

        # create N*(N-1) x K matrix of distances, plus a constant
        Zij = hcat(ones(N*(N-1)),generate_distances(N,K))

        # generate bilateral log trade: log T_ij = 1 + Z_ij*γ + η_ij
        Tij = Zij*γ + ηij
        # obtain total log trade summed over partners: Ti = log( sum_j exp(Tij) )
        Ti  = log.(sum(reshape(exp.(Tij),N-1,N),dims=(1,N))')
        # estimate gravity equation with K distances to obtain predicted bilateral log trade
        Tijhat = Zij*γ
        # generate instrument - predicted total log trade: log(sum_j exp(T_ij_hat) )
        Tihat  = log.(sum(reshape(exp.(Tijhat),N-1,N),dims=(1,N))')
        # output: Yi = 1 + T_i*β + ε_i
        Yi  = ones(N) + Ti*β + εi

        # OLS
        X = hcat(ones(N),Ti)                                                    # add constant
        β_hat_ols = inv(X'X)X'Yi                                                # estimate ols beta
        ε_hat_ols = Yi - X*β_hat_ols                                            # ols residuals
        σ² = (inv(X'X) * (X' * Diagonal(ε_hat_ols*ε_hat_ols') * X) * inv(X'X))  # ols vcv matrix (Cameron & Trivedi eq. 4.21)
        #σ² = inv(x'x) * ε_hat_ols'ε_hat_ols / (N-K)                            # homoskedastic error estimate
        ols_results[r,:] = [ β_hat_ols[2,1] , sqrt.(σ²[2,2]) ]                  # store ols results

        # 2SLS w/ first stage F statistic
        iv_results[r,:] = hcat(twosls(Ti,Yi,Tihat),fstat(Ti,Tihat,K))           # store 2sls results
    end

    # compute coverage probabilities
    iv_cp = Int(round(coverage_probability(iv_results,β)*100;digits=0))
    ols_cp = Int(round(coverage_probability(ols_results,β)*100;digits=0))

    print("coverage probability is ",iv_cp,"% for IV estimate for ρ=",ρ,".\n")
    print("coverage probability is ",ols_cp,"% for OLS estimate for ρ=",ρ,".\n")

    # plot estimates
    print("producing histogram of coefficient estimates.","\n")
    b = histogram(iv_results[:,1],bins=25,
                  xlabel="coefficient estimate",
                  ylabel="count (N=$iter)",
                  label="\\beta_{IV} ($iv_cp% coverage)",
                  legend=:topleft,
                  margin=5Plots.mm)
    histogram!(b,ols_results[:,1],bins=25,
               label="\\beta_{OLS} ($ols_cp% coverage)")
    plot!(b,[β],seriestype="vline",linewidth=2,
                label=string(L"\beta=2"),
                color="red")
    savefig(b,string("gravity_estimates_",ρ,"_test.png"))
    print("histogram complete.","\n")

    # plot F statistic density
    print("producing density of F statistics.","\n")
    f = iv_results[:,3]
    xrange = Int(floor(minimum(f))):Int(ceil(maximum(f)))
    f0 = kde(f);
    f0_line= pdf(f0::UnivariateKDE, xrange);
    xlims = 1:length(xrange)
    Plots.plot!(fplot,f0_line,
                color=c,
                label="\\rho = $ρ",
                title="F statistic (corrected for K distance instruments)",
                xlabel="F statistic",
                ylabel="density",
                legend=:topright,
                margin=5Plots.mm)
    savefig(fplot,string("F_statistics.png"))
    print("density complete.","\n")

end

end
