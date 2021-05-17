# see tutorial https://turinglang.github.io/TuringCallbacks.jl/dev/
using Turing, TuringCallbacks, TensorBoardLogger, ArgParse

function estimate_model(;num_samples = 10000, num_adapts = 100, s_prior_alpha = 2, s_prior_theta = 3)
    @model function demo(x; s_prior_alpha, s_prior_theta)
        s ~ InverseGamma(s_prior_alpha, s_prior_theta)
        m ~ Normal(0, √s)
        for i in eachindex(x)
            x[i] ~ Normal(m, √s)
        end
    end

    xs = randn(100) .+ 1
    model = demo(xs;s_prior_alpha,s_prior_theta)

    # Sampling algorithm to use
    alg = NUTS(num_adapts, 0.65)

    # Create the callback
    callback = TensorBoardCallback("tensorboard_logs/run")

    # Sample
    chain = sample(model, alg, num_samples; callback = callback)
    sum_stats = describe(chain)
    param_names = sum_stats[1][:,1]
    param_rhat = sum_stats[1][:,7]
    param_ess_per_sec = sum_stats[1][:,8]

    # Log the ESS/sec and rhat.  Nice to show as summary results from tensorboard
    for i in 1:length(param_names)
        TensorBoardLogger.log_value(callback.logger, "$(param_names[i])_ess_per_sec", param_ess_per_sec[i])
        TensorBoardLogger.log_value(callback.logger, "$(param_names[i])_rhat", param_rhat[i])
    end
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--num_samples"
            help = "another option with an argument"
            arg_type = Int
            default = 10000
        "--num_adapts"
            help = "another option with an argument"
            arg_type = Int
            default = 100
        "--s_prior_alpha"
            help = "alpha in InverseGamma prior for s"
            arg_type = Float64
            default = 2.0
        "--s_prior_theta"
            help = "theta in InverseGamma prior for s"
            arg_type = Float64
            default = 3.0
    end

    return parse_args(s)
end

# Easily convert to named tuple for arguments
dictkeys(d::Dict) = (collect(Symbol.(keys(d)))...,)
dictvalues(d::Dict) = (collect(values(d))...,)
namedtuple(d::Dict{String,Any})=
    NamedTuple{dictkeys(d)}(dictvalues(d))
    
function main()
    parsed_args = parse_commandline()    
    #Call to estimate model
    estimate_model(;namedtuple(parsed_args)...)  #converts all arguments to named tuple then splat into solution
end


# call with baseline parameters
main()
