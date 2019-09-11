"""
Training a SVM based on the Dual formulation with JuMP
"""

using JuMP, AmplNLWriter
using Plots
using Random
using Statistics, LinearAlgebra
include("utils.jl")
Random.seed!(123);

n_samples = 200
n_features = 2

# Generate the points
y = zeros(n_samples)

r = rand(n_samples)
theta = rand(n_samples)*2*pi
x = hcat(r.*cos.(theta), r.*sin.(theta))

x[1:Int(n_samples/2),1] = x[1:Int(n_samples/2),1] .+ 2
x[1:Int(n_samples/2),2] = x[1:Int(n_samples/2),2] .+ 1
y[1:Int(n_samples/2)] = ones(Int(n_samples/2))

x[Int(n_samples/2)+1:end,1] = x[Int(n_samples/2)+1:end,1] .+ 1
x[Int(n_samples/2)+1:end,2] = x[Int(n_samples/2)+1:end,2] .+ 3
# x[n_samples,1] = 1.25
# x[n_samples,2] = 1
y[Int(n_samples/2)+1:end] = -ones(Int(n_samples/2))


#x,y = make_moons(n_samples=n_samples, noise=0.05, random_state=1)
# x, y = make_blobs(n_samples=n_samples, n_features=n_features, 
#                   centers=[4 0; 1 3], random_state=1)


# Kernel functions

# Linear Kernel
function linear(x1, x2; c=nothing)
    if c == nothing
        c = 0
    end
    if ndims(x1) == 1 && ndims(x2) == 1
        result = x1'*x2 .+ c
    elseif ndims(x1) > 1 && ndims(x2) == 1
        result = x1*x2 .+ c
    elseif ndims(x1) == 1 && ndims(x2) > 1
        result = x2*x1 .+ c
    else
        throw("linear kernel unimplemented for the given dimensions")
    end
    return result
end

# Gaussian Kernel
function rbf(x1, x2; gamma=nothing)
    if gamma == nothing
        gamma = 1.0 / size(x1)[end]
        #println("gamma=$gamma")
    end
    if ndims(x1) == 1 && ndims(x2) == 1
        result = exp(-gamma * norm(x1 - x2).^2)
    elseif ndims(x1) > 1 && ndims(x2) == 1
        result = exp.(-gamma * mapslices(norm, x1 - repeat(x2',size(x1,1)), dims=2).^2)
    elseif ndims(x1) == 1 && ndims(x2) > 1
        result = exp.(-gamma * mapslices(norm, repeat(x1',size(x2,1)) - x2, dims=2).^2)
#     elseif ndims(x1) > 1 && ndims(x2) > 1 && size(x1) == size(x2)
#         result = exp.(-gamma .* mapslices(norm, x1 .- x2, dims=2).^2)
    else
        throw("unimplemented")
    end
    return result
end

# Sigmoid Kernel
function sigmoid(x1, x2; gamma=0.05, c=1)
    if ndims(x1) == 1 && ndims(x2) == 1
        result = tanh(gamma * x1'*x2 + c)
    elseif ndims(x1) > 1 && ndims(x2) == 1
        result = tanh.(gamma .* x1*x2 .+ c)
    elseif ndims(x1) == 1 && ndims(x2) > 1
        result = tanh.(gamma .* x2*x1 .+ c)
    else
        throw("unimplemented")
    end
    return result
end



function solve_dual_svm(x,y,C; kernel="linear", c=nothing, gamma=nothing)
    N, N_dims = size(x)

    # Optimization model
    #model = Model(solver=AmplNLSolver("couenne", [""]));
    model = Model(solver=AmplNLSolver("ipopt", [""]));
    #model = Model(solver=ClpSolver())

    # Variables
    @variable(model, 0.0 <= α[1:N] <= C)

    K = zeros(N,N);
    for i=1:N
        for j=1:N
            # Kernel function
            if kernel == "linear"
                K[i,j] = linear(x[i,:], x[j,:], c=c)
            elseif kernel == "rbf"
                K[i,j] = rbf(x[i,:], x[j,:], gamma=gamma)
            elseif kernel == "sigmoid"
                K[i,j] = sigmoid(x[i,:], x[j,:], gamma=gamma, c=c)
            else
                throw("unimplemented")
            end
        end
    end


    # Objective
    #@NLobjective(model, Max, sum(α[i] for i=1:N) - 0.5*sum(α[i]*α[j]*y[i]*y[j]*x[i,k]*x[j,k] for i=1:N, j=1:N, k=1:N_dims));
    @NLobjective(model, Max, sum(α[i] for i=1:N) - 0.5*sum(α[i]*α[j]*y[i]*y[j]*K[i,j] for i=1:N, j=1:N));

    # Constraints
    @constraint(model, con, sum(α[i]*y[i] for i=1:N) == 0)

    # Solve
    println()
    status = solve(model)
    solvetime = getsolvetime(model)
    obj_value = getobjectivevalue(model);
    println("Solve time: ", solvetime)
    println("Objective=", obj_value);

    # Recover variable values

    α_opt = getvalue(α)

    w = sum(α_opt[i]*y[i]*x[i,:] for i=1:N if α_opt[i] > 0)
                
    if kernel == "linear"
        b = y[α_opt .> 0] - sum(α_opt[i]*y[i]*linear(x[i,:],x[α_opt .> 0,:],c=c) for i=1:N if α_opt[i] > 0)
    elseif kernel == "rbf"
        b = y[α_opt .> 0] - sum(α_opt[i]*y[i]*rbf(x[i,:],x[α_opt .> 0,:],gamma=gamma) for i=1:N if α_opt[i] > 0)
    elseif kernel == "sigmoid"
        b = y[α_opt .> 0] - sum(α_opt[i]*y[i]*sigmoid(x[i,:],x[α_opt .> 0,:],gamma=gamma,c=c) for i=1:N if α_opt[i] > 0)
    end 
    b = mean(b)
    println("w=$w")
    println("b=$b")
    return w, b, α_opt
end

function plot_decision_boundary(x, y, α_opt, b_opt; resolution=100, kernel_func=linear, C=1.0)
    # Generate coordinate grid of shape [resolution x resolution]
    # and evaluate the model over the entire space
    N, N_dims = size(x)

    x_range = range(minimum(x[:,1]), stop=maximum(x[:,1]), length=resolution)
    y_range = range(minimum(x[:,2]), stop=maximum(x[:,2]), length=resolution)

    grid = zeros(resolution, resolution)
    for (i,xr) in enumerate(x_range)
        # Print value of i for the user to see the progress
        if i % 10 == 0
            println("i=$i")
        end
        for (j,yr) in enumerate(y_range)
            x_new = [xr; yr]
            grid[i,j] = sum(α_opt[k]*y[k]*kernel_func(x[k,:], x_new) for k=1:N if α_opt[k] > 1e-7) + b_opt
        end
    end
    grid = grid'
    
    # Plot decision contours using grid and
    # make a scatter plot of training data
    #plt = contour(x_range, y_range, grid, levels=[-1,0,1], linestyles=[:dash,:dash,:dash], color=:gray)
    plt = scatter(x[y .== -1,1], x[y .== -1,2], color=:blue, leg=false, title="C=$C")
    plt = scatter!(x[y .== 1,1], x[y .== 1,2], color=:red, leg=false)
    plt = contour!(x_range, y_range, grid, levels=[-1,1], linestyles=[:dash,:dash], color=:gray)
    plt = contour!(x_range, y_range, grid, levels=[0], linestyles=[:dash], color=:black)
end

function plot_linear_separation(x, y, w_opt, b_opt; title=nothing)
    # Retrieve equation for the optimal separation line
    min_x = minimum(x[:,1])
    max_x = maximum(x[:,1])
    min_y = minimum(x[:,2])
    max_y = maximum(x[:,2])
    xx = range(min_x-abs(min_x)*0.1,stop=max_x+abs(max_x)*0.1,length=1000)
    a = - w_opt[1] / w_opt[2]
    y_line = a.*xx .+ (-b_opt / w_opt[2])

    # Equations for the margins
    y_margin1 = a.*xx .+ ((1 - b_opt) / w_opt[2])
    y_margin2 = a.*xx .+ ((-1 - b_opt) / w_opt[2])

    # Plot result
    if title != nothing
        plt = scatter(x[y .== -1,1], x[y .== -1,2], color=:blue, leg=false, title=title)
    else
        plt = scatter(x[y .== -1,1], x[y .== -1,2], color=:blue, leg=false)
    end
    plt = scatter!(x[y .== 1,1], x[y .== 1,2], color=:red, leg=false)
    plt = plot!(xx, y_line, linestyle=:dash, color=:black, leg=false)
    plt = plot!(xx, y_margin1, linestyle=:dash, color=:gray, leg=false)
    plt = plot!(xx, y_margin2, linestyle=:dash, color=:gray, leg=false)
    return plt
end

############################################################################

i = 1

kernel="linear"
gamma=nothing
c=nothing

#C_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
# C_values = [0.1, 5.0]
C_values = [1.0]
num_plots = length(C_values)
plot_array = Any[]
for C in C_values
    println("C=$C")
    w_opt, b_opt, α_opt = solve_dual_svm(x,y,C, kernel=kernel, gamma=gamma, c=c)

    global plt = plot_linear_separation(x, y, w_opt, b_opt, title="C=$C")    

    if num_plots > 1
        push!(plot_array, plt)
    end
    global i += 1
end

if num_plots > 1
    plt = plot(plot_array..., layout=(2,Int(num_plots/2)))
end

# Save figure
# savefig(plt, "svm_dual2.pdf");
# println("Plot saved.")

# Display plot
display(plt)

println("Press enter to close window")
readline()
