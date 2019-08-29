"""
Training a SVM based on the Dual formulation with JuMP
"""

using JuMP, AmplNLWriter
using Plots
using Random
using Statistics
Random.seed!(123);

num_samples = 200
N_dims = 2

# Generate the points
y = zeros(num_samples)

r = rand(num_samples)
theta = rand(num_samples)*2*pi
x = hcat(r.*cos.(theta), r.*sin.(theta))

x[1:Int(num_samples/2),1] = x[1:Int(num_samples/2),1] .+ 2
x[1:Int(num_samples/2),2] = x[1:Int(num_samples/2),2] .+ 1
y[1:Int(num_samples/2)] = ones(Int(num_samples/2))

x[Int(num_samples/2)+1:end,1] = x[Int(num_samples/2)+1:end,1] .+ 1
x[Int(num_samples/2)+1:end,2] = x[Int(num_samples/2)+1:end,2] .+ 3
# x[num_samples,1] = 1.25
# x[num_samples,2] = 1
y[Int(num_samples/2)+1:end] = -ones(Int(num_samples/2))


function solve_dual_svm(x,y,C)
    N, N_dims = size(x)

    # Optimization model
    #model = Model(solver=AmplNLSolver("couenne", [""]));
    model = Model(solver=AmplNLSolver("ipopt", [""]));
    #model = Model(solver=ClpSolver())

    # Variables
    @variable(model, 0.0 <= α[1:N] <= C)

    # Objective
    @NLobjective(model, Max, sum(α[i] for i=1:N) - 0.5*sum(α[i]*α[j]*y[i]*y[j]*x[i,k]*x[j,k] for i=1:N, j=1:N, k=1:N_dims));

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
    b = y[α_opt .> 0] - x[α_opt .> 0,:]*w
    b = mean(b)
    println("w=$w")
    println("b=$b")
    return w, b
end


i = 1
C_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
#C_values = [1.0]
num_plots = length(C_values)
plot_array = Any[]
for C in C_values
    println("C=$C")
    w_opt, b_opt = solve_dual_svm(x,y,C)

    # Retrieve equation for the optimal separation line
    xx = range(0,stop=5,length=1000)
    a = - w_opt[1] / w_opt[2]
    y_line = a.*xx .+ (-b_opt / w_opt[2])

    # Equations for the margins
    y_margin1 = a.*xx .+ ((1 - b_opt) / w_opt[2])
    y_margin2 = a.*xx .+ ((-1 - b_opt) / w_opt[2])

    # Plot result
    global plt = scatter(x[1:Int(num_samples/2),1], x[1:Int(num_samples/2),2], color=:blue, leg=false, title="C=$C")
    plt = scatter!(x[Int(num_samples/2)+1:end,1], x[Int(num_samples/2)+1:end,2], color=:red, leg=false)
    plt = plot!(xx, y_line, linestyle=:dash, color=:gray, leg=false)
    plt = plot!(xx, y_margin1, linestyle=:dash, color=:gray, leg=false)
    plt = plot!(xx, y_margin2, linestyle=:dash, color=:gray, leg=false)

    if num_plots > 1
        push!(plot_array, plt)
    end
    global i += 1
end

if num_plots > 1
    plt = plot(plot_array..., layout=(2,Int(num_plots/2)))
end

# Save figure
savefig(plt, "svm_dual.pdf");