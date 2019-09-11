"""
Training a SVM with JuMP
"""

using JuMP, AmplNLWriter, Gurobi
using Plots
using Random
include("utils.jl")
Random.seed!(123);


function load_data()
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
    x[num_samples,1] = 1.25
    x[num_samples,2] = 1
    y[Int(num_samples/2)+1:end] = -ones(Int(num_samples/2))
    return x, y
end


function solve_svm(x,y,C)
    num_samples, N_dims = size(x)

    # Optimization model
    #model = Model(solver=AmplNLSolver("couenne", [""]));
    model = Model(solver=GurobiSolver(TimeLimit=400))

    # Variables
    @variable(model, w[1:N_dims])
    @variable(model, b)
    @variable(model, ξ[1:num_samples] >= 0.0)

    # Objective
    # ξ
    @objective(model, Min, 0.5*sum(w[k]^2 for k=1:N_dims) + C*sum(ξ[i] for i=1:num_samples))

    # Constraints
    @constraint(model, con[i=1:num_samples], y[i]*(w'*x[i,:] + b) >= 1 - ξ[i])

    # Solve
    println()
    status = solve(model)
    solvetime = getsolvetime(model)
    obj_value = getobjectivevalue(model);
    println("Solve time: ", solvetime)
    println("Objective=", obj_value);

    # Recover variable values
    w_opt = getvalue(w)
    b_opt = getvalue(b)

    return w_opt, b_opt
end


#x, y = load_data()
x, y = get_clouds()
num_samples, N_dims = size(x)

i = 1
#C_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
C_values = [1.0]
plot_array = Any[]
num_plots = length(C_values)
for C in C_values
    println("C=$C")
    w_opt, b_opt = solve_svm(x,y,C)

    # Retrieve equation for the optimal separation line
    min_x = minimum(x[:,1])
    max_x = maximum(x[:,1])
    xx = range(min_x-abs(min_x)*0.1,stop=max_x+abs(max_x)*0.1,length=1000)
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


# Plot result
# TODO: Plots with PyCall + matplotlib
# using PyCall
# plt = pyimport("matplotlib.pyplot")
# # plt = pyimport("matplotlib.pyplot")
# # plt.figure()
# plt.scatter(x[:,1], x[:,2], c=y)
# plt.plot(xx, y_line, linestyle='dashed', color='gray')
# plt.plot(xx, y_margin1, linestyle='dashed', color='gray')
# plt.plot(xx, y_margin2, linestyle='dashed', color='gray')
# plt.xlim((0,5))
# plt.ylim((0,5))
# plt.show()

# Save figure
#savefig(plt, "svm_primal.pdf");

# Display plot
display(plt)

println("Press enter to close window")
readline()
