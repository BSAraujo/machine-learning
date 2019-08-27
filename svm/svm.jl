"""
Training a SVM with JuMP
"""

using JuMP, AmplNLWriter
using Plots
using Random
Random.seed!(123);

num_samples = 2000
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
y[Int(num_samples/2)+1:end] = -ones(Int(num_samples/2))


# Optimization model - Quadratic Program
model = Model(solver=AmplNLSolver("couenne", [""]));

# Variables
@variable(model, w[1:N_dims])
@variable(model, b)

# Objective
@NLobjective(model, Min, 0.5 * sum(w[k]^2 for k=1:N_dims))

# Constraints
@constraint(model, con[i=1:num_samples], y[i]*(sum(w[k]*x[i,k] for k=1:N_dims) + b) >= 1)

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

# Retrieve equation for the optimal separation line
x_plot = range(0,stop=5,length=1000)
y_line = .- (w_opt[1] / w_opt[2]) .* x_plot  .- (b_opt / w_opt[2])

# Equations for the margins
y_margin1 = .- (w_opt[1] / w_opt[2]) .* x_plot .- (b_opt / w_opt[2] + 1 / w_opt[2])
y_margin2 = .- (w_opt[1] / w_opt[2]) .* x_plot .- (b_opt / w_opt[2] - 1 / w_opt[2])

# Plot result
# TODO: Plots with PyCall + matplotlib
# using PyCall
# plt = pyimport("matplotlib.pyplot")
# plt.figure()
# plt.scatter(x[:,0], x[:,1], c=y)
# plt.plot(x_plot, y_line, linestyle='dashed', color='gray')
# plt.plot(x_plot, y_margin1, linestyle='dashed', color='gray')
# plt.plot(x_plot, y_margin2, linestyle='dashed', color='gray')
# plt.xlim((0,5))
# plt.ylim((0,5))
# plt.show()

# Plot result
plt = scatter(x[1:Int(num_samples/2),1], x[1:Int(num_samples/2),2], color=:blue)
plt = scatter!(x[Int(num_samples/2)+1:end,1], x[Int(num_samples/2)+1:end,2], color=:red)
plt = plot!(x_plot, y_line, linestyle=:dash, color=:gray)
plt = plot!(x_plot, y_margin1, linestyle=:dash, color=:gray)
plt = plot!(x_plot, y_margin2, linestyle=:dash, color=:gray)

# Save figure
png(plt, "tmp.png");