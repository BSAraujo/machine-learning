using Plots
using LinearAlgebra
using Statistics, Distributions
using JuMP, AmplNLWriter
include("utils.jl")

function solve_mpm(X,Y)
    println(size(X))

    # calculate mean of data
    mean_x = dropdims(mean(X, dims=1), dims=1)
    mean_y = dropdims(mean(Y, dims=1), dims=1)

    # calculate covariance matrix of data
    sigmaX = cov(X, dims=1)
    sigmaY = cov(Y, dims=1)

    # number of classes
    n = 2

    # Optimization model
    model = Model(solver=AmplNLSolver("couenne", [""]))

    # Variables
    @variable(model, a[1:n])
    #@variable(model, ξ >= 1e-7)

    # Objective
    @NLobjective(model, Min, sqrt( sum(a[i]*sigmaX[i,j]*a[j] for i=1:n, j=1:n)) +
                             sqrt( sum(a[i]*sigmaY[i,j]*a[j] for i=1:n, j=1:n)));

    # Constraints
    @constraint(model, con, sum(a[i] * (mean_x[i] - mean_y[i]) for i=1:n) == 1);
    #@constraint(model, con, sum(a[i] * (mean_y[i] - mean_x[i]) for i=1:n) == 1);

    # Set initial value to 1 to prevent the solver from trying to calculate the derivative at sqrt(0)
    setvalue(a, [1;1])

    println()
    status = solve(model)
    solvetime = getsolvetime(model)
    obj_value = getobjectivevalue(model);
    println("Solve time: ", solvetime)
    println("Objective=", obj_value);


    a = getvalue(a)
    println("a=$a")

    # calculate b
    b = a'*mean_x - sqrt(a'*sigmaX*a) / (sqrt(a'*sigmaX*a) + sqrt(a'*sigmaY*a))
    return a, b
end

#################################################################
### Main Program

#X_original, Y_original = get_clouds()
X_original, Y_original = get_multivariate_normal()

X = X_original[Y_original .== -1, :];
Y = X_original[Y_original .== 1, :];

a,b = solve_mpm(X,Y)

# Retrieve equation for the optimal separation line
min_x = minimum(X_original[:,1])
max_x = maximum(X_original[:,1])
min_y = minimum(X_original[:,2])
max_y = maximum(X_original[:,2])
xx = range(min_x-abs(min_x)*0.1,stop=max_x+abs(max_x)*0.1,length=1000)

if abs(a[2]) > 1e-5
    y_line = - (a[1]/a[2]).*xx .+ (b/a[2])
end

plt = scatter(X[:,1], X[:,2], color=:blue, leg=false)
plt = scatter!(Y[:,1], Y[:,2], color=:red, leg=false)
if abs(a[2]) > 1e-5
    plot!(xx, y_line, linestyle=:dash, color=:gray, leg=false, ylim=(-2, 10))
else
    println("vertical line")
    vline!([a[1]], linestyle=:dash, color=:gray)
end
display(plt)

println("Press enter to close window")
readline()