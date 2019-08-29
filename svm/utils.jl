using Plots

function get_clouds()
    N = 1000
    c1 = [2; 2]
    c2 = [-2; -2]
    # c1 = np.array([0, 3])
    # c2 = np.array([0, 0])
    X1 = randn(N, 2) + repeat(c1',N)
    X2 = randn(N, 2) + repeat(c2',N)
    X = vcat(X1, X2)
    Y = vcat(-1 .* ones(N), 1 .* ones(N))
    return X, Y
end


function get_donut()
    N = 500
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = randn(Int(N/2)) .+ R_inner
    theta = 2*pi*rand(Int(N/2))
    X_inner = hcat(R1 .* cos.(theta), R1 .* sin.(theta))

    R2 = randn(Int(N/2)) .+ R_outer
    theta = 2*pi*rand(Int(N/2))
    X_outer = hcat(R2 .* cos.(theta), R2 .* sin.(theta))

    X = vcat(X_inner, X_outer)
    Y = vcat(zeros(Int(N/2)), ones(Int(N/2)))
    return X, Y
end


function get_xor()
    N = 200
    X = zeros(N, 2)
    X[1:Int(N/4),:] = rand(Int(N/4), 2) ./ 2 .+ 0.5 # (0.5-1, 0.5-1)
    X[Int(N/4)+1:Int(N/2),:] = rand(Int(N/4), 2) ./ 2 # (0-0.5, 0-0.5)
    X[Int(N/2)+1:Int(3*N/4),:] = rand(Int(N/4), 2) ./ 2 + repeat([0 0.5],Int(N/4)) # (0-0.5, 0.5-1)
    X[Int(3*N/4)+1:end,:] = rand(Int(N/4), 2) ./ 2 + repeat([0.5 0],Int(N/4)) # (0.5-1, 0-0.5)
    Y = vcat(zeros(Int(N/2)), ones(Int(N/2)))
    return X, Y
end

# X,Y = get_clouds()
# let i = 1
#     for label in unique(Y)
#         if i == 1
#             global plt = scatter(X[Y.==label,1], X[Y.==label,2])    
#         else
#             plt = scatter!(X[Y.==label,1], X[Y.==label,2])
#         end
#         i += 1
#     end
# end
# display(plt)

# println("Press enter to close window")
# readline()
