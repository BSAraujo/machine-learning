using Plots, Random

function plot_classes(X, y=nothing)
    if y == nothing
        global plt = scatter(X[:,1], X[:,2])
    else
        let i = 1
            for label in unique(y)
                if i == 1
                    global plt = scatter(X[y.==label,1], X[y.==label,2])    
                else
                    plt = scatter!(X[y.==label,1], X[y.==label,2])
                end
                i += 1
            end
        end
    end
    display(plt);
end

function get_clouds(; n_samples=500, c1=[2; 2], c2=[-2; -2])
    # c1 = np.array([0, 3])
    # c2 = np.array([0, 0])
    X1 = randn(n_samples, 2) + repeat(c1',n_samples)
    X2 = randn(n_samples, 2) + repeat(c2',n_samples)
    X = vcat(X1, X2)
    Y = vcat(-1 .* ones(n_samples), 1 .* ones(n_samples))
    return X, Y
end

function get_donut(; R_inner=5, R_outer=10, n_samples=500)
    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = randn(Int(n_samples/2)) .+ R_inner
    theta = 2*pi*rand(Int(n_samples/2))
    X_inner = hcat(R1 .* cos.(theta), R1 .* sin.(theta))

    R2 = randn(Int(n_samples/2)) .+ R_outer
    theta = 2*pi*rand(Int(n_samples/2))
    X_outer = hcat(R2 .* cos.(theta), R2 .* sin.(theta))

    X = vcat(X_inner, X_outer)
    Y = vcat(-1 .* ones(Int(n_samples/2)), ones(Int(n_samples/2)))
    return X, Y
end


function get_multivariate_normal(; n_samples=500, muX=[2; 5], muY=[10; 4], 
                                   sigX=[4 4; 4 8], sigY=[8 -4; -4 3])
    sigX = (sigX ./ norm(sigX)) .* 2
    X1 = rand(MvNormal(muX, sigX), n_samples)'

    sigY = sigY ./ norm(sigY)
    sigY = sigY .* 3 
    X2 = rand(MvNormal(muY, sigY), n_samples)'
    X = vcat(X1, X2)
    Y = vcat(-1 .* ones(n_samples), 1 .* ones(n_samples))
    return X,Y
end

function get_xor(; n_samples=200)
    X = zeros(n_samples, 2)
    X[1:Int(n_samples/4),:] = rand(Int(n_samples/4), 2) ./ 2 .+ 0.5 # (0.5-1, 0.5-1)
    X[Int(n_samples/4)+1:Int(n_samples/2),:] = rand(Int(n_samples/4), 2) ./ 2 # (0-0.5, 0-0.5)
    X[Int(n_samples/2)+1:Int(3*n_samples/4),:] = rand(Int(n_samples/4), 2) ./ 2 + repeat([0 0.5],Int(n_samples/4)) # (0-0.5, 0.5-1)
    X[Int(3*n_samples/4)+1:end,:] = rand(Int(n_samples/4), 2) ./ 2 + repeat([0.5 0],Int(n_samples/4)) # (0.5-1, 0-0.5)
    Y = vcat(zeros(Int(n_samples/2)), ones(Int(n_samples/2)))
    return X, Y
end


function shuffle_data(X, y; random_state=nothing)
    if size(X)[1] != length(y)
        throw("The first dimension of X and y should exactly match")
    end
    if random_state == nothing
        Random.seed!()
    else
        Random.seed!(random_state)
    end
    idx = shuffle(1:size(X)[1])
    X = X[idx,:]
    y = y[idx]
    return X, y
end


function make_moons(; n_samples=100, shuffle=true, noise=nothing, random_state=nothing)
    """Make two interleaving half circles
    A simple toy dataset to visualize clustering and classification
    algorithms.
    """

    n_samples_out = Int(floor(n_samples / 2))
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = cos.(range(0, stop=pi, length=n_samples_out))
    outer_circ_y = sin.(range(0, stop=pi, length=n_samples_out))
    inner_circ_x = 1 .- cos.(range(0, stop=pi, length=n_samples_in))
    inner_circ_y = 1 .- sin.(range(0, stop=pi, length=n_samples_in)) .- 0.5

    X = hcat(vcat(outer_circ_x, inner_circ_x),
             vcat(outer_circ_y, inner_circ_y))
    y = vcat(-1 .* ones(n_samples_out),
             ones(n_samples_in))

    if shuffle
        X, y = shuffle_data(X, y, random_state=random_state)
    end

    if noise != nothing
        if random_state == nothing
            Random.seed!()
        else
            Random.seed!(random_state)
        end
        X .+= randn(size(X))*noise
    end

    return X, y
end



function make_blobs(; n_samples=100, n_features=2, centers=nothing, cluster_std=1.0, 
                    center_box=(-10.0, 10.0), shuffle=true, random_state=nothing)
    """Generate isotropic Gaussian blobs for clustering.
    """
    #generator = check_random_state(random_state)
    if random_state == nothing
        Random.seed!()
    else 
        Random.seed!(random_state)
    end

    if isa(n_samples, Int)
        # Set n_centers by looking at centers arg
        if centers == nothing
            centers = 3
        end

        if isa(centers, Int)
            n_centers = centers
            centers = (center_box[2] - center_box[1]).*rand(n_centers, n_features) .+ center_box[1]
            #centers = generator.uniform(center_box[0], center_box[1], size=(n_centers, n_features))
        else
            #centers = check_array(centers)
            n_features = size(centers)[2]
            n_centers = size(centers)[1]
        end
    else
        # Set n_centers by looking at [n_samples] arg
        n_centers = len(n_samples)
        if centers == nothing
            centers = (center_box[2] - center_box[1])*rand(n_centers, n_features) + center_box[1]
            #centers = generator.uniform(center_box[0], center_box[1], size=(n_centers, n_features))
        end
        try
            @assert len(centers) == n_centers
        catch e
            if isa(e, TypeError)
                throw(DomainError(string("Parameter `centers` must be array-like. ",
                                         "Got ", typeof(centers), " instead")))
            elseif isa(e, AssertionError)
                throw(DomainError(string("Length of `n_samples` not consistent",
                                         " with number of centers. Got n_samples = ", n_samples,
                                         " and centers = ", centers)))
            end
        end
        #centers = check_array(centers)
        n_features = size(centers)[2]
    end

    # stds: if cluster_std is given as list, it must be consistent
    # with the n_centers
    if (isa(cluster_std, Array) && length(cluster_std) != n_centers)
        throw(DomainError(string("Length of `clusters_std` not consistent with ",
                                 "number of centers. Got centers = ", centers,
                                 " and cluster_std = ", cluster_std)))
    end

    if isa(cluster_std, Number)
        cluster_std = fill(cluster_std, size(centers))
        #cluster_std = np.full(len(centers), cluster_std)
    end

    # Initialize data as empty arrays
    X = reshape([],0,n_features)
    y = []

    # Transform n_samples to an array
    if isa(n_samples, Array)
        n_samples_per_center = n_samples
    else
        n_samples_per_center = repeat([Int(floor(n_samples / n_centers))], n_centers)

        for i=1:(n_samples % n_centers)
            n_samples_per_center[i] += 1
        end
    end

    # Relocate blobs and add random noise
    if n_centers == 2  # in case of binary classes, use -1 and 1 as class labels (which make it more friendly for SVMs)
        # label -1
        n = n_samples_per_center[1]
        std = cluster_std[1]
        X = vcat(X, randn(n, n_features).*std + repeat(centers[1,:]', n))
        y = vcat(y, repeat([-1], n))
        # label 1
        n = n_samples_per_center[2]
        std = cluster_std[2]
        X = vcat(X, randn(n, n_features).*std + repeat(centers[2,:]', n))
        y = vcat(y, repeat([1], n))
    else
        for (i, (n, std)) in enumerate(zip(n_samples_per_center, cluster_std))
            X = vcat(X, randn(n, n_features).*std + repeat(centers[i,:]', n))
            y = vcat(y, repeat([i], n))
        end
    end

    # Shuffle data
    if shuffle
        X, y = shuffle_data(X, y, random_state=random_state)
    end

    return X, y
end


function plot_data(data)
    X, y = data
    if y == nothing
        global plt = scatter(X[:,1], X[:,2])
    else
        let i = 1
            for label in unique(y)
                if i == 1
                    global plt = scatter(X[y.==label,1], X[y.==label,2])    
                else
                    plt = scatter!(X[y.==label,1], X[y.==label,2])
                end
                i += 1
            end
        end
    end
    display(plt)
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
