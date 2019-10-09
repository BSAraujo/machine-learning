
instance_path = "Datasets/p03.txt"

dataset_name = nothing
num_samples = nothing
num_attributes = nothing
attribute_types = nothing
num_classes = nothing
data_classes = nothing

start = time()
open(instance_path) do fp
    global dataset_name = split(readline(fp))[2]
    global num_samples = parse(Int,(split(readline(fp))[2]))
    global num_attributes = parse(Int,(split(readline(fp))[2]))
    global attribute_types = split(readline(fp))[2:end]
    if ~(Set(attribute_types) ⊆ ["N", "C"])
        throw("ERROR: non recognized attribute type")
    end
    global num_classes = parse(Int, split(readline(fp))[2])
    global data_attributes = zeros(num_samples, num_attributes)
    global data_classes = zeros(num_samples)
    
    is_finished = false
    for (i, line) in enumerate(eachline(fp))
        if strip(line) == "EOF"
            is_finished = true
            break
        end
        values = split(line)
        data_attributes[i,:] = parse.([Float64], values[1:(end-1)]) 
        data_classes[i] = parse(Int, values[end])
    end
    num_levels = floor.(Int, maximum(data_attributes, dims=1)) .+ 1
    if ~is_finished
        throw("ERROR when reading instance, EOF has not been found where expected")
    end
    if maximum(data_classes) >= num_classes
        throw("ERROR: class indices should be in 0...num_classes-1")
    end
end
delta = time() - start
println("Instance: $instance_path")
println(string("----- DATASET [", dataset_name, "] LOADED IN ", delta, " (s)"))
println(string("----- NUMBER OF SAMPLES: " , num_samples))
println(string("----- NUMBER OF ATTRIBUTES: ", num_attributes))
println(string("----- NUMBER OF CLASSES: ", num_classes))

X = data_attributes;
y = data_classes .+ 1;



# Min Max Scaling
for p=1:size(X, 2)
    X[:,p] = (X[:,p] .- minimum(X[:,p])) ./ (maximum(X[:,p]) - minimum(X[:,p]))
end

n, p = size(X)

println("Number of observations: $n")
println("Number of features: $p")

X

function mode(values)
    dict = Dict() # Values => Number of repetitions
    modesArray = typeof(values[1])[] # Array of the modes so far
    max = 0 # Max of repetitions so far
 
    for v in values
        # Add one to the dict[v] entry (create one if none)
        if v in keys(dict)
            dict[v] += 1
        else
            dict[v] = 1
        end
 
        # Update modesArray if the number of repetitions
        # of v reaches or surpasses the max value
        if dict[v] >= max
            if dict[v] > max
                empty!(modesArray)
                max += 1
            end
            append!(modesArray, [v])
        end
    end
 
    return modesArray[1]
end

Dmax = 3 # Maximum tree depth
Tnodes = 2^(Dmax+1) - 1 # Number of nodes in the tree
Nmin = 1 # Minimum number of points at each leaf
TBranch = Int(floor(Tnodes/2))
TLeaf = Int(floor(Tnodes/2)) + 1
@assert Tnodes == TBranch + TLeaf

M = n

most_frequent_class = mode(y)
L_hat = sum(y .== most_frequent_class) / length(y)

K = num_classes

α = 0

println("Max Depth of the tree: $Dmax")
println("Number of nodes in the tree: $Tnodes")
println("Number of branch nodes: $TBranch")
println("Number of leaf nodes: $TLeaf")

function parent(t)
    return Int(floor(t/2))
end

function ancestors(t)
    ancs = []
    while t != 1
        anc = parent(t)
        t = anc
        push!(ancs, t)
    end
    return ancs
end

function left_ancestors(t)
    ancs = []
    while t != 1
        anc = parent(t)
        if anc*2 == t
            push!(ancs, anc)
        end
        t = anc
    end
    return ancs
end

function right_ancestors(t)
    ancs = []
    while t != 1
        anc = parent(t)
        if anc*2 + 1 == t
            push!(ancs, anc)
        end
        t = anc
    end
    return ancs
end

Y = zeros(n,K)
for i=1:n, k=1:K
    if y[i] == k
        Y[i,k] = 1
    else
        Y[i,k] = -1
    end
end

Y

ϵ = zeros(p)
for j=1:p
    Xfeat = sort(X[:,j])
    
    diffX = diff(Xfeat)
    ϵ[j] = minimum(diffX[diffX .!= 0])
    #ϵ[j] = minimum(Xfeat[(i+1)] - Xfeat[i] for i=1:(n-1) if Xfeat[(i+1)] != Xfeat[i])
end

ϵ_max = maximum(ϵ)
ϵ_min = minimum(ϵ)

println("Epsilon min = $ϵ_min")
println("Epsilon max = $ϵ_max")

#########################################################

using PyCall
py"""
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def run_decision_tree(X, y, max_depth):
    n, p = X.shape

    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_acc = sum(y_pred == y) / len(y)
    print('Train accuracy (sklearn tree):', train_acc)

    n_nodes = clf.tree_.node_count
    print('Number of nodes in sklearn Tree:', n_nodes)

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    # BFS
    s = 0

    visited = [False] * (n_nodes) 
    queue = [] 
    stack = [s] 

    queue.append(s) 
    visited[s] = True

    while queue: 
        s = queue.pop(0) 
        for i in [children_left[s], children_right[s]]: 
            if visited[i] == False: 
                queue.append(i) 
                stack.append(i)
                visited[i] = True

    feature_list = [feature[i] for i in stack]
    threshold_list = [threshold[i] for i in stack]
    print(threshold_list)
    threshold_list = threshold_list[:int(np.floor(n_nodes/2))]

    A = np.zeros((p, int(np.floor(n_nodes/2))))
    for j in range(int(np.floor(n_nodes/2))):
        if feature_list[j] >= 0:
            A[feature_list[j],j] = 1
    print(A)
    return train_acc, A, threshold_list
"""
train_acc_grd, a_grd, b_grd = py"run_decision_tree"(X, y, Dmax)
b_grd = b_grd + a_grd'*(ϵ.*0.99)
println(b_grd)

#########################################################

using Gurobi, JuMP

model = Model(solver=GurobiSolver(TimeLimit=10))

# Variables
@variable(model, z[1:n,1:TLeaf], Bin); # To track points assigned to each leaf node
@variable(model, l[1:TLeaf], Bin); # indicator variable, l[t] == 1 iff leaf t contains any points

@variable(model, a[1:p,1:TBranch], Bin); # (4) - number of features by number of branch nodes
@variable(model, d[1:TBranch], Bin); # To track which branch nodes apply splits

@variable(model, 0 <= b[1:TBranch] <= 1);

@variable(model, L[1:TLeaf]); # Number of points in the node minus the number of points of the most common label

@variable(model, Nkt[1:K,1:TLeaf], Int); # Number of points of label k in node t
@variable(model, Nt[1:TLeaf], Int); # Total number of points in node t

@variable(model, c[1:K,t=1:TLeaf], Bin); # To track the prediction of each node

# Objective
@objective(model, Min, (1/L_hat)*sum(L[t] for t=1:TLeaf) + α * sum(d[t] for t=1:TBranch));

# Constraints
@constraint(model, conL[t=1:TLeaf], 0 <= L[t]);

@constraint(model, con2[t=1:TBranch], sum(a[j,t] for j=1:p) == d[t]); # (2)
@constraint(model, con3a[t=1:TBranch], 0 <= b[t]); # (3a)
@constraint(model, con3b[t=1:TBranch], b[t] - d[t] <= 0); # (3b)
@constraint(model, con5[t=2:TBranch], d[t] <= d[parent(t)]); # (5)

@constraint(model, con6[t=1:TLeaf, i=1:n], z[i,t] <= l[t]); # (6) track points assigned to each leaf node
@constraint(model, con7[t=1:TLeaf], sum(z[i,t] for i=1:n) >= Nmin * l[t]); # (7) enforce min. number of points at each leaf
@constraint(model, con8[i=1:n], sum(z[i,t] for t=1:TLeaf) == 1); # (8) force that each point is assigned to exactly one leaf

@constraint(model, conNkt[k=1:K,t=1:TLeaf], Nkt[k,t] >= 0);
@constraint(model, conNt[t=1:TLeaf], Nt[t] >= 0);

@constraint(model, con15[k=1:K,t=1:TLeaf], Nkt[k,t] == (1/2) * sum((1 + Y[i,k])*z[i,t] for i=1:n)); # (15)
@constraint(model, con16[t=1:TLeaf], Nt[t] == sum(z[i,t] for i=1:n)); # (16)

@constraint(model, con18[t=1:TLeaf], sum(c[k,t] for k=1:K) == l[t]); # (18)

@constraint(model, con14[i=1:n,t=1:TLeaf,m=right_ancestors(TBranch + t)], 
    sum(a[j,m]*X[i,j] for j=1:p) >= b[m] - (1 - z[i,t]) ); # (14)
# @constraint(model, con13[i=1:n,t=1:TLeaf,m=left_ancestors(TBranch + t)], 
#     sum(a[j,m]*(X[i,j] + ϵ[j]) for j=1:p) <= b[m] + (1 + ϵ_max)*(1 - z[i,t]) ); # (13)
@constraint(model, con13[i=1:n,t=1:TLeaf,m=left_ancestors(TBranch + t)], 
    sum(a[j,m]*X[i,j] for j=1:p) + 0.99*ϵ_min <= b[m] + (1 + ϵ_max)*(1 - z[i,t]) ); # (13)

@constraint(model, con20[k=1:K,t=1:TLeaf], L[t] >= Nt[t] - Nkt[k,t] - n*(1 - c[k,t]) ); # (20)
@constraint(model, con21[k=1:K,t=1:TLeaf], L[t] <= Nt[t] - Nkt[k,t] + n*c[k,t] ); # (21)
@constraint(model, con22[t=1:TLeaf], L[t] >= 0 ); # (22)

setvalue(a, a_grd)
setvalue(b, b_grd)

start = time();
#optimize!(model)
println("Starting solver")
status = solve(model) # version 0.18 of JuMP
println("Exiting solver")
elapsed = time() - start;
solvetime = getsolvetime(model)
# obj_value = objective_value(model);
obj_value = getobjectivevalue(model);
println("Solve time: ", solvetime)
println("Elapsed time: ", elapsed)
println("Objective=", obj_value);
println("Number of nodes: ", getnodecount(model))

# Retrieve variable values
a_opt = getvalue(a)
b_opt = getvalue(b)
c_opt = getvalue(c)

function predict_sample(x)
    m = 1
    while m <= TBranch
        node_m = a_opt[:,m]'*x >= b_opt[m]
        if node_m == true
            m = 2*m + 1
        else
            m = 2*m
        end
    end
    pred_class = findmax(c_opt[:,m-TBranch])[2]
    return pred_class
end

function predict(X)
    n_samples = size(X,1)
    y_pred = zeros(n_samples)
    for i=1:n_samples
        y_pred[i] = predict_sample(X[i,:])
    end
    return y_pred
end     

# Make prediction
y_pred = predict(X);

train_acc = sum(y .== y_pred) / length(y)
println("Train accuracy: $train_acc")
println("Train accuracy (Greedy): $train_acc_grd")


