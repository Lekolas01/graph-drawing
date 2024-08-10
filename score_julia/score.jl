using JSON3

# Structs to read the JSON file into
struct Point
    id::Int
    x::Float64
    y::Float64
end

struct Node
    id::Int
    x::Float64
    y::Float64
end

struct Edge
    source::Int
    target::Int
end

struct Graph
    nodes::Vector{Node}
    edges::Vector{Edge}
    points::Vector{Point}
end

# Enum representing the orientation of three points
@enum Orientation begin
    COLLINEAR = 0
    CLOCKWISE = 1
    COUNTERCLOCKWISE = 2
end
# Enumeration of all possible intersection types for two line segments
@enum Intersection begin
    NO_INTERSECTION = 0
    ENDPOINT_INTERSECTION = 1
    INTERSECTION = 2
    OVERLAP = 3
end

function str(g::Graph)::String
    return "Graph with $(length(g.nodes)) nodes, $(length(g.edges)) edges, and $(length(g.points)) points"
end

function orientation(p1::Array{Float64}, p2::Array{Float64}, p3::Array{Float64})::Orientation
    # https://www.geeksforgeeks.org/orientation-3-ordered-points/
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    val = (y2 - y1) * (x3 - x2) - (x2 - x1) * (y3 - y2)

    if val > 0
        return CLOCKWISE
    elseif val < 0 
        return COUNTERCLOCKWISE
    else
        return COLLINEAR
    end
end

function on_segment(p1::Array{Float64}, p2::Array{Float64}, p3::Array{Float64})::Bool
    # Check whether p3 is on the line segment defined by p1 and p2
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    if min(x1, x2) <= x3 <= max(x1, x2) && min(y1, y2) <= y3 <= max(y1, y2)
        return true
    else
        return false
    end
end

function inside_segment(p1::Array{Float64}, p2::Array{Float64}, p3::Array{Float64})::Bool
    # Check whether p3 is inside the line segment defined by p1 and p2
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    if min(x1, x2) < x3 < max(x1, x2) && min(y1, y2) < y3 < max(y1, y2)
        return true
    else
        return false
    end
end

function cross(p1::Array{Float64}, p2::Array{Float64}, p3::Array{Float64}, p4::Array{Float64})::Intersection
    # The line segments are assumed to be given by: l1=(p1, p2) and l2=(p3, p4)
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # compute the orientations of the points
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    # general case: the orientations have to be different for the lines to intersect
    if o1 != o2 && o3 != o4
        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment
        # compute the point of intersection and check if it is an endpoint of one line
        t_numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        s_numerator = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        t = t_numerator / denominator
        s = s_numerator / denominator

        # check if the intersection point is an endpoint of one of the lines
        if (abs(t) == 0 || abs(t) == 1) && (abs(s) == 0 || abs(s) == 1)
            return ENDPOINT_INTERSECTION
        else
            return INTERSECTION
        end
    end
    
    # special case: the lines are collinear
    if o1 == COLLINEAR && o2 == COLLINEAR && o3 == COLLINEAR && o4 == COLLINEAR
        # check if the lines overlap
        overlap_count = on_segment(p1, p2, p3) + on_segment(p1, p2, p4) + on_segment(p3, p4, p1) + on_segment(p3, p4, p2)
        if overlap_count == 0
            return NO_INTERSECTION
        elseif overlap_count == 2
            # the endpoint of one line and the start point of the other line are inside the corresponding other line.
            # E.g.: p3 in [p1, p2] and p2 in [p3, p4]
            # distinguish between an overlap or just two connecting lines
            if inside_segment(p1, p2, p3) || inside_segment(p1, p2, p4) || inside_segment(p3, p4, p1) || inside_segment(p3, p4, p2)
                return OVERLAP
            else
                return ENDPOINT_INTERSECTION
            end
        elseif overlap_count > 2 
            # one line starts inside the other and ends at the other's endpoint
            return OVERLAP
        end
    end
    
    return NO_INTERSECTION
end

function score(graph::Graph, solution::Vector)::Int
    # create a node mapping, coordinates of each node are at the index of the points
    pm = zeros(Float64, (length(graph.nodes), 2))
    for i in eachindex(solution)
        point = graph.points[solution[i]]
        pm[point.id+1, 1] = point.x  # arrays are 1-indexed
        pm[point.id+1, 2] = point.y
    end

    # compute the score for each pair of edges
    final_score = 0
    for edge_i_idx = 1:length(graph.edges)
        remaining_edges = graph.edges[edge_i_idx+1:end]
        edge_i = graph.edges[edge_i_idx]

        for edge_j in remaining_edges
            # retrieve the point coordinates of the two edges
            p1 = pm[edge_i.source+1, :]
            p2 = pm[edge_i.target+1, :]
            p3 = pm[edge_j.source+1, :]
            p4 = pm[edge_j.target+1, :]
            
            intersection = cross(p1, p2, p3, p4)
            crossing_score = 0
            if intersection == NO_INTERSECTION || intersection == ENDPOINT_INTERSECTION
                crossing_score = 0
            elseif intersection == INTERSECTION
                crossing_score = 1
            elseif intersection == OVERLAP
                crossing_score = size(pm, 1)
            end
            final_score += crossing_score
            
            #println("Edge $(edge_i) and $(edge_j) have $(crossing_score) crossings")
        end
    end

    return final_score
end

function main()
    problem_dir = "../problems/"
    for problem_file in readdir(problem_dir)
        problem_path = joinpath(problem_dir, problem_file)
        problem = JSON3.read(problem_path, Graph)

        # Inspecting Problem
        println("**** Problem: ", problem_file, "****")
        println(str(problem))
        
        # create a simple solution
        simple_solution = collect(1:length(problem.nodes))

        # Measure the execution time of the score function
        start_time = time()
        solution_score = score(problem, simple_solution)
        end_time = time()

        execution_time = end_time - start_time
        println("Score $(solution_score)")
        println("Score computation time: ", execution_time, " seconds")
        println("****\n")
    end
end

main()
