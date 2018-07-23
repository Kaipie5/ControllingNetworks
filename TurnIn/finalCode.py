import networkx as nx

import itertools

from networkx import Graph, find_cycle, NetworkXNoCycle
from networkx.algorithms import bipartite

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from networkx import MultiGraph




#nx.draw(graph)
###################### STRUCTURAL CONTROLLABILITY ################
#CREDIT TO YUHAO (Kind of. Rewrote almost every piece of code 
#she gave me but she at least sent me down the right path.)
def controllability(graph):
    newGraphA = nx.Graph()
    for a in graph.nodes():
        newGraphA.add_node(str(a) + "+")
    newGraphB = nx.Graph()
    for b in graph.nodes():
        newGraphB.add_node(str(b) + "-")
    undirected = nx.Graph()
    undirected.add_nodes_from(newGraphA.nodes(), bipartite=0)
    undirected.add_nodes_from(newGraphB.nodes(), bipartite=1)
    for src, dst in graph.edges():
        newSrc = str(src) + "+"
        newDst = str(dst) + "-"
        undirected.add_edge(newSrc, newDst)
    halfList = []
    for node in undirected.nodes():
        if node.endswith("-"):
            halfList.append(node)
    matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(undirected, halfList)
    unmatched = len(graph.nodes()) - (len(matching) / 2)
    G_source = []
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            G_source.append(node)
    control = (unmatched) / len(graph.nodes())
    #print("Structural Controllability: ", control)
    return control

################ MINIMUM FEEDBACK VERTEX SET CONTROL ###################
def graph_minus(graph, s):
    new_nodes = [x for x in graph.nodes() if x not in s]
    
    newGraph = graph.copy()
    for node in graph.nodes():
        #print(node)
        if node not in new_nodes:
            newGraph.remove_node(node)
    return newGraph



def is_acyclic(g: Graph):
    try:
        cycle = find_cycle(g)
    except NetworkXNoCycle:
        return True
    return False


def remove_node_deg_01(g):
    """
    Delete all nodes of degree 0 or 1 in a graph
    Return `True` if the graph was modified and `False` otherwise
    """
    for v in g.nodes():
        if g.degree(v) <= 1:
            g.remove_node(v)
            return True
    return False

def is_independent_set(g: MultiGraph, f: set) -> bool:
    for edge in itertools.combinations(f, 2):
        if g.has_edge(edge[0], edge[1]):
            return False
    return True
"""
Bruteforce to find the min size feedback vertex set
A true, dumb bruteforce that will guarantee to find the min fbvs
Just try all combinations of nodes in the graph, for each combination check if the induced
subgraph is acyclic
This takes no consideration whether a node is in a cycle or not
"""
#
#def get_fbvs(graph):
#    if is_acyclic(graph):
#        return set()
#
#    # remove all nodes of degree 0 or 1 as they can't be part of any cycles
#    remove_node_deg_01(graph)
#
#    nodes = graph.nodes()
#    for L in range(0, len(nodes) + 1):
#        for subset in itertools.combinations(nodes, L):
#            # make an induced subgraph with the current node subset removed
#            new_graph = graph_minus(graph, subset)
#
#            if is_acyclic(new_graph):
#                return subset
#
#    return set()
#"""
#Iterative Compression algorithm for the undirected feedback vertex set problem from chapter 4.3.1 of Parameterized
#Algorithms, Cygan, M., Fomin, F.V., Kowalik, Ł., Lokshtanov, D., Marx, D., Pilipczuk, M., Pilipczuk, M., Saurabh, S.
#This algorithm will, given a feedback vertex set instance (G, k), in time (5^k)*(n^O(1)) either reports a failure
#or finds a feedback vertex set in G of size at most k.
#Originally designed for the decision version of the problem (via the get_fbvs_max_size() method), this algorithm
#can also be used to solve the optimization version (via the get_fbvs() method).
#"""
#
def get_fbvs(graph):
    if is_acyclic(graph):
        return set()

    if type(graph) is not MultiGraph:
        graph = MultiGraph(graph)

    for i in range(1, graph.number_of_nodes()):
        result = get_fbvs_max_size(graph, i)
        if result is not None:
            return result  # in the worst case, result is n-2 nodes

def get_fbvs_max_size(g, k):
    if len(g) <= k + 2:
        return set(g.nodes()[:k])

    # Construct a trivial FVS of size k + 1 on the first k + 3 vertices of G.
    nodes = g.nodes()
    #print(nodes)

    fixedNodes = []
    for node in g.nodes():
        fixedNodes.append(node)
#    node_set = []
    # The set of nodes currently under consideration.
#    for i in range(0, k+2):
#        node_set.append(nodes[i])
    node_set = set(fixedNodes[:(k+2)])

    # The current best solution, of size (k + 1) before each compression step,
    # and size <= k at the end.
    otherFixedNodes = []
    for node in g.nodes():
        otherFixedNodes.append(node)

    soln = set(otherFixedNodes[:k])
    
    for i in range(k + 2, len(nodes)):
        soln.add(otherFixedNodes[i])
        node_set.add(fixedNodes[i])

        if len(soln) < k + 1:
            continue

#        assert (len(soln) == (k + 1))
        assert (len(node_set) == (i + 1))

        newGraph = graph.copy()
        for node in nodes:
            #print(node)
            if node not in node_set:
                newGraph.remove_node(node)
                
        new_soln = ic_compression(newGraph, soln, k)

        if new_soln is None:
            return None

        soln = new_soln
        assert (len(soln) <= k)

    return soln

def fvs_disjoint(g, w, k):
    """
    Given an undirected graph G and a fbvs W in G of size at least (k + 1), is it possible to construct
    a fbvs X of size at most k using only the nodes of G - W?
    :return: The set X, or `None` if it's not possible to construct X
    """

    # If G[W] isn't a forest, then a solution X not using W can't remove W's cycles.
    if not is_acyclic(g.subgraph(w)):
        return None

    # Apply reductions exhaustively.
    k, soln_redux = apply_reductions(g, w, k)

    # If k becomes negative, it indicates that the reductions included
    # more than k nodes. In other word, reduction 2 shows that there are more than k nodes
    # in G - W that will create cycle in W. Hence, no solution of size <= k exists.
    if k < 0:
        return None

    # From now onwards we assume that k >= 0

    # If G has been reduced to nothing and k is >= 0 then the solution generated by the reductions
    # is already optimal.
    if len(g) == 0:
        return soln_redux

    # Recall that H is a forest as W is a feedback vertex set. Thus H has a node x of degree at most 1.
    # Find an x in H of degree at most 1.
    h = graph_minus(g, w)
    x = None
    for v in h.nodes():
        if h.degree(v) <= 1:
            x = v
            break
    assert x is not None, "There must be at least one node x of degree at most 1"

    # Branch on (G - {x}, W, k−1) and (G, W ∪ {x}, k)
    # G is copied in the left branch (as it is modified), but passed directly in the right.
    soln_left = fvs_disjoint(graph_minus(g, {x}), w, k - 1)

    if soln_left is not None:
        return soln_redux.union(soln_left).union({x})

    soln_right = fvs_disjoint(g, w.union({x}), k)

    if soln_right is not None:
        return soln_redux.union(soln_right)

    return None

def ic_compression(g, z, k):
    """
    Given a graph G and an FVS Z of size (k + 1), construct an FVS of size at most k.
    Return `None` if no such solution exists.
    """
    assert (len(z) == k + 1)
    # i in {0 .. k}
    for i in range(0, k + 1):
        for xz in itertools.combinations(z, i):
            x = fvs_disjoint(graph_minus(g, xz), z.difference(xz), k - i)
            if x is not None:
                return x.union(xz)
    return None

def reduction1(g, w, h, k):
    """
    Delete all nodes of degree 0 or 1 as they can't be part of any cycles.
    """
    changed = False
    baseGraph = g.copy()
    for v in baseGraph.nodes():
        if g.degree(v) <= 1:
            g.remove_node(v)
            h.remove_nodes_from([v])
            changed = True
    return k, None, changed

def reduction2(g, w, h, k):
    """
    If there exists a node v in H such that G[W ∪ {v}]
    contains a cycle, then include v in the solution, delete v and decrease the
    parameter by 1. That is, the new instance is (G - {v}, W, k - 1).
    If v introduces a cycle, it must be part of X as none of the vertices in W
    will be available to neutralise this cycle.
    """
    for v in h.nodes():
        # Check if G[W ∪ {v}] contains a cycle.
        if not is_acyclic(g.subgraph(w.union({v}))):
            g.remove_node(v)
            h.remove_nodes_from([v])
            return k - 1, v, True
    return k, None, False

def reduction3(g, w, h, k):
    """
    If there is a node v ∈ V(H) of degree 2 in G such
    that at least one neighbor of v in G is from V (H), then delete this node
    and make its neighbors adjacent (even if they were adjacent before; the graph
    could become a multigraph now).
    """
    for v in h.nodes():
        if g.degree(v) == 2:
            # If v has a neighbour in H, short-curcuit it.
            if len(h[v]) >= 1:
                # Delete v and make its neighbors adjacent.
                [n1, n2] = g.neighbors(v)
                g.remove_node(v)
                g.add_edge(n1, n2)
                # Update H accordingly.
                h.remove_nodes_from([v])
                if n1 not in w and n2 not in w:
                    h.add_edge(n1, n2)
                return k, None, True
    return k, None, False

def apply_reductions(g, w, k):
    """
    Exhaustively apply reductions. The three reductions are:
    Reduction 1: Delete all the nodes of degree at most 1 in G.
    Reduction 2: If there exists a node v in H such that G[W ∪ {v}]
        contains a cycle, then include v in the solution, delete v and decrease the
        parameter by 1. That is, the new instance is (G - {v}, W, k - 1).
    Reduction 3: If there is a node v ∈ V(H) of degree 2 in G such
        that at least one neighbor of v in G is from V(H), then delete this node
        and make its neighbors adjacent (even if they were adjacent before; the graph
        could become a multigraph now).
    """
    # Current H.
    h = graph_minus(g, w)

    # Set of nodes included in the solution as a result of reductions.
    x = set()
    while True:
        reduction_applied = False
        for f in [reduction1, reduction2, reduction3]:
            (k, solx, changed) = f(g, w, h, k)

            if changed:
                reduction_applied = True
                if solx is not None:
                    x.add(solx)

        if not reduction_applied:
            return k, x

#from networkx import number_connected_components, connected_components
#
#"""
#Maximum Induced Forest algorithm for the undirected feedback vertex set problem from chapter 6.2 of
#Exact Exponential Algorithms by Fomin, Fedor V., Kratsch, Dieter
#Instead of computing a minimum feedback vertex set directly, this algorithm finds the maximum size
#of an induced forest in a graph. In fact, it solves a more general problem: for any
#acyclic set F it finds the maximum size of an induced forest containing F.
#"""
#
#def get_fbvs(graph: Graph):
#    if is_acyclic(graph):
#        return set()
#
#    # Save the original node set for later use since we'll mutate the graph
#    nodes = set(graph.nodes())
#
#    if type(graph) is not MultiGraph:
#        graph = MultiGraph(graph)
#
#    mif_set = preprocess_1(graph, set(), None)
#    if mif_set is not None:
#        fbvs = nodes.difference(mif_set)
#        return fbvs
#
#    return None
#
#def get_fbvs_max_size(graph: Graph, k: int) -> set:
#    raise Exception("Undefined for this algorithm")
#
#def preprocess_1(g: MultiGraph, f: set, active_v) -> set:
#    if number_connected_components(g) >= 2:
#        mif_set = set()
#        for component in connected_components(g):
#            f_i = component.intersection(f)
#            gx = g.subgraph(component)
#            component_mif_set = preprocess_2(gx, f_i, active_v)
#            if component_mif_set:
#                mif_set = mif_set.union(component_mif_set)
#        return mif_set
#    return preprocess_2(g, f, active_v)
#
#def preprocess_2(g: MultiGraph, f: set, active_v) -> set:
#    mif_set = set()
#    while not is_independent_set(g, f):
#        mif_set = mif_set.union(f)
#        for component in connected_components(g.subgraph(f)):
#            if len(component) > 1:
#                if active_v in component:
#                    active_v = component.pop()
#                    compressed_node = active_v
#                else:
#                    compressed_node = component.pop()
#                g = compress(g, component, compressed_node, True)
#                f = f.intersection(g.nodes())
#                # Maybe faster with
#                # f = f.difference(component)
#                # f.add(compressed_node)
#                mif_set = mif_set.union(component)
#                break
#    mif_set2 = mif_main(g, f, active_v)
#    if mif_set2:
#        mif_set = mif_set2.union(mif_set)
#
#    return mif_set
#
#def compress(g: MultiGraph, t: set, compressed_node, mutate=False) -> MultiGraph:
#    if not t:
#        return g
#    if mutate:
#        gx = g
#    else:
#        gx = g.copy()
#
#    tx = t
#    if compressed_node in tx:
#        tx = t.copy()
#        tx.remove(compressed_node)
#    gx.add_node(compressed_node)
#
#    for node in tx:
#        for edge in gx.edges(node):
#            if edge[0] == node:
#                node_2 = edge[1]
#            else:
#                node_2 = edge[0]
#            if not (node_2 in t or node_2 == compressed_node):
#                gx.add_edge(compressed_node, node_2)
#        gx.remove_node(node)
#
#    remove = set()
#    for node in gx.adj[compressed_node]:
#        if len(gx.adj[compressed_node][node]) >= 2:
#            # Using a set to remove to avoid messing up iteration of adj
#            remove.add(node)
#
#    for node in remove:
#        gx.remove_node(node)
#
#    return gx
#
#def generalized_degree(g: MultiGraph, f: set, active_node, node) -> (int, set):
#    assert g.has_node(node), "Calculating gd for node which is not in g!"
#
#    k = set(g.neighbors(node))
#    k.remove(active_node)
#    k = k.intersection(f)
#
#    gx = compress(g, k, node)
#
#    neighbors = gx.neighbors(node)
#    neighbors.remove(active_node)
#
#    return len(neighbors), neighbors
#
#def mif_main(g: MultiGraph, f: set, t) -> set:
#    if f == g.nodes():
#        return f
#    if not f:
#        g_degree = g.degree()
#        g_max_degree_node = max(g_degree, key=lambda n: g_degree[n])
#        if g_degree[g_max_degree_node] <= 1:
#            return set(g.nodes())
#        else:
#            fx = f.copy()
#            fx.add(g_max_degree_node)
#            gx = g.copy()
#            gx.remove_node(g_max_degree_node)
#            mif_set1 = preprocess_1(g, fx, t)
#            mif_set2 = preprocess_1(gx, f, t)
#            if not mif_set1:
#                return mif_set2
#            elif not mif_set2:
#                return mif_set1
#            else:
#                return max(mif_set1, mif_set2, key=len)
#
#    # Set t as active vertex
#    if t is None or t not in f:
#        t = next(iter(f))
#
#    gd_over_3 = None
#    gd_2 = None
#    for v in g.neighbors_iter(t):
#        (gd_v, gn_v) = generalized_degree(g, f, t, v)
#        if gd_v <= 1:
#            f.add(v)
#            return preprocess_1(g, f, t)
#        elif gd_v >= 3:
#            gd_over_3 = v
#        else:
#            gd_2 = (v, gn_v)
#    if gd_over_3 is not None:
#        # Cannot simply use "if gd_over_3" because v might be 0
#        fx = f.copy()
#        fx.add(gd_over_3)
#        gx = g.copy()
#        gx.remove_node(gd_over_3)
#        mif_set1 = preprocess_1(g, fx, t)
#        mif_set2 = preprocess_1(gx, f, t)
#        if not mif_set1:
#            return mif_set2
#        elif not mif_set2:
#            return mif_set1
#        else:
#            return max(mif_set1, mif_set2, key=len)
#    elif gd_2 is not None:
#        (v, gn) = gd_2
#        fx1 = f.copy()
#        fx2 = f.copy()
#        fx1.add(v)
#        for n in gn:
#            fx2.add(n)
#        gx = g.copy()
#        gx.remove_node(v)
#        try:
#            find_cycle(gx.subgraph(fx2))
#            mif_set1 = None
#        except:
#            mif_set1 = preprocess_1(gx, fx2, t)
#        mif_set2 = preprocess_1(g, fx1, t)
#        if not mif_set1:
#            return mif_set2
#        elif not mif_set2:
#            return mif_set1
#        else:
#            return max(mif_set1, mif_set2, key=len)
#    return None
#
#from networkx import cycle_basis
#
#"""
#A mix of Bruteforce and BruteforceCycle, where we just bruteforce those nodes in at least one cycle
#"""
#
#def get_fbvs(graph):
#    if is_acyclic(graph):
#        return set()
#
#    # remove all nodes of degree 0 or 1 as they can't be part of any cycles
#    remove_node_deg_01(graph)
#
#    # get the set of nodes that is in at least one cycle
#    cycles = cycle_basis(graph)
#    nodes_in_cycles = set([item for sublist in cycles for item in sublist])
#
#    for L in range(0, len(nodes_in_cycles) + 1):
#        for subset in itertools.combinations(nodes_in_cycles, L):
#            # make an induced subgraph with the current node subset removed
#            new_graph = graph_minus(graph, subset)
#
#            if is_acyclic(new_graph):
#                return subset
#
#    return set()
#
#Look at all in edges. If a node has no in edges it is a source node
def findSourceNodes(graph):
    numSourceNodes = 0
    for node in graph.nodes:
        numInEdges = 0
        for a,b in graph.edges():
            if b == node:
                numInEdges = numInEdges + 1
        if numInEdges == 0:
            numSourceNodes = numSourceNodes + 1
            
    return numSourceNodes


#graph = nx.read_edgelist("toy_copy.txt", create_using=nx.DiGraph())
#graph = nx.read_edgelist("airportNetwork.txt", create_using=nx.DiGraph())
#graph = nx.read_edgelist("facebookCaltech.txt", create_using=nx.DiGraph())
#graph = nx.read_edgelist("fig3.txt", create_using=nx.DiGraph())
#graph = nx.gnm_random_graph(10, 10, seed=None, directed=True)

realWorldSC = []
realWorldNFC = []

##### Airport Network ######
graph = nx.read_edgelist("BIOHELP.txt", create_using=nx.DiGraph())

ogGraph = graph
control = controllability(graph)

realWorldSC.append(control)
print("structuralControllability: ", control)

graph = nx.read_edgelist("BIOHELP.txt", create_using=nx.Graph())

minVertexSetSize = len(get_fbvs(graph))
print("minimum vertex Set Size: ", minVertexSetSize)

sourceNodes = findSourceNodes(graph)
print("num Source Nodes: ", sourceNodes)

feedBackVertexSetControl = 0
if minVertexSetSize is not None:
    feedBackVertexSetControl = (minVertexSetSize + sourceNodes)/len(ogGraph.nodes)
    print(len(ogGraph.nodes()))
    print("feedBackVertexSetControl: ", feedBackVertexSetControl)

#realWorldNFC.append(feedBackVertexSetControl)
######FACEBOOK#####
#graph = nx.read_edgelist("facebookCaltech.txt", create_using=nx.DiGraph())
#
#ogGraph = graph
#control = controllability(graph)
#
#realWorldSC.append(control)
#print("structuralControllability: ", control)
#minVertexSetSize = len(get_fbvs(graph))
#print("minimum vertex Set Size: ", minVertexSetSize)
#
#sourceNodes = findSourceNodes(graph)
#print("num Source Nodes: ", sourceNodes)
#
#feedBackVertexSetControl = 0
#if minVertexSetSize is not None:
#    feedBackVertexSetControl = (minVertexSetSize + sourceNodes)/len(ogGraph.nodes)
#    print(len(ogGraph.nodes()))
#    print("feedBackVertexSetControl: ", feedBackVertexSetControl)
#
#realWorldNFC.append(feedBackVertexSetControl)

########WIKI VOTE######
#graph = nx.read_edgelist("wiki-Vote.txt", create_using=nx.DiGraph())
#
#ogGraph = graph
#control = controllability(graph)
#
#realWorldSC.append(control)
#print("structuralControllability: ", control)
#minVertexSetSize = len(get_fbvs(graph))
#print("minimum vertex Set Size: ", minVertexSetSize)
#
#sourceNodes = findSourceNodes(graph)
#print("num Source Nodes: ", sourceNodes)
#
#feedBackVertexSetControl = 0
#if minVertexSetSize is not None:
#    feedBackVertexSetControl = (minVertexSetSize + sourceNodes)/len(ogGraph.nodes)
#    print(len(ogGraph.nodes()))
#    print("feedBackVertexSetControl: ", feedBackVertexSetControl)
#
#realWorldNFC.append(feedBackVertexSetControl)


#numNodes = []
#numEdges = []
#numSuccesses = []
#feedBackVertexSetList = []
#StructuralControllabilityList = []
#StructuralControllabilityListSuccess = []
#feedBackVertexSetListSuccess = []
#for i in range (30, 31):
#    successfulGraphs = []
#    for p in range(0, 10000): 
#        graph = nx.gnm_random_graph(15, i, seed=None, directed=True)
#        OGGraph = graph
#        
#        
#        #graph = nx.read_edgelist("toy_copy.txt", create_using=nx.DiGraph())
#        #graph = nx.read_edgelist("airportNetwork.txt", create_using=nx.DiGraph())
#        #graph = nx.read_edgelist("fig3.txt", create_using=nx.DiGraph())
#        
#        control = controllability(graph)
#        
#        minVertexSetSize = len(get_fbvs(graph))
#        #print("minimum vertex Set Size: ", minVertexSetSize)
#        
#        sourceNodes = findSourceNodes(graph)
#        #print("num Source Nodes: ", sourceNodes)
#        
#        feedBackVertexSetControl = 0
#        if minVertexSetSize is not None:
#            feedBackVertexSetControl = (minVertexSetSize + sourceNodes)/len(OGGraph.nodes)
#            #print("feedBackVertexSetControl: ", feedBackVertexSetControl)
#            
#        
#        if control > 0.5 and feedBackVertexSetControl > 0.5:
#            successfulGraphs.append(graph)
#            feedBackVertexSetListSuccess.append(feedBackVertexSetControl)
#            StructuralControllabilityListSuccess.append(control)
#        else:
#            feedBackVertexSetList.append(feedBackVertexSetControl)
#            StructuralControllabilityList.append(control)
#    
#    print("NUM SUCCESSES for: ", i, "WITH THIS MANY SUCCESSES: ", len(successfulGraphs))
#    numEdges.append(i)
#    numSuccesses.append(len(successfulGraphs))
#
#print(numEdges)
#print(numSuccesses)
#plt.plot(numEdges, numSuccesses)
#plt.xlabel('NumEdges')
#plt.ylabel('NumSuccesses')
#plt.title("Success falloff based on Number of Edges")
#plt.show()

#plot1 = plt.scatter(feedBackVertexSetList, StructuralControllabilityList, color='black')
#plot2 = plt.scatter(feedBackVertexSetListSuccess, StructuralControllabilityListSuccess, color='blue')
#plot3 = plt.scatter(realWorldNFC, realWorldSC, color='red')
#plt.xlabel('Feedback vertex set control')
#plt.ylabel('Structural Controllability')
#black = mpatches.Patch(color='black', label='RandomUnsuccessful')
#blue = mpatches.Patch(color='blue', label='RandomSuccessful')
#red = mpatches.Patch(color='red', label='RealWorldNetworks')
#plt.legend(handles=[black, blue, red])
#plt.ylim(0,1.0)
#plt.xlim(0,1.0)
#plt.show()