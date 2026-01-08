from collections import defaultdict, deque
from heapq import heappop, heappush
from typing import Callable, Hashable, Iterable, Iterator



Node = Hashable

def search(
    start: Iterable[Node],
    find_next: Callable[[Node], Iterable[Node]],
    found: Callable[[Node], bool] | None = None,
    stop: Callable[[Node], bool] | None = None,
) -> Iterator[Node]:
    """
    Depth first search.

    Parameters
    ----------
    start : Iterable[Node]
        Initial nodes to start searching from
    find_next : Callable(Node) -> Iterable[Node]
        Generate all candidates for the next node in the search
    found : Callable(Node) -> bool
        Whether or not a given node is a solution
    stop : Callable(Node) -> bool
        Whether or not to stop searching from a given node
    """
    stack = deque(start)
    get_node, visit_nodes = stack.pop, stack.extend
    while stack:
        node = get_node()
        if found is None or found(node):
            yield node
        if stop is None or not stop(node):
            visit_nodes(find_next(node))

def dijkstra(
    source: Node,
    neighbors_fn: Callable[[Node], Iterable[Node]],
    edge_weight_fn: Callable[[Node, Node], float],
) -> tuple[dict[Node, float], dict[Node, Node]]:
    """
    Use Dijkstra's algorithm to find the shortest path from a source node
    to all other nodes in a graph.

    Parameters
    ----------
    source : Node
        Source node
    neighbors_fn : Callable(Node) -> Iterable[Node]
        Function that returns all neighbors of a given node.
    edge_weight_fn : Callable(Node, Node) -> float
        Function that returns the weight of the edge between two nodes.

    Returns
    -------
    dist : dict[Node, float]
        Shortest distance from the source node to each node
    prev : dict[Node, Node]
        Predecessor of each node in the shortest path
    """
    dist = defaultdict(lambda: float('inf'), {source: 0})
    prev = defaultdict(lambda: None)
    queue = [(0, source)]
    while queue:
        dist_u, u = heappop(queue)
        if dist_u > dist[u]:
            continue
        for v in neighbors_fn(u):
            alt = dist[u] + edge_weight_fn(u, v)
            if alt < dist[v]:
                dist[v], prev[v] = alt, u
                heappush(queue, (alt, v))

    return dist, prev

def find_cycles(
    find_next: Callable[[tuple[Node, ...]], Iterable[Node]] = lambda path: [],
    current_path: list[Node] | None = None,
) -> Iterator[list[Node]]:
    """
    Find cycles in a directed graph.

    Parameters
    ----------
    find_next : Callable(tuple[Node, ...]) -> Iterable[Node]
        Function that returns all candidates for the next node in the path
    current_path : list[Node]
        Current path in the graph
    """
    start_path = tuple(current_path or ())
    find_next_path = lambda path: (path + (v,) for v in find_next(path))
    found = lambda path: path and path[-1] in path[:-1]
    for cycle in search([start_path], find_next_path, found=found, stop=found):
        yield cycle[cycle.index(cycle[-1]):-1]

def find_functional_cycles(
    f: Callable[[int], int],
    search: Iterable[int],
    domain: range,
    on_cycle: Callable[[int, int], None],
):
    """
    Find cycles in the functional graph defined by f(n).

    Parameters
    ----------
    f : Callable(int) -> int
        Function defining the graph
    search : Iterable[int]
        Starting points to search for cycles
    domain : range
        Range of valid nodes in the graph
    on_cycle : Callable(cycle_start: int, cycle_node: int)
        Callback function for when a cycle is found,
        called for each node in the cycle
    """
    if domain.step != 1:
        raise ValueError("domain must have step size 1")

    low = domain.start
    cycle_id = [None] * len(domain)
    for start in search:
        # Advance until we find a cycle
        x, i = start, start - low
        while x in domain and cycle_id[i] is None:
            cycle_id[i], x = start, f(x)
            i = x - low

        # If this is a new cycle, walk through it
        if x in domain and cycle_id[i] == start:
            y = x
            on_cycle(x, y)
            while (y := f(y)) != x:
                on_cycle(x, y)

def topological_sort(graph: dict[Node, Iterable[Node]]) -> list[Node]:
    """
    Perform a topological sort on a directed acyclic graph (DAG).
    Uses depth-first search.

    Parameters
    ----------
    graph : dict[Node, Iterable[Node]]
        Graph represented as an adjacency list
    """
    visited, current_path, order = set(), set(), []
    nodes = set(graph.keys()).union(*(set(neighbors) for neighbors in graph.values()))
    for start in nodes:
        if start in visited:
            continue

        # Maintain a stack of (node, state) tuples
        # where state takes on values: 0 = enter, 1 = exit
        stack = [(start, 0)]
        while stack:
            v, state = stack.pop()
            if state == 0:
                # Skip visited nodes and detect cycles
                if v in visited:
                    continue
                if v in current_path:
                    raise ValueError("Detected cycle in graph.")

                # Schedule exit and push neighbors
                current_path.add(v)
                stack.append((v, 1))
                for u in graph.get(v, ()):
                    if u not in visited:
                        stack.append((u, 0))
            else:
                # Add node to topological ordering
                current_path.remove(v)
                visited.add(v)
                order.append(v)

    order.reverse()
    return order

def bron_kerbosch(
    graph: dict[Node, set[Node]],
    R: set[Node] | None = None,
    P: set[Node] | None = None,
    X: set[Node] | None = None,
) -> list[set[Node]]:
    """
    Recursive implementation of the Bron-Kerbosch algorithm
    for finding maximal cliques.

    Parameters
    ----------
    graph : dict[Node, set[Node]]
        Graph represented as an adjacency list
    R : set[Node]
        Current clique
    P : set[Node]
        Nodes that can be added to clique
    X : set[Node]
        Nodes to be excluded from clique

    Returns
    -------
    maximal_cliques : list[set[Node]]
        List of maximal cliques in the graph
    """
    R = set() if R is None else R
    P = set(graph.keys()) if P is None else P
    X = set() if X is None else X

    maximal_cliques = []
    if not P and not X:
        maximal_cliques.append(R)

    # Choose pivot node u to maximize |P ∩ N(u)|
    u = max(P | X, key=lambda v: len(graph[v]), default=None)
    candidates = P - (graph[u] if u is not None else set())

    # Explore candidates
    for v in candidates:
        maximal_cliques += bron_kerbosch(graph, R | {v}, P & graph[v], X & graph[v])
        P = P - {v}
        X = X | {v}

    return maximal_cliques

def kruskal(
    nodes: Iterable[Node],
    edges: Iterable[tuple[Node, Node]],
    get_edge_weight: Callable[[Node, Node], float],
) -> list[tuple[Node, Node]]:
    """
    Use Kruskal's algorithm to find a minimum spanning tree.

    Parameters
    ----------
    nodes : Iterable[Node]
        Nodes in the graph
    edges : Iterable[tuple[Node, Node]]
        Edges in the graph
    get_edge_weight : Callable(Node, Node) -> float
        Function that returns the weight of the edge between two nodes

    Returns
    -------
    minimum_spanning_tree : list[tuple[Node, Node]]
        Edges in the minimum spanning tree
    """
    parent, rank = {v: v for v in nodes}, defaultdict(int)

    # Path compression
    def find(x: Node) -> Node:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    # Union by rank
    def union(x: Node, y: Node) -> bool:
        root_x, root_y = find(x), find(y)
        if root_x == root_y:
            return False
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1
        return True

    minimum_spanning_tree = []
    for u, v in sorted(edges, key=lambda edge: get_edge_weight(*edge)):
        if union(u, v):
            minimum_spanning_tree.append((u, v))

    return minimum_spanning_tree
