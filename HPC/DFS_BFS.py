from queue import Queue
from threading import Thread
import time

class Graph:
    def __init__(self, V):
        self.V = V  # Number of vertices
        self.adj = [[] for _ in range(V)]  # Adjacency list

    # Add an edge to the graph
    def addEdge(self, v, w):
        self.adj[v].append(w)

    # Parallel Depth-First Search
    def parallelDFS(self, startVertex):
        visited = [False] * self.V
        dfs_result = []
        start_time = time.time()
        self.parallelDFSUtil(startVertex, visited, dfs_result)
        end_time = time.time()
        return dfs_result, end_time - start_time

    # Parallel DFS utility function
    def parallelDFSUtil(self, v, visited, dfs_result):
        visited[v] = True
        dfs_result.append(v)

        threads = []
        for n in self.adj[v]:
            if not visited[n]:
                t = Thread(target=self.parallelDFSUtil, args=(n, visited, dfs_result))
                t.start()
                threads.append(t)
        
        for t in threads:
            t.join()

    # Parallel Breadth-First Search
    def parallelBFS(self, startVertex):
        visited = [False] * self.V
        bfs_result = []
        q = Queue()

        visited[startVertex] = True
        q.put(startVertex)

        start_time = time.time()
        while not q.empty():
            v = q.get()
            bfs_result.append(v)

            threads = []
            for n in self.adj[v]:
                if not visited[n]:
                    visited[n] = True
                    q.put(n)

        end_time = time.time()
        return bfs_result, end_time - start_time

# Take input from user for the start vertex
start_vertex = int(input("Enter the start vertex (0 to 6): "))

# Create a graph
g = Graph(7)

# Take input from user for adding edges
num_edges = int(input("Enter the number of edges: "))
print("Enter edges in the format 'v w', where v and w are vertices (0 to 6):")
for _ in range(num_edges):
    v, w = map(int, input().split())
    g.addEdge(v, w)

dfs_result, dfs_time = g.parallelDFS(start_vertex)
print("Depth-First Search (DFS):", " ".join(map(str, dfs_result)))
print("Time taken for Depth-First Search (DFS):", dfs_time, "seconds")

bfs_result, bfs_time = g.parallelBFS(start_vertex)
print("Breadth-First Search (BFS):", " ".join(map(str, bfs_result)))
print("Time taken for Breadth-First Search (BFS):", bfs_time, "seconds")

''' Enter the start vertex (0 to 6): 0
Enter the number of edges: 6
Enter edges in the format 'v w', where v and w are vertices (0 to 6):
0 1
0 2
1 3
1 4
2 5
2 6
Depth-First Search (DFS): 0 1 3 4 2 5 6
Time taken for Depth-First Search (DFS): 0.00028967857360839844 seconds
Breadth-First Search (BFS): 0 1 2 3 4 5 6
Time taken for Breadth-First Search (BFS): 0.0 seconds
'''