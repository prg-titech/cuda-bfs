#include <queue>

void bfs_sequential(
    graph_t *graph,
    int start_vertex, 
    int *result)
{
    bool *visited = new bool[graph->num_vertices];
    fill_n(visited, graph->num_vertices, 0);
    visited[start_vertex] = true;
    
    fill_n(result, graph->num_vertices, MAX_DIST);
    result[start_vertex] = 0;

    queue<int> next_vertices;
    next_vertices.push(start_vertex);

    while (!next_vertices.empty())
    {
        int vertex = next_vertices.front();
        next_vertices.pop();

        for (
            int n = graph->v_adj_begin[vertex]; 
            n < graph->v_adj_begin[vertex] + graph->v_adj_length[vertex]; 
            n++)
        {
            int neighbor = graph->v_adj_list[n];

            if (!visited[neighbor])
            {
                visited[neighbor] = true;
                result[neighbor] = result[vertex] + 1;
                next_vertices.push(neighbor);
            }
        }
    }
}
