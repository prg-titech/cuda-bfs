#include <cuda.h>
#include <limits>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <iterator>
#include <chrono>

using namespace std;

#define BLOCKS 128
#define THREADS 256

#define MAX_DIST 1073741824

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct graph_t
{
    int *v_adj_list;
    int *v_adj_begin;
    int *v_adj_length;  
    int num_vertices;    
    int num_edges;
};

bool report_time = false;

#include "bfs_cpu.cu"
#include "bfs_cuda_simple.cu"
#include "bfs_cuda_frontier.cu"
#include "bfs_cuda_frontier_numbers.cu"
#include "bfs_cuda_virtual_warp_centric.cu"
#include "bfs_cuda_per_edge_basic.cu"
#include "bfs_cuda_per_edge_frontier_numbers.cu"
#include "bfs_cuda_frontier_queue.cu"
#include "bfs_cuda_frontier_scan.cu"
#include "bfs_cuda_frontier_numbers_sort.cu"
#include "bfs_cuda_frontier_numbers_defer.cu"
#include "bfs_cuda_frontier_numbers_reorder.cu"

typedef int (* bfs_func)(int*, int*, int*, int, int, int, int*);

int runs;
#define RT_STUDY_NUM_V 20

int run_bfs(bfs_func func, graph_t *graph, int start_vertex, int *expected)
{
    int *result = new int[graph->num_vertices];

    int runtime = 0;

    if (start_vertex == -2)
    {
        // Performance study: Choose multiple start vertices and calculate avg
        double rt_sum = 0.0;

        for (int i = 0; i < RT_STUDY_NUM_V; i++)
        {
            int runtime = 0;

            for (int j = 0; j < runs; j++) 
            {
                int next_time = func(
                    graph->v_adj_list, 
                    graph->v_adj_begin, 
                    graph->v_adj_length, 
                    graph->num_vertices, 
                    graph->num_edges,
                    i, 
                    result);

                if (runtime == 0)
                {
                    runtime = next_time;
                }
                else
                {
                    runtime = min(next_time, runtime);
                }
            }

            rt_sum += runtime;
        }

        printf("%f\n", rt_sum / RT_STUDY_NUM_V);
        return (int) (rt_sum / RT_STUDY_NUM_V);
    }

    for (int i = 0; i < runs; i++)
    {
        int next_time = func(
            graph->v_adj_list, 
            graph->v_adj_begin, 
            graph->v_adj_length, 
            graph->num_vertices, 
            graph->num_edges,
            start_vertex, 
            result);

        if (runtime == 0)
        {
            runtime = next_time;
        }
        else
        {
            runtime = min(next_time, runtime);
        }

        if (!equal(result, result + graph->num_vertices, expected))
        {
            printf("Result incorrect\n");
            printf("DIFF:\n");
            for (int i = 0; i < graph->num_vertices; i++)
            {
                if (result[i] != expected[i])
                {
                    //printf("[%i] Expected %i, found %i\n", i, expected[i], result[i]);
                }
            }

            return -1;
        }
    }

    if (runs > 1)
    {
        printf("%i\n", runtime);
    }

    free(result);

    return runtime;
}

// 2nd version: Only active vertices are working
// 3rd version: Active vertices + load balancing
// 4th version: Virtual warp-centric programming
// 5th version: Hierarchical queue: http://impact.crhc.illinois.edu/shared/papers/effective2010.pdf

// ... version: Try to utilize shared memory

void run_all_bfs(graph_t *graph, int start_vertex)
{
    // Run BFS
    int *result = new int[graph->num_vertices];

    if (start_vertex != -2)
    {
        bfs_sequential(graph, start_vertex, result);
    }

    // Run BFS on CUDA
    if (!run_bfs(&bfs_cuda_simple, graph, start_vertex, result))
    {
        printf("bfs_cuda_simple failed\n");
        exit(1);
    }
    //printf("A");

    // Run BFS on CUDA - frontier
    if (!run_bfs(&bfs_cuda_frontier, graph, start_vertex, result))
    {
        printf("bfs_cuda_frontier failed\n");
        exit(1);
    }
    //printf("B");

    // Run BFS on CUDA - frontier as number
    if (!run_bfs(&bfs_cuda_frontier_numbers, graph, start_vertex, result))
    {
        printf("bfs_cuda_frontier_numbers failed\n");
        exit(1);
    }
    //printf("C");

    // Run BFS on CUDA - virtual warp centric
    if (!run_bfs(&bfs_cuda_virtual_wc, graph, start_vertex, result))
    {
        printf("bfs_cuda_virtual_wc failed\n");
        exit(1);
    }
    //printf("D");

    // Run BFS on CUDA - parallelize per edges
    if (!run_bfs(&bfs_cuda_per_edge_basic, graph, start_vertex, result))
    {
        printf("bfs_cuda_per_edge_basic failed\n");
        exit(1);
    }
    //printf("E");

    // Run BFS on CUDA - parallelize per edges, frontier as number
    if (!run_bfs(&bfs_cuda_per_edge_frontier_numbers, graph, start_vertex, result))
    {
        printf("bfs_cuda_per_edge_frontier_numbers failed\n");
        exit(1);
    }
    //printf("F");

    // Run BFS on CUDA - shared queue
    if (!run_bfs(&bfs_cuda_frontier_queue, graph, start_vertex, result))
    {
        printf("bfs_cuda_frontier_queue failed\n");
        exit(1);
    }
    //printf("G");

    // Run BFS on CUDA - explicit queue with prefix sum
    //run_bfs(&bfs_cuda_frontier_scan, graph, start_vertex, result);
    //printf("H");

    // Run BFS on CUDA - frontier as number + sorted
    if (!run_bfs(&bfs_cuda_frontier_numbers_sort, graph, start_vertex, result))
    {
        printf("bfs_cuda_frontier_numbers_sort failed\n");
        exit(1);
    }
    //printf("I");

    // Run BFS on CUDA - frontier as number + defer outliers
    if (!run_bfs(&bfs_cuda_frontier_numbers_defer, graph, start_vertex, result))
    {
        printf("bfs_cuda_frontier_numbers_defer failed\n");
        exit(1);
    }
    //printf("J");

    // Runs BFS on CUDA - frontier as number + reordered
    if (!run_bfs(&bfs_cuda_frontier_numbers_reorder, graph, start_vertex, result))
    {
        printf("bfs_cuda_simple failed\n");
        exit(1);
    }
    //printf("K");

    if (!report_time)
    {
        printf(".\n");
    }

    free(result);
    fflush(stdout);
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage %s filename vertex_id runs.\n", argv[0]);
        exit(1);
    }

    int start_vertex = atoi(argv[2]);
    runs = atoi(argv[3]);

    // Find number of vertices
    printf("Reading input file\n");
    ifstream infile(argv[1]);
    int from, to;
    int num_edges = 0;

    map<int, int> index_map;
    int next_index = 0;

    while (infile >> from >> to)
    {
        if (!index_map.count(from))
        {
            index_map[from] = next_index++;
        }

        if (!index_map.count(to))
        {
            index_map[to] = next_index++;
        }

        num_edges++;
    }

    int num_vertices = next_index;

    printf("Input file has %d vertices and %i edges\n", num_vertices, num_edges);

    // Build adajacency lists (still reading file)
    infile.clear();
    infile.seekg(0, ios::beg);

    int *v_adj_begin = new int[num_vertices];
    int *v_adj_length = new int[num_vertices];
    vector<int> *v_adj_lists = new vector<int>[num_vertices]();
    int *v_adj_list = new int[num_edges];

    int max_degree = 0;

    while (infile >> from >> to)
    {
        v_adj_lists[index_map[from]].push_back(index_map[to]);
        max_degree = max(max_degree, (int) v_adj_lists[index_map[from]].size());
    }

    // Show degree distribution
    printf("Compute out-degree histogram\n");
    int *degree_histogram = new int[max_degree + 1]();

    for (int i = 0; i < num_vertices; i++)
    {
        degree_histogram[v_adj_lists[i].size()]++;
    }

    printf("Histogram for Vertex Degrees\n");

    for (int i = 0; i < max_degree + 1; i++)
    {
        //printf("deg %i        %i\n", i, degree_histogram[i]);
    }

    // Generate data structure
    printf("Build ajacency lists\n");
    int next_offset = 0;

    for (int i = 0; i < num_vertices; i++)
    {
        int list_size = v_adj_lists[i].size();
        
        v_adj_begin[i] = next_offset;
        v_adj_length[i] = list_size;

        memcpy(v_adj_list + next_offset, &v_adj_lists[i][0], list_size * sizeof(int));
        next_offset += list_size;
    }

    graph_t *graph = new graph_t;
    graph->v_adj_list = v_adj_list;
    graph->v_adj_begin = v_adj_begin;
    graph->v_adj_length = v_adj_length;
    graph->num_vertices = num_vertices;
    graph->num_edges = num_edges;

    printf("Running...\n");
    if (start_vertex == -1)
    {
        // Test correctness for all start vertices
        for (int i = 0; i < graph->num_vertices; i++)
        {
            run_all_bfs(graph, i);
        }
    }
    else if (start_vertex == -2)
    {
        // Performance study
        run_all_bfs(graph, start_vertex);
    }
    else
    {
        report_time = runs == 1;
        run_all_bfs(graph, start_vertex);
    }

    printf("\n");
}