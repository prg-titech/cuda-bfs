#include <cuda.h>
#include <limits>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <iterator>
#include <sys/time.h>
#include <assert.h>

using namespace std;

int BLOCKS = 0;
int THREADS = 0;

#define MAX_DIST 1073741824
#define MAX_KERNEL_RUNS 2048

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

int report_time = false;

/* in milliseconds (ms) */
long get_elapsed_time(struct timeval *begin, struct timeval *end)
{
    return (end->tv_sec - begin->tv_sec) * 1000 * 1000
            + (end->tv_usec - begin->tv_usec); /// 1000.0;
}


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

int run_bfs(bfs_func func, graph_t *graph, int start_vertex, int *expected, int runs)
{
    int *result = new int[graph->num_vertices];

    int runtime = 1073741824;

    for (int i = 0; i < runs; i++)
    {
        // Reset result array
        memset(result, 0, sizeof(int) * graph->num_vertices);

        int next_time = func(
            graph->v_adj_list, 
            graph->v_adj_begin, 
            graph->v_adj_length, 
            graph->num_vertices, 
            graph->num_edges,
            start_vertex, 
            result);

        runtime = min(next_time, runtime);
        
        if (!equal(result, result + graph->num_vertices, expected))
        {
            // Wrong result
            return -1;
        }
    }

    free(result);

    return runtime;
}

void run_all_bfs(graph_t *graph, int start_vertex)
{
    const bfs_func bfs_functions[] = {
        &bfs_cuda_simple, 
        &bfs_cuda_frontier, 
        &bfs_cuda_frontier_numbers, 
        &bfs_cuda_virtual_wc, 
        &bfs_cuda_per_edge_basic, 
        &bfs_cuda_per_edge_frontier_numbers, 
        &bfs_cuda_frontier_scan, 
        &bfs_cuda_frontier_numbers_sort, 
        &bfs_cuda_frontier_numbers_defer, 
        &bfs_cuda_frontier_numbers_reorder,
        &bfs_cuda_frontier_queue};

    string bfs_names[] = {
        "bfs_cuda_simple",
        "bfs_cuda_frontier", 
        "bfs_cuda_frontier_numbers", 
        "bfs_cuda_virtual_wc", 
        "bfs_cuda_per_edge_basic", 
        "bfs_cuda_per_edge_frontier_numbers", 
        "bfs_cuda_frontier_scan", 
        "bfs_cuda_frontier_numbers_sort", 
        "bfs_cuda_frontier_numbers_defer", 
        "bfs_cuda_frontier_numbers_reorder",
        "bfs_cuda_frontier_queue"};
    
    int num_bfs = sizeof(bfs_functions) / sizeof(*bfs_functions);
    double *runtime = new double[num_bfs]();
    bool *wrong_result = new bool[num_bfs]();

    int range_from, range_to;
    if (start_vertex == -1)
    {
        // Run for all start vertices
        range_from = 0;
        range_to = graph->num_vertices;
    }
    else if (start_vertex == -2)
    {
        // Run for RT_STUDY_NUM_V many vertices
        range_from = 0;
        range_to = RT_STUDY_NUM_V;
    }
    else
    {
        range_from = start_vertex;
        range_to = start_vertex + 1;
    }

    
    int *expected = new int[graph->num_vertices];

    for (int vertex = range_from; vertex < range_to; vertex++)
    {
        bfs_sequential(graph, vertex, expected);

        for (int i = 0; i < num_bfs; i++)
        {
            int next_runtime = run_bfs(bfs_functions[i], graph, vertex, expected, runs);

            if (next_runtime == -1)
            {
                // Wrong result
                wrong_result[i] = true;
            }
            else
            {
                runtime[i] += next_runtime;
            }
        }
    }

    for (int i = 0; i < num_bfs; i++)
    {
        double avg_runtime = runtime[i] / (range_to - range_from);

        if (!wrong_result[i])
        {
            printf("%s,%i,%i,%f\n", bfs_names[i].c_str(), BLOCKS, THREADS, avg_runtime);
        }
        else
        {
            printf("%s,%i,%i,-1\n", bfs_names[i].c_str(), BLOCKS, THREADS);
        }
    }

    free(expected);
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        printf("Usage: %s filename vertex_id runs blocks threads.\n", argv[0]);
        exit(1);
    }

    int start_vertex = atoi(argv[2]);
    runs = atoi(argv[3]);
    BLOCKS = atoi(argv[4]);
    THREADS = atoi(argv[5]);

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

    /*
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
        printf("deg %i        %i\n", i, degree_histogram[i]);
    }
    */

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
    run_all_bfs(graph, start_vertex);

    printf("\n");
}