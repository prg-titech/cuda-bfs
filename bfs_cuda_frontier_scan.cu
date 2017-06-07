// Accelerating large graph algorithms on the GPU using CUDA
// http://dl.acm.org/citation.cfm?id=1782200

#include <math.h>
//#include <cutil.h>

// includes, kernels
#include "nvidia/scan.cu"  // defines prescanArray()

__global__ void kernel_cuda_frontier_scan_main(
    int *v_adj_list,
    int *v_adj_begin,
    int *v_adj_length,
    int num_vertices,
    int *result,
    int *queue,
    bool *updated,
    bool *visited)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    int input_queue_size = queue[0];

    for (int v = 0; v < input_queue_size; v += num_threads)
    {
        if (v + tid < input_queue_size)
        {
            int vertex = queue[v + tid + 1];

            for (int n = 0; n < v_adj_length[vertex]; n++)
            {
                int neighbor = v_adj_list[v_adj_begin[vertex] + n];

                if (!visited[neighbor])
                {
                    result[neighbor] = result[vertex] + 1;
                    updated[neighbor] = true;
                }
            }
        }
    }
}

__global__ void kernel_cuda_frontier_scan(
    int num_vertices,
    bool *updated,
    bool *frontier,
    bool *visited,
    int *prefix_in_array)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int v = 0; v < num_vertices; v += num_threads)
    {
        int vertex = v + tid;

        if (vertex < num_vertices)
        {
            if (updated[vertex])
            {
                visited[vertex] = true;
            }

            frontier[vertex] = updated[vertex];
            prefix_in_array[vertex] = updated[vertex];

            updated[vertex] = false;
        }
    }
}

__global__ void kernel_cuda_generate_queue(
    int *prefix_sum,
    bool *frontier,
    int *queue,
    int num_vertices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int v = 0; v < num_vertices; v += num_threads)
    {
        int vertex = v + tid;

        if (vertex < num_vertices && frontier[vertex])
        {
            queue[prefix_sum[vertex] + 1] = vertex;
        }
    }

    // Set size of queue
    if (tid == 0)
    {
        queue[0] = prefix_sum[num_vertices - 1] + (int) frontier[num_vertices - 1];
    }
}

int bfs_cuda_frontier_scan(
    int *v_adj_list,
    int *v_adj_begin, 
    int *v_adj_length, 
    int num_vertices, 
    int num_edges,
    int start_vertex, 
    int *result)
{
    int *k_v_adj_list;
    int *k_v_adj_begin;
    int *k_v_adj_length;
    int *k_result;
    bool *k_updated;
    bool *k_frontier;
    bool *k_visited;
    int *k_queue;
    int *k_prefix_sum;
    int *k_prefix_in_array;

    int kernel_runs = 0;

    int *prefix_sum = new int[num_vertices];

    bool *updated = new bool[num_vertices];
    fill_n(updated, num_vertices, false);

    bool *visited = new bool[num_vertices];
    fill_n(visited, num_vertices, false);
    visited[start_vertex] = true;

    bool *frontier = new bool[num_vertices];
    fill_n(frontier, num_vertices, false);
    frontier[start_vertex] = true;

    fill_n(result, num_vertices, MAX_DIST);
    result[start_vertex] = 0;

    int *queue = new int[num_vertices];

    cudaMalloc(&k_v_adj_list, sizeof(int) * num_edges);
    cudaMalloc(&k_v_adj_begin, sizeof(int) * num_vertices);
    cudaMalloc(&k_v_adj_length, sizeof(int) * num_vertices);
    cudaMalloc(&k_result, sizeof(int) * num_vertices);
    cudaMalloc(&k_updated, sizeof(bool) * num_vertices);
    cudaMalloc(&k_frontier, sizeof(bool) * num_vertices);
    cudaMalloc(&k_visited, sizeof(bool) * num_vertices);
    cudaMalloc(&k_queue, sizeof(int) * (num_vertices + 1));     // First one is #elements
    cudaMalloc(&k_prefix_sum, sizeof(int) * num_vertices);
    cudaMalloc(&k_prefix_in_array, sizeof(int) * num_vertices);

    cudaMemcpy(k_v_adj_list, v_adj_list, sizeof(int) * num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(k_v_adj_begin, v_adj_begin, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_v_adj_length, v_adj_length, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_result, result, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_updated, updated, sizeof(bool) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_visited, visited, sizeof(bool) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_frontier, frontier, sizeof(bool) * num_vertices, cudaMemcpyHostToDevice);


    // --- START MEASURE TIME ---


    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    
    queue[0] = 1;
    queue[1] = start_vertex;

    cudaMemcpy(k_queue, queue, sizeof(int) * 2, cudaMemcpyHostToDevice);
    preallocBlockSums(num_vertices);

    while (queue[0] > 0)
    {
        kernel_cuda_frontier_scan_main<<<BLOCKS, THREADS>>>(
            k_v_adj_list, 
            k_v_adj_begin, 
            k_v_adj_length, 
            num_vertices, 
            k_result, 
            k_queue,
            k_updated,
            k_visited);

        kernel_cuda_frontier_scan<<<BLOCKS, THREADS>>>(
            num_vertices,
            k_updated,
            k_frontier,
            k_visited,
            k_prefix_in_array);

        prescanArray(k_prefix_sum, k_prefix_in_array, num_vertices);
        //cudaThreadSynchronize();

        // Generate new queue
        kernel_cuda_generate_queue<<<BLOCKS, THREADS>>>(
            k_prefix_sum, 
            k_frontier, 
            k_queue,
            num_vertices);
        //gpuErrchk(cudaThreadSynchronize());

        kernel_runs++;

        if (kernel_runs > MAX_KERNEL_RUNS)
        {
            return -1;
        }

        cudaMemcpy(queue, k_queue, sizeof(int) * num_vertices, cudaMemcpyDeviceToHost);
    }

    cudaThreadSynchronize();

    gettimeofday(&t2, NULL);
    long long time = get_elapsed_time(&t1, &t2);

    if (report_time)
    {
        printf("%s,%i,%i,%i,%i,%lld\n", __FILE__, num_vertices, num_edges, BLOCKS, THREADS, time); 
    }


    // --- END MEASURE TIME ---


    cudaMemcpy(result, k_result, sizeof(int) * num_vertices, cudaMemcpyDeviceToHost);

    cudaFree(k_v_adj_list);
    cudaFree(k_v_adj_begin);
    cudaFree(k_v_adj_length);
    cudaFree(k_result);
    cudaFree(k_updated);
    cudaFree(k_frontier);
    cudaFree(k_prefix_in_array);
    cudaFree(k_visited);
    cudaFree(k_prefix_sum);
    cudaFree(k_queue);

    // printf("%i kernel runs\n", kernel_runs);

    return time;
}
