// Accelerating large graph algorithms on the GPU using CUDA
// http://dl.acm.org/citation.cfm?id=1782200

#define DEFER_MAX 16
#define D_BLOCK_QUEUE_SIZE 2048

__global__ void kernel_cuda_frontier_numbers_defer(
    int *v_adj_list,
    int *v_adj_begin,
    int *v_adj_length,
    int num_vertices,
    int *result,
    bool *still_running,
    int iteration)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    __shared__ int queue_size;
    __shared__ int next_queue[D_BLOCK_QUEUE_SIZE];

    if (threadIdx.x == 0)
    {
        queue_size = 0;
    }

    __syncthreads();

    for (int v = 0; v < num_vertices; v += num_threads)
    {
        int vertex = v + tid;

        if (vertex < num_vertices && result[vertex] == iteration)
        {
            if (v_adj_length[vertex] < DEFER_MAX || queue_size >= D_BLOCK_QUEUE_SIZE - blockDim.x)
            {
                for (int n = 0; n < v_adj_length[vertex]; n++)
                {
                    int neighbor = v_adj_list[v_adj_begin[vertex] + n];

                    if (result[neighbor] == MAX_DIST)
                    {
                        result[neighbor] = iteration + 1;
                        *still_running = true;
                    }
                }
            }
            else
            {
                // Add to queue (atomicAdd returns original value)
                int position = atomicAdd(&queue_size, 1);
                next_queue[position] = vertex;
            }
        }

        __syncthreads();
    }

    // Process outliers
    for (int v = 0; v < queue_size; v += blockDim.x)
    {
        if (v + threadIdx.x < queue_size)
        {
            int vertex = next_queue[v + threadIdx.x];

            for (int n = 0; n < v_adj_length[vertex]; n++)
            {
                int neighbor = v_adj_list[v_adj_begin[vertex] + n];

                if (result[neighbor] == MAX_DIST)
                {
                    result[neighbor] = iteration + 1;
                    *still_running = true;
                }
            }
        }
    }
}

int bfs_cuda_frontier_numbers_defer(
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
    bool *k_still_running;

    int kernel_runs = 0;

    fill_n(result, num_vertices, MAX_DIST);
    result[start_vertex] = 0;

    bool *still_running = new bool[1];

    cudaMalloc(&k_v_adj_list, sizeof(int) * num_edges);
    cudaMalloc(&k_v_adj_begin, sizeof(int) * num_vertices);
    cudaMalloc(&k_v_adj_length, sizeof(int) * num_vertices);
    cudaMalloc(&k_result, sizeof(int) * num_vertices);
    cudaMalloc(&k_still_running, sizeof(bool) * 1);

    cudaMemcpy(k_v_adj_list, v_adj_list, sizeof(int) * num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(k_v_adj_begin, v_adj_begin, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_v_adj_length, v_adj_length, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_result, result, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);


    // --- START MEASURE TIME ---


    auto start_time = chrono::high_resolution_clock::now();

    do
    {
        *still_running = false;
        cudaMemcpy(k_still_running, still_running, sizeof(bool) * 1, cudaMemcpyHostToDevice);

        kernel_cuda_frontier_numbers_defer<<<BLOCKS, THREADS>>>(
            k_v_adj_list, 
            k_v_adj_begin, 
            k_v_adj_length, 
            num_vertices, 
            k_result, 
            k_still_running,
            kernel_runs);

        kernel_runs++;

        cudaMemcpy(still_running, k_still_running, sizeof(bool) * 1, cudaMemcpyDeviceToHost);
    } while (*still_running);

    cudaThreadSynchronize();

    auto end_time = chrono::high_resolution_clock::now();
    long long time = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

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
    cudaFree(k_still_running);

    // printf("%i kernel runs\n", kernel_runs);

    return time;
}
