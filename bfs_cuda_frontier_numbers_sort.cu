// Accelerating large graph algorithms on the GPU using CUDA
// http://dl.acm.org/citation.cfm?id=1782200

__global__ void kernel_cuda_frontier_numbers_sort(
    int *v_adj_list,
    int *v_adj_begin,
    int *v_adj_length,
    int num_vertices,
    int *result,
    bool *still_running,
    int iteration,
    int *reordering)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    //int range_from = segment_size * tid;
    //int range_to = min((segment_size * (tid + 1)), num_vertices);

    for (int v = 0; v < num_vertices; v += num_threads)
    {
        int vertex = v + tid;

        if (vertex < num_vertices && result[vertex] == iteration)
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
    }
}

int bfs_cuda_frontier_numbers_sort(
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
    int *k_reordering;

    int kernel_runs = 0;

    fill_n(result, num_vertices, MAX_DIST);
    result[start_vertex] = 0;

    bool *still_running = new bool[1];

    // Generate reordering
    // pair<degree, index>
    pair<int, int> *sorted = new pair<int, int>[num_vertices];
    for (int i = 0; i < num_vertices; i++)
    {
        sorted[i].first = v_adj_length[i];
        sorted[i].second = i;
    }
    sort(sorted, sorted + num_vertices);

    int *reordering = new int[num_vertices];
    for (int i = 0; i < num_vertices; i++)
    {
        reordering[i] = sorted[i].second;
    }

    gpuErrchk(cudaMalloc(&k_v_adj_list, sizeof(int) * num_edges));
    cudaMalloc(&k_v_adj_begin, sizeof(int) * num_vertices);
    cudaMalloc(&k_v_adj_length, sizeof(int) * num_vertices);
    cudaMalloc(&k_result, sizeof(int) * num_vertices);
    cudaMalloc(&k_still_running, sizeof(bool) * 1);
    cudaMalloc(&k_reordering, sizeof(int) * num_vertices);

    cudaMemcpy(k_v_adj_list, v_adj_list, sizeof(int) * num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(k_v_adj_begin, v_adj_begin, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_v_adj_length, v_adj_length, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_result, result, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_reordering, reordering, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);


    // --- START MEASURE TIME ---


    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    do
    {
        *still_running = false;
        cudaMemcpy(k_still_running, still_running, sizeof(bool) * 1, cudaMemcpyHostToDevice);

        kernel_cuda_frontier_numbers_sort<<<BLOCKS, THREADS>>>(
            k_v_adj_list, 
            k_v_adj_begin, 
            k_v_adj_length, 
            num_vertices, 
            k_result, 
            k_still_running,
            kernel_runs,
            k_reordering);

        kernel_runs++;

        cudaMemcpy(still_running, k_still_running, sizeof(bool) * 1, cudaMemcpyDeviceToHost);
    } while (*still_running);

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
    cudaFree(k_still_running);
    cudaFree(k_reordering);

    // printf("%i kernel runs\n", kernel_runs);

    return time;
}
