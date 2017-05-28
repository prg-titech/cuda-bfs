__global__ void kernel_cuda_per_edge_frontier_numbers(
    int *v_adj_from,
    int *v_adj_to,
    int num_edges,
    int *result,
    bool *still_running,
    int iteration)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int e = 0; e < num_edges; e += num_threads)
    {
        int edge = e + tid;

        if (edge < num_edges)
        {
            int from_vertex = v_adj_from[edge];
            int to_vertex = v_adj_to[edge];

            if (result[from_vertex] == iteration && result[to_vertex] == MAX_DIST)
            {
                result[to_vertex] = iteration + 1;
                *still_running = true;
            }
        }
    }
}

void bfs_cuda_per_edge_frontier_numbers(
    int *v_adj_list,
    int *v_adj_begin, 
    int *v_adj_length, 
    int num_vertices, 
    int num_edges,
    int start_vertex, 
    int *result)
{
    // Convert data
    // TODO: Check if it is better to allocate only one array
    int *v_adj_from = new int[num_edges];
    int *v_adj_to = new int[num_edges];

    int next_index = 0;
    for (int i = 0; i < num_vertices; i++)
    {
        for (int j = v_adj_begin[i]; j < v_adj_length[i] + v_adj_begin[i]; j++)
        {
            v_adj_from[next_index] = i;
            v_adj_to[next_index++] = v_adj_list[j];
        }
    }

    int *k_v_adj_from;
    int *k_v_adj_to;
    int *k_result;
    bool *k_still_running;

    int kernel_runs = 0;

    fill_n(result, num_vertices, MAX_DIST);
    result[start_vertex] = 0;

    bool *still_running = new bool[1];

    cudaMalloc(&k_v_adj_from, sizeof(int) * num_edges);
    cudaMalloc(&k_v_adj_to, sizeof(int) * num_edges);
    cudaMalloc(&k_result, sizeof(int) * num_vertices);
    cudaMalloc(&k_still_running, sizeof(bool) * 1);

    cudaMemcpy(k_v_adj_from, v_adj_from, sizeof(int) * num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(k_v_adj_to, v_adj_to, sizeof(int) * num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(k_result, result, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);


    // --- START MEASURE TIME ---


    auto start_time = chrono::high_resolution_clock::now();

    do
    {
        *still_running = false;
        cudaMemcpy(k_still_running, still_running, sizeof(bool) * 1, cudaMemcpyHostToDevice);

        kernel_cuda_per_edge_frontier_numbers<<<BLOCKS, THREADS>>>(
            k_v_adj_from, 
            k_v_adj_to, 
            num_edges, 
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

    cudaFree(k_v_adj_from);
    cudaFree(k_v_adj_to);
    cudaFree(k_result);
    cudaFree(k_still_running);

    free(v_adj_from);
    free(v_adj_to);

    // printf("%i kernel runs\n", kernel_runs);
}
