// Accelerating large graph algorithms on the GPU using CUDA
// http://dl.acm.org/citation.cfm?id=1782200

__global__ void kernel_cuda_frontier_numbers_reorder(
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

int bfs_cuda_frontier_numbers_reorder(
    int *v_adj_list_u,
    int *v_adj_begin_u, 
    int *v_adj_length_u, 
    int num_vertices, 
    int num_edges,
    int start_vertex_u, 
    int *result)
{
    int *k_v_adj_list;
    int *k_v_adj_begin;
    int *k_v_adj_length;
    int *k_result;
    bool *k_still_running;

    int kernel_runs = 0;

    fill_n(result, num_vertices, MAX_DIST);

    bool *still_running = new bool[1];

    // Generate reordering
    // pair<degree, index>
    pair<int, int> *sorted = new pair<int, int>[num_vertices];
    for (int i = 0; i < num_vertices; i++)
    {
        sorted[i].first = v_adj_length_u[i];
        sorted[i].second = i;
    }
    sort(sorted, sorted + num_vertices);

    int *mapping_old_to_new = new int[num_vertices];
    int *v_adj_list = new int[num_edges];
    int *v_adj_begin = new int[num_vertices];
    int *v_adj_length = new int[num_vertices];

    for (int i = 0; i < num_vertices; i++)
    {
        v_adj_begin[i] = v_adj_begin_u[sorted[i].second];
        v_adj_length[i] = v_adj_length_u[sorted[i].second];
        mapping_old_to_new[sorted[i].second] = i;
    }

    for (int i = 0; i < num_edges; i++)
    {
        v_adj_list[i] = mapping_old_to_new[v_adj_list_u[i]];
    }

    int start_vertex = mapping_old_to_new[start_vertex_u];

    result[start_vertex] = 0;

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


    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    //int num_threads = BLOCKS * THREADS;
    //int segment_size = (num_vertices + num_threads - 1) / num_threads;; // divide and round up

    do
    {
        *still_running = false;
        cudaMemcpy(k_still_running, still_running, sizeof(bool) * 1, cudaMemcpyHostToDevice);

        kernel_cuda_frontier_numbers_reorder<<<BLOCKS, THREADS>>>(
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

    gettimeofday(&t2, NULL);
    long long time = get_elapsed_time(&t1, &t2);

    if (report_time)
    {
        printf("%s,%i,%i,%i,%i,%lld\n", __FILE__, num_vertices, num_edges, BLOCKS, THREADS, time); 
    }


    // --- END MEASURE TIME ---



    // Result still has the wrong ordering
    int *result2 = new int[num_vertices];
    cudaMemcpy(result2, k_result, sizeof(int) * num_vertices, cudaMemcpyDeviceToHost);

    // Restore original order
    for (int i = 0; i < num_vertices; i++)
    {
        //result[i] = sorted[i].second;
        result[sorted[i].second] = result2[i];
    }

    cudaFree(k_v_adj_list);
    cudaFree(k_v_adj_begin);
    cudaFree(k_v_adj_length);
    cudaFree(k_result);
    cudaFree(k_still_running);

    free(mapping_old_to_new);
    free(v_adj_list);
    free(v_adj_begin);
    free(v_adj_length);
    free(sorted);
    free(result2);

    // printf("%i kernel runs\n", kernel_runs);

    return time;
}
