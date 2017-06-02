// Accelerating large graph algorithms on the GPU using CUDA
// http://dl.acm.org/citation.cfm?id=1782200

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
    int ceil_num_vertices,
    bool *updated,
    bool *frontier,
    bool *visited,
    int *prefix_sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int v = 0; v < ceil_num_vertices; v += num_threads)
    {
        int vertex = v + tid;

        if (vertex < num_vertices)
        {
            if (updated[vertex])
            {
                visited[vertex] = true;
            }

            frontier[vertex] = updated[vertex];
            prefix_sum[vertex] = updated[vertex];

            updated[vertex] = false;
        }
        else if (vertex < ceil_num_vertices)
        {
            frontier[vertex] = false;
            prefix_sum[vertex] = 0;
        }
    }
}

__global__ void kernel_cuda_up_sweep( 
    int *prefix_sum,
    int d,
    int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
     
    if (tid < d)
    {
        int ai = offset * (2 * tid + 1) - 1;
        int bi = offset * (2 * tid + 2) - 1;

        prefix_sum[bi] += prefix_sum[ai];
    }
}

__global__ void kernel_cuda_down_sweep( 
    int *prefix_sum,
    int d,
    int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < d)
    {
        int ai = offset * (2 * tid + 1) - 1;
        int bi = offset * (2 * tid + 2) - 1;

        int temp = prefix_sum[ai];
        prefix_sum[ai] = prefix_sum[bi];
        prefix_sum[bi] += temp;
    }
}

__global__ void kernel_cuda_combined_sweeps(int *g_odata, int *g_idata, int n)
{
    __shared__ int temp[1024];  // allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;

        

    temp[2*thid] = g_idata[2*thid]; // load input into shared memory
    temp[2*thid+1] = g_idata[2*thid+1];


    for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
    { 
        __syncthreads();
        
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
   
        offset *= 2;
    }
   
    if (thid == 0) { temp[n - 1] = 0; } // clear the last element
               
    
    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();

        if (thid < d)                     
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
    
       
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t; 
        }
    }

    __syncthreads();

    g_odata[2*thid] = temp[2*thid]; // write results to device memory
    g_odata[2*thid+1] = temp[2*thid+1];
}

__global__ void kernel_cuda_generate_queue(
    int *prefix_sum,
    bool *frontier,
    int *queue,
    int num_vertices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;


    if (tid < num_vertices && frontier[tid])
    {
        queue[prefix_sum[tid] + 1] = tid;
    }

    // Set size of queue
    if (tid == num_vertices - 1)
    {
        queue[0] = prefix_sum[tid] + (int) frontier[tid];
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

    int kernel_runs = 0;

    int ceil_num_vertices = (int) pow(2, ceil(log(num_vertices)/log(2)));

    int *prefix_sum = new int[ceil_num_vertices];

    bool *updated = new bool[num_vertices];
    fill_n(updated, num_vertices, false);

    bool *visited = new bool[num_vertices];
    fill_n(visited, num_vertices, false);
    visited[start_vertex] = true;

    bool *frontier = new bool[ceil_num_vertices];
    fill_n(frontier, ceil_num_vertices, false);
    frontier[start_vertex] = true;

    fill_n(result, num_vertices, MAX_DIST);
    result[start_vertex] = 0;

    int *queue = new int[num_vertices];
    int zero_value = 0;

    cudaMalloc(&k_v_adj_list, sizeof(int) * num_edges);
    cudaMalloc(&k_v_adj_begin, sizeof(int) * num_vertices);
    cudaMalloc(&k_v_adj_length, sizeof(int) * num_vertices);
    cudaMalloc(&k_result, sizeof(int) * num_vertices);
    cudaMalloc(&k_updated, sizeof(bool) * num_vertices);
    cudaMalloc(&k_frontier, sizeof(bool) * ceil_num_vertices);
    cudaMalloc(&k_visited, sizeof(bool) * num_vertices);
    cudaMalloc(&k_queue, sizeof(int) * (num_vertices + 1));     // First one is #elements
    cudaMalloc(&k_prefix_sum, sizeof(int) * ceil_num_vertices);

    cudaMemcpy(k_v_adj_list, v_adj_list, sizeof(int) * num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(k_v_adj_begin, v_adj_begin, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_v_adj_length, v_adj_length, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_result, result, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_updated, updated, sizeof(bool) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_visited, visited, sizeof(bool) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_frontier, frontier, sizeof(bool) * ceil_num_vertices, cudaMemcpyHostToDevice);


    // --- START MEASURE TIME ---


    auto start_time = chrono::high_resolution_clock::now();

    int reduce_steps = (int) ceil(log(ceil_num_vertices)/log(2));
    
    queue[0] = 1;
    queue[1] = start_vertex;

    cudaMemcpy(k_queue, queue, sizeof(int) * 2, cudaMemcpyHostToDevice);

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

        gpuErrchk(cudaThreadSynchronize());

        kernel_cuda_frontier_scan<<<BLOCKS, THREADS>>>(
            num_vertices,
            ceil_num_vertices,
            k_updated,
            k_frontier,
            k_visited,
            k_prefix_sum);

        gpuErrchk(cudaThreadSynchronize());


        if (ceil_num_vertices > 1024)
        {
            // Prefix sum algorithm: https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
            // Up Sweep
            int offset = 1;

            for (int d = ceil_num_vertices >> 1; d > 0; d >>= 1)
            {
                kernel_cuda_up_sweep<<<2 * BLOCKS, THREADS>>>(k_prefix_sum, d, offset);
                offset *= 2;
            }

            // Clear last
            cudaMemcpy(k_prefix_sum + ceil_num_vertices - 1, &zero_value, sizeof(int) * 1, cudaMemcpyHostToDevice);

            // Down Sweep
            for (int d = 1; d < ceil_num_vertices; d *= 2)
            {
                offset >>= 1;
                kernel_cuda_down_sweep<<<2 * BLOCKS, THREADS>>>(k_prefix_sum, d, offset);
            }
        }
        else
        {
            kernel_cuda_combined_sweeps<<<1, ceil_num_vertices>>>(
                k_prefix_sum, k_prefix_sum, ceil_num_vertices);
        }

        /*cudaMemcpy(prefix_sum k_prefix_sum, sizeof(int) * ceil_num_vertices, cudaMemcpyDeviceToHost);

        for (int  i = 0; i < ceil_num_vertices; i++)
        {
            printf("%i\n", prefix_sum[i]);
        }*/


        // Generate new queue
        kernel_cuda_generate_queue<<<BLOCKS, THREADS>>>(
            k_prefix_sum, 
            k_frontier, 
            k_queue,
            num_vertices);
        gpuErrchk(cudaThreadSynchronize());

        //exit(1);

        kernel_runs++;

        cudaMemcpy(queue, k_queue, sizeof(int) * num_vertices, cudaMemcpyDeviceToHost);
    }

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
    cudaFree(k_updated);
    cudaFree(k_frontier);
    cudaFree(k_visited);
    cudaFree(k_prefix_sum);
    cudaFree(k_queue);

    // printf("%i kernel runs\n", kernel_runs);

    return time;
}
