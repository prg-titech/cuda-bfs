// Accelerating large graph algorithms on the GPU using CUDA
// http://dl.acm.org/citation.cfm?id=1782200

#define BLOCK_QUEUE_SIZE 8192

__global__ void kernel_cuda_frontier_queue(
    int *v_adj_list,
    int *v_adj_begin,
    int *v_adj_length,
    int num_vertices,
    int *result,
    int iteration,
    int *input_queue,
    int *output_queue)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    //assert(false);

    // TODO: Read to shared memory? Maybe also the input queue?
    __shared__ int input_queue_size;

    if (threadIdx.x == 0)
    {
        input_queue_size = *input_queue;
    }

    __syncthreads();

    __shared__ int queue_size;
    __shared__ int next_queue[BLOCK_QUEUE_SIZE];

    if (threadIdx.x == 0)
    {
        queue_size = 0;
    }

    __syncthreads();

    for (int v = 0; v < input_queue_size; v += num_threads)
    {
        if (v + tid < input_queue_size)
        {
            int vertex = input_queue[v + tid + 1];

            for (int n = 0; n < v_adj_length[vertex]; n++)
            {
                int neighbor = v_adj_list[v_adj_begin[vertex] + n];

                if (result[neighbor] == MAX_DIST)
                {
                    result[neighbor] = iteration + 1;

                    // Add to queue (atomicAdd returns original value)
                    int position = atomicAdd(&queue_size, 1);
                    next_queue[position] = neighbor;
                }
            }
        }

        __syncthreads();

        __shared__ int global_offset;

        if (threadIdx.x == 0)
        {
            // First value is size of queue
            global_offset = atomicAdd(output_queue, queue_size);
        }

        __syncthreads();

        // Copy queue to global memory
        for (int i = 0; i < queue_size; i += blockDim.x)
        {
            if (i + threadIdx.x < queue_size)
            {
                output_queue[global_offset + i + threadIdx.x + 1] = next_queue[i + threadIdx.x];
            }
        }

        __syncthreads();

        queue_size = 0;

        __syncthreads();
    }
}

int bfs_cuda_frontier_queue(
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
    int *k_queue_1;
    int *k_queue_2;

    int kernel_runs = 0;
    int zero_value = 0;

    fill_n(result, num_vertices, MAX_DIST);
    result[start_vertex] = 0;

    int *input_queue_size = new int;

    cudaMalloc(&k_v_adj_list, sizeof(int) * num_edges);
    cudaMalloc(&k_v_adj_begin, sizeof(int) * num_vertices);
    cudaMalloc(&k_v_adj_length, sizeof(int) * num_vertices);
    cudaMalloc(&k_result, sizeof(int) * num_vertices);
    cudaMalloc(&k_queue_1, sizeof(int) * num_vertices * 8); // Not sure how big?
    cudaMalloc(&k_queue_2, sizeof(int) * num_vertices * 8); // Not sure how big?

    cudaMemcpy(k_v_adj_list, v_adj_list, sizeof(int) * num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(k_v_adj_begin, v_adj_begin, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_v_adj_length, v_adj_length, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_result, result, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);


    // --- START MEASURE TIME ---


    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    int *k_input_queue = k_queue_1;
    int *k_output_queue = k_queue_2;

    int first_queue[] = { 1, start_vertex };
    *input_queue_size = 1;
    cudaMemcpy(k_input_queue, first_queue, sizeof(int) * 2, cudaMemcpyHostToDevice);

    do
    {
        cudaMemcpy(k_output_queue, &zero_value, sizeof(int) * 1, cudaMemcpyHostToDevice);

        int blocks = min(BLOCKS, max(1, *input_queue_size / THREADS));
        int threads = *input_queue_size <= THREADS ? *input_queue_size : THREADS;

        kernel_cuda_frontier_queue<<<blocks, threads>>>(
            k_v_adj_list, 
            k_v_adj_begin, 
            k_v_adj_length, 
            num_vertices, 
            k_result, 
            kernel_runs,
            k_input_queue,
            k_output_queue);

        kernel_runs++;

        if (kernel_runs > MAX_KERNEL_RUNS)
        {
            return -1;
        }

        // Swap queues
        int *tmp = k_input_queue;
        k_input_queue = k_output_queue;
        k_output_queue = tmp;

        cudaMemcpy(input_queue_size, k_input_queue, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    } while (*input_queue_size > 0);

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
    cudaFree(k_queue_1);
    cudaFree(k_queue_2);

    // printf("%i kernel runs\n", kernel_runs);

    return time;
}
