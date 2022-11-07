
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <fstream>

cudaError_t runCuda(uint64_t* c, const int* pool, const int* bounds, unsigned int poolSize, unsigned int boundsSize, int min, int max);

__global__ void check7Rolls(uint64_t *c, const int *pool, const int *bounds)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    //Bit vector encoding the desired chest loot in a cursed way kap, WTF
    int target = 21 | 2 << 6 | 1 << 12 | 1 << 18 | 1 << 24;
    uint64_t upperBits = (uint64_t)(6 * (id + pool[94]) + 4) << 17;
    int index = 0;

    for(uint64_t lowerBits = 0; lowerBits < 1 << 17; lowerBits++) {
        uint64_t seed = lowerBits | upperBits;
        int total = 0;
       
        for(int i = 0; i < 7; i++) {
            seed = (seed * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
            int item = pool[(int)(seed >> 17) % 94];

            if (item >= 5) {
                total = 0;
                break;
            }

            int min = bounds[item * 2];
            int max = bounds[item * 2 + 1];
            int count = 1;

            if(max != 0) {
                seed = (seed * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
                count = (int)(seed >> 17) % (max - min + 1) + min;
            }

            total += count << (item * 6);
        }
        
        if(total == target) {
            c[id] = ((0xDFE05BCB1365ULL * (upperBits | lowerBits) + 0x615C0E462AA9ULL) ^ 0x5DEECE66DULL) & ((1ULL << 48) - 1);
        }
    }
}

__global__ void check8Rolls(uint64_t* c, const int* pool, const int* bounds)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int target = 21 | 2 << 6 | 1 << 12 | 1 << 18 | 1 << 24;
    uint64_t upperBits = (uint64_t)(6 * (id + pool[94]) + 5) << 17;
    int index = 0;

    for (uint64_t lowerBits = 0; lowerBits < 1 << 17; lowerBits++) {
        uint64_t seed = lowerBits | upperBits;
        int total = 0;

        for (int i = 0; i < 8; i++) {
            seed = (seed * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
            int item = pool[(int)(seed >> 17) % 94];

            if (item >= 5) {
                total = 0;
                break;
            }

            int min = bounds[item * 2];
            int max = bounds[item * 2 + 1];
            int count = 1;

            if (max != 0) {
                seed = (seed * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
                count = (int)(seed >> 17) % (max - min + 1) + min;
            }

            total += count << (item * 6);
        }

        if (total == target) {
            c[id] = ((0xDFE05BCB1365ULL * (upperBits | lowerBits) + 0x615C0E462AA9ULL) ^ 0x5DEECE66DULL) & ((1ULL << 48) - 1);
        }
    }
}

int main()
{
    int min = 0, max = 64512;
    int* pool = new int[95];
    int* bounds = new int[68];
    uint64_t c[64512] = { 0ULL };

    int id = 0;
    for (int i = 0; i < 3; i++) pool[id++] = 5; //"diamond";
    for (int i = 0; i < 10; i++) pool[id++] = 6; //"iron_ingot";
    for (int i = 0; i < 5; i++) pool[id++] = 3; //"gold_ingot";
    for (int i = 0; i < 15; i++) pool[id++] = 7; //"bread";
    for (int i = 0; i < 15; i++) pool[id++] = 1; //"apple";
    for (int i = 0; i < 5; i++) pool[id++] = 8; //"iron_pickaxe";
    for (int i = 0; i < 5; i++) pool[id++] = 9; //"iron_sword";
    for (int i = 0; i < 5; i++) pool[id++] = 10; //"iron_chestplate";
    for (int i = 0; i < 5; i++) pool[id++] = 2; //"iron_helmet";
    for (int i = 0; i < 5; i++) pool[id++] = 11; //"iron_leggings";
    for (int i = 0; i < 5; i++) pool[id++] = 12; //"iron_boots";
    for (int i = 0; i < 5; i++) pool[id++] = 0;//"obsidian";
    for (int i = 0; i < 5; i++) pool[id++] = 13; //"oak_sapling";
    for (int i = 0; i < 3; i++) pool[id++] = 14; //"saddle";
    for (int i = 0; i < 1; i++) pool[id++] = 15; //"iron_horse_armor";
    for (int i = 0; i < 1; i++) pool[id++] = 4; //"golden_horse_armor";
    for (int i = 0; i < 1; i++) pool[id++] = 16; //"diamond_horse_armor";

    for (int i = 0; i < 68; i++) bounds[i] = 0;
    bounds[5 * 2] = 1; bounds[5 * 2 + 1] = 3;
    bounds[1 * 2] = 1; bounds[1 * 2 + 1] = 3;
    bounds[3 * 2] = 1; bounds[3 * 2 + 1] = 3;
    bounds[7 * 2] = 1; bounds[7 * 2 + 1] = 3;

    bounds[6 * 2] = 1; bounds[6 * 2 + 1] = 5;

    bounds[0 * 2] = 3; bounds[0 * 2 + 1] = 7;
    bounds[13 * 2] = 3; bounds[13 * 2 + 1] = 7;
    cudaError_t cudaStatus;

    std::ofstream myfile;
    myfile.open("seeds.txt");
    

    for(int offset = 0; offset < 357913941 + 64512; offset += 64512) {
        pool[94] = offset;
        cudaStatus = runCuda(c, pool, bounds, 95, 68, min, max);

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "runCuda failed!");
            return 1;
        }

        myfile << "BLOCK #" << (offset / 64512) << "================================\n";

        for (int i = 0; i < 64512; i++) {
            if (c[i] != 0) {
                myfile << c[i] << "\n";
            }
        }

        myfile.flush();
    }

    myfile.close();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t runCuda(uint64_t*c, const int *pool, const int *bounds, unsigned int poolSize, unsigned int boundsSize, int min, int max)
{
    int *dev_a = 0;
    int *dev_b = 0;
    uint64_t* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, (max - min + 1) * sizeof(uint64_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, poolSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, boundsSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, pool, poolSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, bounds, boundsSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.1024,128
    check8Rolls<<<512, 126>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, (max - min + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
