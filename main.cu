
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <fstream>

#define POOL_SIZE       94
#define SEEDS_PER_BLOCK 64512

cudaError_t runCuda(uint64_t* c, const int16_t* pool, const uint8_t* bounds, unsigned int poolSize, unsigned int boundsSize, int min, int max);

__global__ void checknRolls(int n, uint64_t* c, const int16_t* pool, const uint8_t* bounds)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // # of: Obsidian, apple, iron_helmet, gold_ingot, golden_horse_armor (as defined by the values of pool)
    uint8_t target[] = { 21, 2, 1, 1, 1 };
    const size_t targetSize = sizeof(target) / sizeof(target[0]);

    if (targetSize > 8)
        return; // This should never happen

    uint64_t upperBits = (uint64_t)(6 * (id + (SEEDS_PER_BLOCK * pool[POOL_SIZE])) + n - 3) << 17;

    for (uint64_t lowerBits = 0; lowerBits < 1 << 17; lowerBits++) {
        uint64_t seed = lowerBits | upperBits;

        uint8_t total[targetSize] = {0};

        for (int i = 0; i < n; i++) {
            seed = (seed * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
            int id = (seed >> 17) % POOL_SIZE;
            int item = pool[id];

            if (item == -1) {
                total[0] = 0xFF; // 0xFF is just used to indicate that the seed is invalid
                break;
            }

            if (item >= targetSize) {
                return;
            }

            int min = bounds[id] & 0xF;
            int max = bounds[id] >> 4;
            int count = 1;

            if (max != 0) {
                seed = (seed * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
                count = (int)(seed >> 17) % (max - min + 1) + min;
            }

            total[item] += count;
        }

        bool equal = true;
        for (int i = 0; i < targetSize; i++) {
            if (total[i] != target[i]) {
                equal = false;
                break;
            }
        }
        if (equal) {
            c[id] = ((0xDFE05BCB1365ULL * (upperBits | lowerBits) + 0x615C0E462AA9ULL) ^ 0x5DEECE66DULL) & ((1ULL << 48) - 1);
        }
    }
}

int main()
{
    int min = 0, max = SEEDS_PER_BLOCK;
    int16_t* pool = new int16_t[POOL_SIZE + 1];
    uint8_t* bounds = new uint8_t[POOL_SIZE];
    uint64_t c[SEEDS_PER_BLOCK] = { 0ULL };

    for (int i = 0; i < POOL_SIZE; i++) bounds[i] = 0;

    int id = 0;
    // Mark items in order (corresponding with target[] in checknRolls)
    // All unused items set to -1
    for (int i = 0; i < 3; i++) {pool[id] =   -1; /* diamond */     bounds[id++] = 1 | (3 << 4);}
    for (int i = 0; i < 10; i++){pool[id] =   -1; /* iron_ingot */  bounds[id++] = 1 | (5 << 4);}
    for (int i = 0; i < 5; i++) {pool[id] =    3; /* gold_ingot */  bounds[id++] = 1 | (3 << 4);}
    for (int i = 0; i < 15; i++){pool[id] =   -1; /* bread */       bounds[id++] = 1 | (3 << 4);}
    for (int i = 0; i < 15; i++){pool[id] =    1; /* apple */       bounds[id++] = 1 | (3 << 4);}
    for (int i = 0; i < 5; i++)  pool[id++] = -1; /* iron_pickaxe */
    for (int i = 0; i < 5; i++)  pool[id++] = -1; /* iron_sword */
    for (int i = 0; i < 5; i++)  pool[id++] = -1; /* iron_chestplate */
    for (int i = 0; i < 5; i++)  pool[id++] =  2; /* iron_helmet */
    for (int i = 0; i < 5; i++)  pool[id++] = -1; /* iron_leggings */
    for (int i = 0; i < 5; i++)  pool[id++] = -1; /* iron_boots */
    for (int i = 0; i < 5; i++) {pool[id] =    0; /* obsidian */    bounds[id++] = 3 | (7 << 4);}
    for (int i = 0; i < 5; i++) {pool[id] =   -1; /* oak_sapling */ bounds[id++] = 3 | (7 << 4);}
    for (int i = 0; i < 3; i++)  pool[id++] = -1; /* saddle */
    for (int i = 0; i < 1; i++)  pool[id++] = -1; /* iron_horse_armor */
    for (int i = 0; i < 1; i++)  pool[id++] =  4; /* golden_horse_armor */
    for (int i = 0; i < 1; i++)  pool[id++] = -1; /* diamond_horse_armor */

    cudaError_t cudaStatus;

    std::ofstream myfile;
    myfile.open("seeds.txt");

    for(int offset = 0; offset < 1 + (357913941 / SEEDS_PER_BLOCK); offset += 1) {
        pool[POOL_SIZE] = offset;
        cudaStatus = runCuda(c, pool, bounds, POOL_SIZE + 1, POOL_SIZE, min, max);

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "runCuda failed!\n");
            return 1;
        }

        myfile << "BLOCK #" << (offset) << "================================\n";

        for (int i = 0; i < SEEDS_PER_BLOCK; i++) {
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
cudaError_t runCuda(uint64_t*c, const int16_t *pool, const uint8_t *bounds, unsigned int poolSize, unsigned int boundsSize, int min, int max)
{
    int16_t *dev_a = 0;
    uint8_t *dev_b = 0;
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

    cudaStatus = cudaMalloc((void**)&dev_a, poolSize * sizeof(int16_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, boundsSize * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, pool, poolSize * sizeof(int16_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, bounds, boundsSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.1024,128
    checknRolls<<<512, 126>>>(8, dev_c, dev_a, dev_b);

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
