## Getting default C++ project
1. Make sure you have a C++ compiler that can target C++17 or higher.

2. Install cmake: https://cmake.org/download/

3. Install ninja. Easiest way is probably from pip: `python -m pip install ninja`
    - Otherwise, see https://ninja-build.org/ for instructions.

4. Configure cmake build: `cmake -G Ninja -S . -B build`

5. Build project: `cmake --build build`

6. Run project: `./build/Parallel`

## GPU Acceleration
7. For GPU compilation read this article: https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/
    - Make sure you install `nvc++`
    - Make sure you compile manully (without the CMake commands) with the `-stdpar` flag as per the article's example
