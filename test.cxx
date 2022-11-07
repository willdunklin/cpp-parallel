#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <execution>
#include <chrono>
using namespace std::chrono;

#define SIZE 10000000

int main(int argc, char** argv) {
    srand(0);

    int* v = new int[SIZE];
    for (int i = 0; i < SIZE; ++i) {
        v[i] = rand();
    }

    auto start = high_resolution_clock::now();
    // std::sort(v, v + SIZE);
    std::sort(std::execution::par, v, v + SIZE);
    auto end = high_resolution_clock::now();

    // std::cout << v[0] << std::endl;
    // for (int i = 0; i < 100; ++i) {
    //     std::cout << v[SIZE - i - 1] << " ";
    // }
    // std::cout << std::endl;
    std::cout << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << std::endl;
    return 0;
}
