// Hardware-test fixture: hold one small allocation until the test stops us.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <unistd.h>
int main(int argc, char** argv) {
    if (argc != 2) return 1;
    void* allocation = nullptr;
    if (hipMalloc(&allocation, 1024 * 1024) != hipSuccess ||
        hipMemset(allocation, 0, 1024 * 1024) != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess) return 1;
    FILE* ready = std::fopen(argv[1], "w");
    if (!ready) return 1;
    std::fputs("ready", ready);
    std::fclose(ready);
    while (true) pause();
}
