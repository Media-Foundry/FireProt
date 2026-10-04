// Independent host identity probe. No model, tensor or GPU kernel execution.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

int main() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count != 1) {
        std::fprintf(stderr, "Expected exactly one visible HIP device\n");
        return 1;
    }
    char pci[64] = {};
    hipDeviceProp_t properties{};
    if (hipDeviceGetPCIBusId(pci, sizeof(pci), 0) != hipSuccess ||
        hipGetDeviceProperties(&properties, 0) != hipSuccess) {
        std::fprintf(stderr, "HIP device identity lookup failed\n");
        return 1;
    }
    // Only the base architecture enters capability policy (e.g. gfx90a).
    char* feature = std::strchr(properties.gcnArchName, ':');
    if (feature) *feature = '\0';
    if (std::strspn(pci, "0123456789abcdefABCDEF:.") != std::strlen(pci) ||
        std::strspn(properties.gcnArchName, "abcdefghijklmnopqrstuvwxyz0123456789") !=
            std::strlen(properties.gcnArchName)) return 1;
    std::printf("{\"pci_address\":\"%s\",\"architecture\":\"%s\"}\n", pci, properties.gcnArchName);
    return 0;
}
