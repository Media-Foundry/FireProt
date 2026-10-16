// SPDX-License-Identifier: GPL-2.0-only WITH OpenFold-3-exception
// Keep one known HIP6 allocation alive for read-only process-detection tests.
#include <hip/hip_runtime.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
int main() {
    const char* visible = std::getenv("HIP_VISIBLE_DEVICES");
    if (!visible || std::strcmp(visible,"6")) return 1;
    char pci[32]{};
    void* memory = nullptr;
    if (hipSetDevice(0)!=hipSuccess || hipDeviceGetPCIBusId(pci,sizeof(pci),0)!=hipSuccess
        || std::strcmp(pci,"0000:8e:00.0") || hipMalloc(&memory,16*1024*1024)!=hipSuccess
        || hipMemset(memory,0,16*1024*1024)!=hipSuccess || hipDeviceSynchronize()!=hipSuccess) return 2;
    std::printf("{\"pid\":%ld,\"pci_address\":\"%s\"}\n",long(getpid()),pci);
    std::fflush(stdout);
    char stop[8];
    if (!std::fgets(stop,sizeof(stop),stdin)) return 3;
    return hipFree(memory)==hipSuccess ? 0 : 4;
}
