// SPDX-License-Identifier: GPL-2.0-only WITH OpenFold-3-exception
// CPU-only fault injection. Never linked into the real probe.
#include <amd_smi/amdsmi.h>
#include <unistd.h>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace {
bool test(const char* name) {
    const char* value = std::getenv("FIREPROT_TEST_SMI_CASE");
    return value && std::string(value) == name;
}
amdsmi_status_t error() { return static_cast<amdsmi_status_t>(1); }
}
extern "C" amdsmi_status_t amdsmi_init(uint64_t flags) {
    return flags != AMDSMI_INIT_AMD_GPUS || test("init_error") ? error() : AMDSMI_STATUS_SUCCESS;
}
extern "C" amdsmi_status_t amdsmi_shut_down() {
    return test("shutdown_error") ? error() : AMDSMI_STATUS_SUCCESS;
}
extern "C" amdsmi_status_t amdsmi_get_lib_version(amdsmi_version_t* value) {
    if (test("version_error")) return error();
    value->major = test("unsupported_version") ? 28 : 27;
    value->minor = value->release = 0;
    return AMDSMI_STATUS_SUCCESS;
}
extern "C" amdsmi_status_t amdsmi_get_socket_handles(uint32_t* n, amdsmi_socket_handle* values) {
    if (test("socket_error")) return error();
    if (values) values[0] = reinterpret_cast<amdsmi_socket_handle>(1);
    *n = test("zero_sockets") ? 0 : test("too_many_sockets") ? 2048 :
        test("socket_growth") && values ? 2 : 1;
    return AMDSMI_STATUS_SUCCESS;
}
extern "C" amdsmi_status_t amdsmi_get_processor_handles(amdsmi_socket_handle, uint32_t* n,
                                                         amdsmi_processor_handle* values) {
    if (test("processor_error")) return error();
    if (values) {
        values[0] = reinterpret_cast<amdsmi_processor_handle>(test("null_processor") ? 0 : 16);
        values[1] = reinterpret_cast<amdsmi_processor_handle>(test("duplicate_processor") ? 16 : 17);
    }
    *n = test("zero_processors") ? 0 : test("too_many_processors") ? 2048 :
        test("processor_growth") && values ? 3 : 2;
    return AMDSMI_STATUS_SUCCESS;
}
extern "C" amdsmi_status_t amdsmi_get_gpu_device_bdf(amdsmi_processor_handle device, amdsmi_bdf_t* value) {
    if (test("bdf_error")) return error();
    value->as_uint = 0;
    value->bdf.domain_number = test("large_domain") ? 65536 : 0;
    value->bdf.bus_number = uintptr_t(device) == 16 || test("duplicate_bdf") ? 0x8e : 0xae;
    return AMDSMI_STATUS_SUCCESS;
}
extern "C" amdsmi_status_t amdsmi_get_gpu_process_list(amdsmi_processor_handle device, uint32_t* n,
                                                       amdsmi_proc_info_t* values) {
    if (test("process_error") || (test("second_device_error") && uintptr_t(device)==17)) return error();
    if (test("empty")) { *n = 0; return AMDSMI_STATUS_SUCCESS; }
    if (test("bad_resize")) return AMDSMI_STATUS_OUT_OF_RESOURCES;
    if (test("oversize_resize")) { *n=20000; return AMDSMI_STATUS_OUT_OF_RESOURCES; }
    if (test("continuous_growth")) { *n+=1; return AMDSMI_STATUS_OUT_OF_RESOURCES; }
    if (test("success_overflow")) { *n+=1; return AMDSMI_STATUS_SUCCESS; }
    if (test("resize")) {
        if (*n<300) { *n=300; return AMDSMI_STATUS_OUT_OF_RESOURCES; }
        *n=300;
        for (uint32_t i=0; i<*n; ++i) values[i].pid=100000000U+i;
        return AMDSMI_STATUS_SUCCESS;
    }
    *n=2;
    values[0].pid=test("zero_pid") ? 0 : 4242;
    values[1].pid=test("duplicate_pid") ? 4242 : uint32_t(getpid());
    return AMDSMI_STATUS_SUCCESS;
}
