// SPDX-License-Identifier: GPL-2.0-only WITH OpenFold-3-exception
// Read-only, one-shot AMD SMI adapter for FireProt's existing Rust monitor.
#include <amd_smi/amdsmi.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t max_devices = 1024;
constexpr uint32_t max_processes = 16384;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void checked(amdsmi_status_t status, const char* operation) {
    if (status != AMDSMI_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation)+": AMD SMI status "+std::to_string(status));
}
struct Session {
    bool live = false;
    Session() {
        checked(amdsmi_init(AMDSMI_INIT_AMD_GPUS), "init");
        live = true;
    }
    void finish() {
        auto status = amdsmi_shut_down();
        live = false;
        checked(status, "shutdown");
    }
    ~Session() { if (live) amdsmi_shut_down(); }
};
std::vector<amdsmi_processor_handle> devices() {
    uint32_t count = 0;
    checked(amdsmi_get_socket_handles(&count, nullptr), "socket count");
    require(count > 0 && count <= max_devices, "invalid socket count");
    std::vector<amdsmi_socket_handle> sockets(count);
    const auto capacity = count;
    checked(amdsmi_get_socket_handles(&count, sockets.data()), "socket list");
    require(count > 0 && count <= capacity, "socket count changed beyond capacity");
    sockets.resize(count);
    std::set<amdsmi_socket_handle> seen_sockets;
    std::set<amdsmi_processor_handle> seen_devices;
    std::vector<amdsmi_processor_handle> result;
    // Preserve the official Python binding's socket/processor enumeration order.
    for (auto socket : sockets) {
        require(socket && seen_sockets.insert(socket).second, "null or duplicate socket");
        uint32_t size = 0;
        checked(amdsmi_get_processor_handles(socket, &size, nullptr), "processor count");
        require(size <= max_devices, "invalid processor count");
        if (size == 0) continue;
        std::vector<amdsmi_processor_handle> handles(size);
        const auto available = size;
        checked(amdsmi_get_processor_handles(socket, &size, handles.data()), "processor list");
        require(size <= available, "processor count changed beyond capacity");
        handles.resize(size);
        for (auto handle : handles) {
            require(handle && seen_devices.insert(handle).second, "null or duplicate processor");
            result.push_back(handle);
            require(result.size() <= max_devices, "too many processors");
        }
    }
    require(!result.empty(), "no AMD GPU processors");
    return result;
}
std::string bdf(amdsmi_processor_handle handle) {
    amdsmi_bdf_t value{};
    checked(amdsmi_get_gpu_device_bdf(handle, &value), "device BDF");
    require(value.bdf.domain_number <= 0xffff, "BDF domain exceeds Rust protocol");
    char text[13];
    const auto written = std::snprintf(text, sizeof(text), "%04x:%02x:%02x.%x",
        unsigned(value.bdf.domain_number), unsigned(value.bdf.bus_number),
        unsigned(value.bdf.device_number), unsigned(value.bdf.function_number));
    require(written == 12, "invalid BDF rendering");
    return text;
}
std::set<uint32_t> pids(amdsmi_processor_handle handle) {
    uint32_t capacity = 256;
    for (unsigned attempt = 0; attempt < 4; ++attempt) {
        std::vector<amdsmi_proc_info_t> rows(capacity);
        uint32_t size = capacity;
        auto status = amdsmi_get_gpu_process_list(handle, &size, rows.data());
        if (status == AMDSMI_STATUS_OUT_OF_RESOURCES) {
            require(size > capacity && size <= max_processes, "invalid process resize request");
            capacity = size;
            continue;
        }
        checked(status, "process list");
        require(size <= capacity, "process count exceeds capacity");
        std::set<uint32_t> result;
        for (uint32_t i = 0; i < size; ++i) {
            const uint32_t pid = rows[i].pid;
            // Match the binding: exclude only this query process, never a worker.
            if (pid == uint32_t(getpid())) continue;
            require(pid != 0 && result.insert(pid).second, "zero or duplicate PID");
        }
        return result;
    }
    throw std::runtime_error("process list kept growing");
}
}
int main(int argc, char** argv) {
    if (argc != 3 || std::string(argv[2]) != "--json" ||
        (std::string(argv[1]) != "list" && std::string(argv[1]) != "process" &&
         std::string(argv[1]) != "snapshot-v1")) {
        std::cerr << "Usage: fireprot-smi-probe {list|process|snapshot-v1} --json\n";
        return 64;
    }
    try {
        const bool snapshot = std::string(argv[1]) == "snapshot-v1";
        Session session;
        amdsmi_version_t version{};
        checked(amdsmi_get_lib_version(&version), "library version");
        require(version.major == 27 && version.minor == 0 && version.release == 0,
                "this probe requires qualified AMD SMI 27.0.0");
        auto handles = devices();
        std::set<std::string> addresses;
        std::ostringstream out;
        if (snapshot) out << "{\"schema_version\":1,\"devices\":";
        out << '[';
        for (size_t i = 0; i < handles.size(); ++i) {
            const auto address = bdf(handles[i]);
            require(addresses.insert(address).second, "duplicate device BDF");
            if (i) out << ',';
            out << "{\"gpu\":" << i;
            if (snapshot || std::string(argv[1]) == "list") out << ",\"bdf\":\"" << address << '"';
            if (snapshot || std::string(argv[1]) == "process") {
                const auto values = pids(handles[i]);
                out << ",\"process_list\":[";
                if (values.empty()) out << "{\"process_info\":\"No running processes detected\"}";
                bool first = true;
                for (auto pid : values) {
                    if (!first) out << ',';
                    first = false;
                    out << "{\"process_info\":{\"pid\":" << pid << "}}";
                }
                out << ']';
            }
            out << '}';
        }
        out << ']';
        if (snapshot) out << '}';
        out << '\n';
        session.finish();
        // Emit only after every query and shutdown succeeds: no partial success.
        std::cout << out.str();
        return std::cout.good() ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fireprot-smi-probe: " << error.what() << '\n';
        return 1;
    }
}
