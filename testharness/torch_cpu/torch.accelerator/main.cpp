#include "fuzzer_utils.h"             // General fuzzing utilities
#include <ATen/DeviceAccelerator.h>   // at::accelerator APIs
#include <c10/core/Device.h>          // c10::Device
#include <c10/core/Stream.h>          // c10::Stream helpers
#include <iostream>                   // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to test with
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get index from the next byte if available
        int64_t index = 0;
        if (offset < Size) {
            uint8_t index_byte = Data[offset++];
            index = static_cast<int64_t>(index_byte % 8); // Limit to reasonable range
        }
        
        // Query accelerator availability and exercise basic APIs
        // getAccelerator(false) returns nullopt if no accelerator is available (doesn't throw)
        auto maybe_accelerator = at::accelerator::getAccelerator(false);
        if (maybe_accelerator.has_value()) {
            c10::DeviceType acc_type = maybe_accelerator.value();
            int64_t device_count = at::accelerator::deviceCount();
            if (device_count > 0) {
                c10::DeviceIndex device_index = static_cast<c10::DeviceIndex>(index % device_count);
                at::accelerator::setDeviceIndex(device_index);
                auto current_index = at::accelerator::getDeviceIndex();
                c10::Stream current_stream = at::accelerator::getCurrentStream(current_index);
                at::accelerator::synchronizeDevice(current_index);
                at::accelerator::setCurrentStream(current_stream);

                // Move tensor to the accelerator and back
                c10::Device acc_device(acc_type, current_index);
                auto acc_tensor = tensor.to(acc_device, tensor.scalar_type(), /*non_blocking=*/false, /*copy=*/true);
                auto back_to_cpu = acc_tensor.to(torch::kCPU, acc_tensor.scalar_type(), /*non_blocking=*/false, /*copy=*/true);
                (void)back_to_cpu.sum();
            }
        }

        // Always exercise CPU path to ensure fuzzer makes progress even without accelerator
        auto cpu_tensor = tensor.to(torch::kCPU, tensor.scalar_type(), /*non_blocking=*/false, /*copy=*/false);
        (void)cpu_tensor.sum();

        // Exercise getAccelerator with check=true path as well (may throw if no accelerator)
        // This is wrapped in inner try-catch since it's expected to fail without accelerator
        try {
            auto acc_with_check = at::accelerator::getAccelerator(true);
            (void)acc_with_check;
        } catch (...) {
            // Expected when no accelerator is available
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}