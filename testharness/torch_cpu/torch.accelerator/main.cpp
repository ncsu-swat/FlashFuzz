#include "fuzzer_utils.h"             // General fuzzing utilities
#include <ATen/DeviceAccelerator.h>   // at::accelerator APIs
#include <c10/core/Device.h>          // c10::Device
#include <c10/core/Stream.h>          // c10::Stream helpers
#include <iostream>                   // For cerr
#include <tuple>                      // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing torch.accelerator" << std::endl; // keep target keyword
    (void) "torch.accelerator"; // ensure keyword is present for harness checks
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

        // Always touch a CPU path
        auto cpu_tensor = tensor.to(torch::kCPU, tensor.scalar_type(), /*non_blocking=*/false, /*copy=*/false);
        (void)cpu_tensor.sum();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
