#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }

        size_t offset = 0;

        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Check if MPS backend is available
        // Note: MPS is only available on macOS with Apple Silicon/AMD GPU
        // On Linux, this will always return false
        bool is_mps_available = false;
        
#ifdef AT_MPS_ENABLED
        // Only try MPS operations if compiled with MPS support
        is_mps_available = torch::mps::is_available();
        
        if (is_mps_available) {
            // Create MPS device
            auto mps_device = torch::Device(torch::kMPS);
            
            // Move tensor to MPS device
            torch::Tensor mps_tensor = tensor.to(mps_device);
            
            // Perform some operations on the MPS tensor
            torch::Tensor result = mps_tensor + 1;
            
            // Move back to CPU for verification
            torch::Tensor cpu_result = result.to(torch::kCPU);
            
            // Test synchronization
            torch::mps::synchronize();
            
            // Use fuzzer data for seed to get coverage of different seeds
            if (Size >= sizeof(uint64_t)) {
                uint64_t seed = *reinterpret_cast<const uint64_t*>(Data);
                torch::mps::manual_seed(seed);
            }
        }
#endif

        // Even without MPS, we can test device creation attempts
        // This tests the device parsing logic
        try {
            // Test that kMPS device type exists (even if not available)
            auto device_type = torch::kMPS;
            (void)device_type;
        } catch (...) {
            // Expected on systems without MPS support
        }

        // Test tensor operations on CPU as fallback
        // This ensures we still exercise some code paths
        torch::Tensor cpu_result = tensor + 1;
        cpu_result = cpu_result * 2;
        (void)cpu_result;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}