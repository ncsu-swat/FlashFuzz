#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>
#include <ATen/Context.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the mode
        if (Size < 1) {
            return 0;
        }
        
        // Extract a byte to determine whether to enable or disable flush denormal
        bool mode = Data[offset++] & 0x1;
        
        // Set flush denormal mode using ATen context
        // at::set_flush_denormal returns true if the operation succeeded (CPU supports it)
        bool success = at::set_flush_denormal(mode);
        (void)success;  // Suppress unused variable warning
        
        // Create a tensor with potentially denormal values to test the effect
        if (offset < Size) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure we're working with float tensors for denormal testing
                if (!tensor.is_floating_point()) {
                    tensor = tensor.to(torch::kFloat32);
                }
                
                // Perform operations that might generate denormal values
                // Denormal float32 values are roughly in range 1e-45 to 1e-38
                torch::Tensor result = tensor * 1e-38f;
                
                // Perform another operation
                torch::Tensor squared = result * result;  // This may produce denormals
                
                // Sum to force computation
                torch::Tensor sum = squared.sum();
                
                // Access the value to ensure the computation is performed
                float value = sum.item<float>();
                (void)value;  // Suppress unused variable warning
                
                // Also test with the opposite mode
                bool opposite_mode = !mode;
                at::set_flush_denormal(opposite_mode);
                
                // Perform same operations under different mode
                torch::Tensor result2 = tensor * 1e-38f;
                torch::Tensor squared2 = result2 * result2;
                torch::Tensor sum2 = squared2.sum();
                float value2 = sum2.item<float>();
                (void)value2;
                
            } catch (const c10::Error&) {
                // Tensor creation or operation may fail for invalid fuzz input - silently ignore
            }
        }
        
        // Reset flush denormal mode to default (false)
        at::set_flush_denormal(false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}