#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For std::memcpy

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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get max value from remaining data
        float max_value = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&max_value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Apply clamp_max_ operation (in-place)
        input_tensor.clamp_max_(max_value);
        
        // Force evaluation to ensure the operation is actually executed
        (void)input_tensor.sum().item<float>();
        
        // Test with different max values if we have more data
        if (offset + sizeof(float) <= Size) {
            float second_max;
            std::memcpy(&second_max, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Apply another clamp_max_ with different value
            input_tensor.clamp_max_(second_max);
            (void)input_tensor.sum().item<float>();
        }
        
        // Test edge cases with fresh tensors
        if (Size >= 8) {
            // Test with NaN as max value
            torch::Tensor nan_test = fuzzer_utils::createTensor(Data, Size, offset = 0);
            try {
                nan_test.clamp_max_(std::numeric_limits<float>::quiet_NaN());
                (void)nan_test.sum().item<float>();
            } catch (...) {
                // NaN handling may throw - that's acceptable
            }
            
            // Test with infinity as max value
            torch::Tensor inf_test = fuzzer_utils::createTensor(Data, Size, offset = 0);
            inf_test.clamp_max_(std::numeric_limits<float>::infinity());
            (void)inf_test.sum().item<float>();
            
            // Test with negative infinity as max value
            torch::Tensor neg_inf_test = fuzzer_utils::createTensor(Data, Size, offset = 0);
            neg_inf_test.clamp_max_(-std::numeric_limits<float>::infinity());
            (void)neg_inf_test.sum().item<float>();
        }
        
        // Test with Tensor as max argument (if supported)
        if (Size >= 8) {
            offset = 0;
            torch::Tensor tensor_a = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor tensor_b = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                // clamp_max_ also accepts a tensor as the max value
                tensor_a.clamp_max_(tensor_b);
                (void)tensor_a.sum().item<float>();
            } catch (...) {
                // Shape mismatch or other tensor-tensor issues - silent catch
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}