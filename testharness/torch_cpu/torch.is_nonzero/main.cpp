#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t
#include <limits>         // For numeric_limits

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
        
        // Use first byte to determine test case
        uint8_t test_case = Data[0] % 8;
        offset = 1;
        
        switch (test_case) {
            case 0: {
                // Test with a scalar tensor from fuzzer data
                if (Size > offset) {
                    torch::Tensor scalar_tensor = torch::tensor(static_cast<int>(Data[offset]));
                    bool result = torch::is_nonzero(scalar_tensor);
                    volatile bool dummy = result;
                    (void)dummy;
                }
                break;
            }
            case 1: {
                // Test with float scalar tensor
                if (Size > offset) {
                    float val = static_cast<float>(Data[offset]) / 255.0f;
                    torch::Tensor float_tensor = torch::tensor(val);
                    bool result = torch::is_nonzero(float_tensor);
                    volatile bool dummy = result;
                    (void)dummy;
                }
                break;
            }
            case 2: {
                // Test with boolean tensor
                if (Size > offset) {
                    bool bool_value = Data[offset] % 2 != 0;
                    torch::Tensor bool_tensor = torch::tensor(bool_value);
                    bool result = torch::is_nonzero(bool_tensor);
                    volatile bool dummy = result;
                    (void)dummy;
                }
                break;
            }
            case 3: {
                // Test with zero value
                torch::Tensor zero_tensor = torch::tensor(0);
                bool result = torch::is_nonzero(zero_tensor);
                volatile bool dummy = result;
                (void)dummy;
                break;
            }
            case 4: {
                // Test with double scalar
                if (Size > offset) {
                    double val = static_cast<double>(Data[offset]) - 128.0;
                    torch::Tensor double_tensor = torch::tensor(val);
                    bool result = torch::is_nonzero(double_tensor);
                    volatile bool dummy = result;
                    (void)dummy;
                }
                break;
            }
            case 5: {
                // Test with NaN (should be nonzero since NaN != 0)
                torch::Tensor nan_tensor = torch::tensor(std::numeric_limits<float>::quiet_NaN());
                try {
                    bool result = torch::is_nonzero(nan_tensor);
                    volatile bool dummy = result;
                    (void)dummy;
                } catch (const std::exception &) {
                    // Handle potential exception for NaN
                }
                break;
            }
            case 6: {
                // Test with Inf
                torch::Tensor inf_tensor = torch::tensor(std::numeric_limits<float>::infinity());
                try {
                    bool result = torch::is_nonzero(inf_tensor);
                    volatile bool dummy = result;
                    (void)dummy;
                } catch (const std::exception &) {
                    // Handle potential exception for Inf
                }
                break;
            }
            case 7: {
                // Test with multi-element tensor (should throw - is_nonzero requires single element)
                torch::Tensor multi_tensor = torch::ones({2});
                try {
                    bool result = torch::is_nonzero(multi_tensor);
                    volatile bool dummy = result;
                    (void)dummy;
                } catch (const std::exception &) {
                    // Expected exception for multi-element tensor
                }
                break;
            }
        }
        
        // Additional test: create tensor from fuzzer data and squeeze to scalar if possible
        if (Size > 2) {
            size_t tensor_offset = 1;
            torch::Tensor tensor = fuzzer_utils::createTensor(Data + 1, Size - 1, tensor_offset);
            
            // Only test if tensor has exactly one element
            if (tensor.numel() == 1) {
                bool result = torch::is_nonzero(tensor);
                volatile bool dummy = result;
                (void)dummy;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}