#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for sigmoid
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply sigmoid operation
        torch::Tensor output = torch::sigmoid(input);
        
        // Try some variants of the operation
        if (offset + 1 < Size) {
            // Use in-place version if we have more data
            torch::Tensor input_copy = input.clone();
            input_copy.sigmoid_();
            
            // Try functional version with out parameter
            torch::Tensor output2 = torch::empty_like(input);
            torch::sigmoid_out(output2, input);
        }
        
        // Try with edge cases if we have more data
        if (offset + 1 < Size) {
            uint8_t edge_case_selector = Data[offset++];
            
            // Create some edge case tensors based on the selector
            if (edge_case_selector % 5 == 0) {
                // Very large values
                torch::Tensor large_vals = torch::full({2, 2}, 1e38, torch::kFloat);
                torch::Tensor large_result = torch::sigmoid(large_vals);
            }
            else if (edge_case_selector % 5 == 1) {
                // Very small values
                torch::Tensor small_vals = torch::full({2, 2}, -1e38, torch::kFloat);
                torch::Tensor small_result = torch::sigmoid(small_vals);
            }
            else if (edge_case_selector % 5 == 2) {
                // NaN values
                torch::Tensor nan_vals = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN(), torch::kFloat);
                torch::Tensor nan_result = torch::sigmoid(nan_vals);
            }
            else if (edge_case_selector % 5 == 3) {
                // Infinity values
                torch::Tensor inf_vals = torch::full({2, 2}, std::numeric_limits<float>::infinity(), torch::kFloat);
                torch::Tensor inf_result = torch::sigmoid(inf_vals);
            }
            else {
                // Empty tensor
                torch::Tensor empty_tensor = torch::empty({0}, torch::kFloat);
                torch::Tensor empty_result = torch::sigmoid(empty_tensor);
            }
        }
        
        // Test with different dtypes if we have more data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            
            try {
                if (dtype_selector == 0) {
                    // Double precision
                    torch::Tensor double_input = input.to(torch::kDouble);
                    torch::Tensor double_output = torch::sigmoid(double_input);
                }
                else if (dtype_selector == 1) {
                    // Half precision (if supported)
                    torch::Tensor half_input = input.to(torch::kHalf);
                    torch::Tensor half_output = torch::sigmoid(half_input);
                }
                else {
                    // BFloat16 (if supported)
                    torch::Tensor bf16_input = input.to(torch::kBFloat16);
                    torch::Tensor bf16_output = torch::sigmoid(bf16_input);
                }
            }
            catch (const std::exception &) {
                // Some dtypes may not be supported on all platforms, silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}