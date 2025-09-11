#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
            
            // Try functional version
            torch::Tensor output2 = torch::sigmoid(input);
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
            
            if (edge_case_selector % 5 == 1) {
                // Very small values
                torch::Tensor small_vals = torch::full({2, 2}, -1e38, torch::kFloat);
                torch::Tensor small_result = torch::sigmoid(small_vals);
            }
            
            if (edge_case_selector % 5 == 2) {
                // NaN values
                torch::Tensor nan_vals = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN(), torch::kFloat);
                torch::Tensor nan_result = torch::sigmoid(nan_vals);
            }
            
            if (edge_case_selector % 5 == 3) {
                // Infinity values
                torch::Tensor inf_vals = torch::full({2, 2}, std::numeric_limits<float>::infinity(), torch::kFloat);
                torch::Tensor inf_result = torch::sigmoid(inf_vals);
            }
            
            if (edge_case_selector % 5 == 4) {
                // Empty tensor
                std::vector<int64_t> empty_shape = {0};
                torch::Tensor empty_tensor = torch::empty(empty_shape);
                torch::Tensor empty_result = torch::sigmoid(empty_tensor);
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
