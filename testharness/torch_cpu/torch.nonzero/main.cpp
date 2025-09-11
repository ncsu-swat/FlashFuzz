#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <vector>         // For std::vector

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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply nonzero operation
        torch::Tensor result = torch::nonzero(input_tensor);
        
        // Try different variants of nonzero
        if (offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            // Test as_tuple variant
            if (variant % 2 == 0) {
                auto tuple_result = torch::nonzero_numpy(input_tensor);
                
                // Access elements from the vector to ensure it's properly processed
                if (!tuple_result.empty() && !tuple_result[0].sizes().empty()) {
                    auto first_dim = tuple_result[0];
                    if (first_dim.numel() > 0) {
                        auto _ = first_dim.item<int64_t>();
                    }
                }
            }
        }
        
        // Test with different layout if we have more data
        if (offset + 1 < Size) {
            uint8_t layout_selector = Data[offset++];
            
            // Create a non-contiguous tensor by permuting dimensions if tensor has at least 2 dimensions
            if (input_tensor.dim() >= 2 && layout_selector % 3 == 0) {
                auto permuted = input_tensor.permute({input_tensor.dim() - 1, 0});
                torch::Tensor permuted_result = torch::nonzero(permuted);
            }
            
            // Test with transposed tensor if possible
            if (input_tensor.dim() >= 2 && layout_selector % 3 == 1) {
                auto transposed = input_tensor.transpose(0, input_tensor.dim() - 1);
                torch::Tensor transposed_result = torch::nonzero(transposed);
            }
        }
        
        // Test with different options if we have more data
        if (offset + 1 < Size) {
            uint8_t option_selector = Data[offset++];
            
            // Test with out parameter
            if (option_selector % 2 == 0) {
                // Create an output tensor with appropriate shape
                // The shape will be adjusted by nonzero
                torch::Tensor out_tensor = torch::empty({0, input_tensor.dim()}, torch::kLong);
                torch::nonzero_out(out_tensor, input_tensor);
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
