#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <vector>         // For std::vector

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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply nonzero operation
        torch::Tensor result = torch::nonzero(input_tensor);
        
        // Verify result shape: should be (N, input_tensor.dim()) where N is number of nonzero elements
        if (result.dim() == 2 && result.size(1) == input_tensor.dim()) {
            // Access some elements to ensure proper processing
            if (result.numel() > 0) {
                auto first_idx = result[0];
            }
        }
        
        // Try different variants of nonzero
        if (offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            // Test nonzero_numpy variant (returns vector of tensors, one per dimension)
            if (variant % 2 == 0) {
                try {
                    auto tuple_result = torch::nonzero_numpy(input_tensor);
                    
                    // Access elements from the vector to ensure it's properly processed
                    if (!tuple_result.empty() && tuple_result[0].numel() > 0) {
                        auto first_dim = tuple_result[0];
                        auto _ = first_dim[0].item<int64_t>();
                    }
                } catch (...) {
                    // Silently ignore expected failures
                }
            }
        }
        
        // Test with different layout if we have more data
        if (offset + 1 < Size) {
            uint8_t layout_selector = Data[offset++];
            
            // Create a non-contiguous tensor by transposing if tensor has at least 2 dimensions
            if (input_tensor.dim() >= 2) {
                try {
                    if (layout_selector % 3 == 0) {
                        // Transpose first and last dimensions
                        auto transposed = input_tensor.transpose(0, input_tensor.dim() - 1);
                        torch::Tensor transposed_result = torch::nonzero(transposed);
                    }
                    else if (layout_selector % 3 == 1) {
                        // Create non-contiguous view via slice
                        if (input_tensor.size(0) > 1) {
                            auto sliced = input_tensor.slice(0, 0, input_tensor.size(0), 2);
                            torch::Tensor sliced_result = torch::nonzero(sliced);
                        }
                    }
                } catch (...) {
                    // Silently ignore expected failures
                }
            }
        }
        
        // Test with out parameter
        if (offset + 1 < Size) {
            uint8_t option_selector = Data[offset++];
            
            if (option_selector % 2 == 0) {
                try {
                    // Create an output tensor - nonzero_out will resize it appropriately
                    torch::Tensor out_tensor = torch::empty({0, input_tensor.dim()}, torch::kLong);
                    torch::nonzero_out(out_tensor, input_tensor);
                } catch (...) {
                    // Silently ignore expected failures
                }
            }
        }
        
        // Test with different dtypes
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            
            try {
                torch::Tensor converted;
                switch (dtype_selector % 4) {
                    case 0:
                        converted = input_tensor.to(torch::kFloat);
                        break;
                    case 1:
                        converted = input_tensor.to(torch::kInt);
                        break;
                    case 2:
                        converted = input_tensor.to(torch::kBool);
                        break;
                    case 3:
                        converted = input_tensor.to(torch::kDouble);
                        break;
                }
                torch::Tensor dtype_result = torch::nonzero(converted);
            } catch (...) {
                // Silently ignore expected failures
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