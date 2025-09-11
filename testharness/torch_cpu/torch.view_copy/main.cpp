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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract some bytes for new shape parameters if available
        std::vector<int64_t> new_shape;
        int64_t total_elements = input_tensor.numel();
        
        // Determine number of dimensions for the new shape
        uint8_t new_rank = 0;
        if (offset < Size) {
            new_rank = fuzzer_utils::parseRank(Data[offset++]);
        } else {
            // Default to 1D if no more data
            new_rank = 1;
        }
        
        // Parse the new shape
        if (new_rank > 0) {
            // Parse dimensions for the new shape
            if (offset < Size) {
                new_shape = fuzzer_utils::parseShape(Data, offset, Size, new_rank);
            } else {
                // Default shape if no more data
                new_shape.push_back(total_elements);
            }
            
            // Validate that the new shape has the same number of elements
            int64_t new_total_elements = 1;
            for (const auto& dim : new_shape) {
                new_total_elements *= dim;
            }
            
            // If shapes are incompatible, try view_copy anyway to test error handling
            // Don't add defensive checks - let PyTorch handle invalid inputs
        }
        
        // Apply view_copy operation
        torch::Tensor result;
        if (new_shape.empty()) {
            // Test scalar view
            result = torch::view_copy(input_tensor, {});
        } else {
            result = torch::view_copy(input_tensor, new_shape);
        }
        
        // Test that view_copy creates a new tensor (not a view)
        if (result.data_ptr() == input_tensor.data_ptr() && result.numel() > 0 && input_tensor.numel() > 0) {
            throw std::runtime_error("view_copy should create a new tensor, not a view");
        }
        
        // Test some edge cases with additional view_copy operations
        if (offset + 1 < Size) {
            uint8_t edge_case = Data[offset++];
            
            // Try different edge cases based on the byte
            switch (edge_case % 5) {
                case 0:
                    // Try flattening the tensor
                    result = torch::view_copy(input_tensor, {-1});
                    break;
                case 1:
                    // Try reshaping with a -1 dimension (auto-inferred)
                    if (input_tensor.dim() > 1 && input_tensor.numel() > 0) {
                        std::vector<int64_t> inferred_shape;
                        inferred_shape.push_back(input_tensor.size(0));
                        inferred_shape.push_back(-1);
                        result = torch::view_copy(input_tensor, inferred_shape);
                    }
                    break;
                case 2:
                    // Try adding a dimension of size 1
                    {
                        std::vector<int64_t> expanded_shape = input_tensor.sizes().vec();
                        expanded_shape.push_back(1);
                        result = torch::view_copy(input_tensor, expanded_shape);
                    }
                    break;
                case 3:
                    // Try removing a dimension of size 1
                    if (input_tensor.dim() > 1) {
                        for (int64_t i = 0; i < input_tensor.dim(); i++) {
                            if (input_tensor.size(i) == 1) {
                                std::vector<int64_t> squeezed_shape;
                                for (int64_t j = 0; j < input_tensor.dim(); j++) {
                                    if (j != i) {
                                        squeezed_shape.push_back(input_tensor.size(j));
                                    }
                                }
                                result = torch::view_copy(input_tensor, squeezed_shape);
                                break;
                            }
                        }
                    }
                    break;
                case 4:
                    // Try a potentially invalid shape (same number of elements but different layout)
                    if (input_tensor.numel() > 1) {
                        std::vector<int64_t> random_shape;
                        int64_t remaining = input_tensor.numel();
                        
                        // Use the next bytes as factors if available
                        while (remaining > 1 && offset < Size && random_shape.size() < 4) {
                            uint8_t factor_byte = Data[offset++];
                            int64_t factor = (factor_byte % remaining) + 1;
                            if (factor > 1 && remaining % factor == 0) {
                                random_shape.push_back(factor);
                                remaining /= factor;
                            }
                        }
                        
                        if (remaining > 0) {
                            random_shape.push_back(remaining);
                        }
                        
                        result = torch::view_copy(input_tensor, random_shape);
                    }
                    break;
            }
        }
        
        // Verify that the result has the expected number of elements
        if (result.numel() != input_tensor.numel()) {
            throw std::runtime_error("view_copy result has different number of elements");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
