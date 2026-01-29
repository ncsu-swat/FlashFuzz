#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::sort, std::unique

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
        
        // Need at least a few bytes to create a tensor and parse parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip if tensor is empty or has no dimensions
        if (input_tensor.numel() == 0 || input_tensor.dim() == 0) {
            return 0;
        }
        
        // Parse parameters for tensor_split
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Parse sections parameter (number of chunks or indices)
        uint8_t sections_type = Data[offset++];
        
        // Decide whether to use integer sections or indices vector
        if (sections_type % 2 == 0) {
            // Use integer sections
            if (offset < Size) {
                int64_t sections = static_cast<int64_t>(Data[offset++]);
                // Ensure sections is at least 1
                sections = sections > 0 ? sections : 1;
                
                // Parse dimension parameter
                int64_t dim = 0;
                if (offset < Size && input_tensor.dim() > 0) {
                    dim = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
                }
                
                // Call tensor_split with sections
                try {
                    std::vector<torch::Tensor> result = torch::tensor_split(input_tensor, sections, dim);
                } catch (const std::exception& e) {
                    // Expected exceptions for invalid parameters
                }
            }
        } else {
            // Use indices vector
            if (offset < Size) {
                // Parse number of indices
                uint8_t num_indices = Data[offset++] % 10; // Limit to reasonable number
                
                // Parse indices
                std::vector<int64_t> indices;
                for (uint8_t i = 0; i < num_indices && offset < Size; ++i) {
                    int64_t index = static_cast<int64_t>(Data[offset++]);
                    indices.push_back(index);
                }
                
                // Sort indices to ensure they're in ascending order
                if (!indices.empty()) {
                    std::sort(indices.begin(), indices.end());
                    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
                }
                
                // Parse dimension parameter
                int64_t dim = 0;
                if (offset < Size && input_tensor.dim() > 0) {
                    dim = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
                }
                
                // Call tensor_split with indices
                try {
                    if (!indices.empty()) {
                        std::vector<torch::Tensor> result = torch::tensor_split(input_tensor, indices, dim);
                    }
                } catch (const std::exception& e) {
                    // Expected exceptions for invalid parameters
                }
            }
        }
        
        // Try tensor_split with negative dimension
        if (offset < Size && input_tensor.dim() > 0) {
            try {
                int64_t neg_dim = -1 * (static_cast<int64_t>(Data[offset++] % input_tensor.dim()) + 1);
                std::vector<torch::Tensor> result = torch::tensor_split(input_tensor, 2, neg_dim);
            } catch (const std::exception& e) {
                // Expected exceptions for invalid dimensions
            }
        }
        
        // Try tensor_split with sections larger than tensor dimension
        if (input_tensor.dim() > 0 && input_tensor.size(0) > 0) {
            try {
                int64_t large_sections = input_tensor.size(0) + 5;
                std::vector<torch::Tensor> result = torch::tensor_split(input_tensor, large_sections, 0);
            } catch (const std::exception& e) {
                // Expected exceptions for large sections
            }
        }
        
        // Test with all dimensions
        for (int64_t d = 0; d < input_tensor.dim(); ++d) {
            try {
                int64_t size_at_dim = input_tensor.size(d);
                if (size_at_dim > 1) {
                    std::vector<torch::Tensor> result = torch::tensor_split(input_tensor, 2, d);
                }
            } catch (const std::exception& e) {
                // Expected exceptions
            }
        }
        
        // Test with indices at boundaries
        if (input_tensor.dim() > 0 && input_tensor.size(0) > 2) {
            try {
                std::vector<int64_t> boundary_indices = {1, input_tensor.size(0) - 1};
                std::vector<torch::Tensor> result = torch::tensor_split(input_tensor, boundary_indices, 0);
            } catch (const std::exception& e) {
                // Expected exceptions
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