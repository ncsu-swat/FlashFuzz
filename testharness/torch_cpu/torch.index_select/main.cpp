#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least a few bytes for basic operations
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // index_select requires at least 1D tensor
        if (input_tensor.dim() == 0) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        
        // Get a dimension to select along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure dim is valid
        dim = dim % input_tensor.dim();
        if (dim < 0) {
            dim = -dim;
        }
        
        int64_t dim_size = input_tensor.size(dim);
        if (dim_size == 0) {
            // Can't select from empty dimension
            return 0;
        }
        
        // Create index tensor
        torch::Tensor index_tensor;
        
        if (offset < Size) {
            uint8_t index_type = Data[offset++];
            
            if (index_type % 4 == 0) {
                // Create a tensor with a single index
                int64_t idx = 0;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&idx, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                }
                // Ensure index is valid
                idx = ((idx % dim_size) + dim_size) % dim_size;
                index_tensor = torch::tensor({idx}, torch::kInt64);
            }
            else if (index_type % 4 == 1) {
                // Create a tensor with multiple indices
                std::vector<int64_t> indices;
                int64_t num_indices = 1 + (index_type % 8); // 1-8 indices
                
                for (int64_t i = 0; i < num_indices; i++) {
                    int64_t idx = 0;
                    if (offset + sizeof(int64_t) <= Size) {
                        std::memcpy(&idx, Data + offset, sizeof(int64_t));
                        offset += sizeof(int64_t);
                    }
                    // Ensure index is valid
                    idx = ((idx % dim_size) + dim_size) % dim_size;
                    indices.push_back(idx);
                }
                
                index_tensor = torch::tensor(indices, torch::kInt64);
            }
            else if (index_type % 4 == 2) {
                // Create index tensor using arange for contiguous selection
                int64_t start = 0;
                int64_t end = dim_size;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&start, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    start = ((start % dim_size) + dim_size) % dim_size;
                }
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&end, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    end = ((end % dim_size) + dim_size) % dim_size;
                }
                if (start > end) {
                    std::swap(start, end);
                }
                if (start == end) {
                    end = start + 1;
                }
                index_tensor = torch::arange(start, end, torch::kInt64);
            }
            else {
                // Create index tensor with repeated indices (tests gather behavior)
                int64_t idx = 0;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&idx, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                }
                idx = ((idx % dim_size) + dim_size) % dim_size;
                int64_t repeats = 1 + (index_type % 4);
                std::vector<int64_t> indices(repeats, idx);
                index_tensor = torch::tensor(indices, torch::kInt64);
            }
        }
        else {
            // Default: select first element
            index_tensor = torch::tensor({int64_t(0)}, torch::kInt64);
        }
        
        // Apply index_select operation
        torch::Tensor result = torch::index_select(input_tensor, dim, index_tensor);
        
        // Verify result properties
        if (result.numel() > 0) {
            // Perform operations on the result to ensure it's computed
            auto sum = result.sum();
            
            // Test that result has correct shape
            // Result should have same dims, but selected dim has size = index.numel()
            if (result.dim() != input_tensor.dim()) {
                std::cerr << "Unexpected: result dim mismatch" << std::endl;
            }
        }
        
        // Test with different dimensions if tensor has multiple dims
        if (input_tensor.dim() > 1 && offset < Size) {
            int64_t new_dim = Data[offset++] % input_tensor.dim();
            int64_t new_dim_size = input_tensor.size(new_dim);
            if (new_dim_size > 0) {
                torch::Tensor new_index = torch::tensor({int64_t(0)}, torch::kInt64);
                torch::Tensor result2 = torch::index_select(input_tensor, new_dim, new_index);
                (void)result2.sum();
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}