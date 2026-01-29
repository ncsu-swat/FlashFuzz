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
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Need a non-empty tensor with at least one dimension
        if (input_tensor.dim() == 0 || input_tensor.numel() == 0) {
            return 0;
        }
        
        // Get a dimension to fill along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Use modulo to ensure dim is within valid range
            dim = dim % input_tensor.dim();
            if (dim < 0) {
                dim += input_tensor.dim();
            }
        }
        
        // Get the size along the selected dimension
        int64_t dim_size = input_tensor.size(dim);
        if (dim_size == 0) {
            return 0;
        }
        
        // Create index tensor (indices to fill)
        torch::Tensor index_tensor;
        if (offset + sizeof(int64_t) <= Size) {
            // Determine how many indices to create (1-4 indices)
            int num_indices = 1 + (Data[offset] % 4);
            offset++;
            
            std::vector<int64_t> indices;
            for (int i = 0; i < num_indices && offset + sizeof(int64_t) <= Size; i++) {
                int64_t idx;
                std::memcpy(&idx, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                // Ensure index is within valid range [0, dim_size)
                idx = idx % dim_size;
                if (idx < 0) {
                    idx += dim_size;
                }
                indices.push_back(idx);
            }
            
            if (indices.empty()) {
                indices.push_back(0);
            }
            
            index_tensor = torch::tensor(indices, torch::kLong);
        } else {
            // Create a simple index tensor if we don't have enough data
            index_tensor = torch::tensor({0}, torch::kLong);
        }
        
        // Get a value to fill with
        torch::Scalar value;
        if (offset + sizeof(float) <= Size) {
            float val;
            std::memcpy(&val, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Avoid NaN/Inf for more predictable behavior
            if (std::isnan(val) || std::isinf(val)) {
                val = 1.0f;
            }
            value = val;
        } else {
            value = 1.0f;
        }
        
        // Try different variants of index_fill
        uint8_t variant = (offset < Size) ? Data[offset++] % 4 : 0;
        
        try {
            torch::Tensor result;
            
            switch (variant) {
                case 0: {
                    // Out-of-place version with Scalar value
                    result = input_tensor.index_fill(dim, index_tensor, value);
                    break;
                }
                case 1: {
                    // In-place version with Scalar value
                    result = input_tensor.clone();
                    result.index_fill_(dim, index_tensor, value);
                    break;
                }
                case 2: {
                    // Out-of-place version with Tensor value (0-dim tensor)
                    torch::Tensor value_tensor = torch::tensor(value.toFloat());
                    result = input_tensor.index_fill(dim, index_tensor, value_tensor);
                    break;
                }
                case 3: {
                    // In-place version with Tensor value (0-dim tensor)
                    torch::Tensor value_tensor = torch::tensor(value.toFloat());
                    result = input_tensor.clone();
                    result.index_fill_(dim, index_tensor, value_tensor);
                    break;
                }
            }
            
            // Verify result is valid
            if (result.defined() && result.numel() > 0) {
                // Access sum to ensure computation happens
                auto sum = result.sum();
                (void)sum;
            }
        } catch (const c10::Error &e) {
            // Expected errors from invalid tensor operations - silently ignore
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}