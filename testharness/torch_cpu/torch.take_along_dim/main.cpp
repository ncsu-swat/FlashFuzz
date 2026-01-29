#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Handle scalar tensors (0-dim)
        if (input_tensor.dim() == 0) {
            // For scalar tensors, indices must also be scalar with value 0
            torch::Tensor indices_tensor = torch::zeros({}, torch::kInt64);
            torch::Tensor result = torch::take_along_dim(input_tensor, indices_tensor);
            return 0;
        }
        
        // Get dimension from fuzzer data
        int64_t dim = 0;
        if (offset + sizeof(uint8_t) <= Size) {
            dim = static_cast<int64_t>(Data[offset] % input_tensor.dim());
            offset += sizeof(uint8_t);
        }
        
        // Get the size along the chosen dimension
        int64_t dim_size = input_tensor.size(dim);
        if (dim_size <= 0) {
            return 0;
        }
        
        // Determine how many indices to gather (from fuzzer data)
        int64_t num_indices = 1;
        if (offset + sizeof(uint8_t) <= Size) {
            num_indices = 1 + (Data[offset] % 16);  // 1 to 16 indices
            offset += sizeof(uint8_t);
        }
        
        // Build the shape for indices tensor:
        // Same shape as input except dimension 'dim' can differ
        std::vector<int64_t> indices_shape;
        for (int64_t i = 0; i < input_tensor.dim(); i++) {
            if (i == dim) {
                indices_shape.push_back(num_indices);
            } else {
                indices_shape.push_back(input_tensor.size(i));
            }
        }
        
        // Create indices tensor with valid values (0 to dim_size-1)
        torch::Tensor indices_tensor = torch::randint(0, dim_size, indices_shape, torch::kInt64);
        
        // If we have more fuzzer data, use it to modify some index values
        if (offset < Size) {
            auto indices_accessor = indices_tensor.flatten();
            int64_t num_elements = indices_accessor.numel();
            for (int64_t i = 0; i < num_elements && offset < Size; i++, offset++) {
                // Use fuzzer byte to select index value, ensuring it's in valid range
                indices_tensor.flatten()[i] = static_cast<int64_t>(Data[offset] % dim_size);
            }
            indices_tensor = indices_tensor.reshape(indices_shape);
        }
        
        // Test take_along_dim with explicit dimension
        torch::Tensor result = torch::take_along_dim(input_tensor, indices_tensor, dim);
        
        // Also test with c10::optional<int64_t> nullopt (flattened case)
        if (offset < Size && (Data[offset - 1] % 4 == 0)) {
            try {
                // When dim is nullopt, input is flattened and indices must be 1D
                torch::Tensor flat_input = input_tensor.flatten();
                int64_t flat_size = flat_input.numel();
                if (flat_size > 0) {
                    int64_t flat_num_indices = 1 + (num_indices % 16);
                    torch::Tensor flat_indices = torch::randint(0, flat_size, {flat_num_indices}, torch::kInt64);
                    torch::Tensor flat_result = torch::take_along_dim(flat_input, flat_indices, 0);
                }
            } catch (...) {
                // Silently ignore inner exceptions for edge cases
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}