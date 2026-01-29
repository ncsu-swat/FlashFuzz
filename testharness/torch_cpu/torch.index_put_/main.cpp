#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Create the input tensor with at least 1 dimension
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has at least one dimension for indexing
        if (tensor.dim() == 0) {
            tensor = tensor.unsqueeze(0);
        }
        
        // Determine number of indices to create (between 1 and min(3, tensor.dim()))
        uint8_t max_indices = std::min(static_cast<int64_t>(3), tensor.dim());
        if (max_indices == 0) {
            return 0;
        }
        
        uint8_t num_indices = 1;
        if (offset < Size) {
            num_indices = (Data[offset++] % max_indices) + 1;
        }
        
        // Create indices for indexing
        std::vector<std::optional<torch::Tensor>> indices;
        
        // Create indices tensors with valid index values
        for (uint8_t i = 0; i < num_indices && offset < Size && i < tensor.dim(); ++i) {
            int64_t dim_size = tensor.size(i);
            if (dim_size <= 0) {
                continue;
            }
            
            // Determine the number of elements in the index tensor
            uint8_t index_len = 1;
            if (offset < Size) {
                index_len = (Data[offset++] % 4) + 1;  // 1-4 elements
            }
            
            // Create index values that are valid for this dimension
            std::vector<int64_t> index_values;
            for (uint8_t j = 0; j < index_len && offset < Size; ++j) {
                int64_t idx = Data[offset++] % dim_size;
                index_values.push_back(idx);
            }
            
            if (index_values.empty()) {
                index_values.push_back(0);
            }
            
            torch::Tensor index = torch::tensor(index_values, torch::kLong);
            indices.push_back(index);
        }
        
        // If no valid indices were created, create a default one
        if (indices.empty()) {
            if (tensor.size(0) > 0) {
                indices.push_back(torch::tensor({int64_t(0)}, torch::kLong));
            } else {
                return 0;
            }
        }
        
        // Create values tensor to put at the indexed locations
        // The shape of values should be broadcastable to the indexed result shape
        torch::Tensor values;
        if (offset < Size - 2) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a simple scalar or small tensor
            values = torch::ones({1}, tensor.options());
        }
        
        // Match dtype with the original tensor
        try {
            if (values.scalar_type() != tensor.scalar_type()) {
                values = values.to(tensor.scalar_type());
            }
        } catch (...) {
            // If conversion fails, create a simple tensor with matching dtype
            values = torch::ones({1}, tensor.options());
        }
        
        // Get accumulate flag (boolean)
        bool accumulate = false;
        if (offset < Size) {
            accumulate = Data[offset++] & 1;
        }
        
        // Clone tensor to avoid modifying the original (though index_put_ is in-place)
        torch::Tensor tensor_copy = tensor.clone();
        
        // Convert to c10::List for index_put_
        c10::List<std::optional<torch::Tensor>> indices_list;
        for (const auto& idx : indices) {
            indices_list.push_back(idx);
        }
        
        // Apply index_put_ operation - this may throw for shape mismatches
        // which is expected behavior, so we catch silently in inner try
        try {
            tensor_copy.index_put_(indices_list, values, accumulate);
        } catch (...) {
            // Shape mismatches are expected with random inputs
            // Try with a 0-dim tensor (scalar tensor) instead
            try {
                // Create a scalar tensor with the same dtype as the target
                torch::Tensor scalar_tensor = torch::zeros({}, tensor.options());
                tensor_copy.index_put_(indices_list, scalar_tensor, accumulate);
            } catch (...) {
                // Still failed, that's okay - invalid input combination
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