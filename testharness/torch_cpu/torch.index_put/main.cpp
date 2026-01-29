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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create the source tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create indices tensors
        std::vector<torch::Tensor> indices;
        
        // Determine number of indices to create (between 1 and 3)
        uint8_t num_indices = 1;
        if (offset < Size) {
            num_indices = (Data[offset++] % 3) + 1;
        }
        
        // Create index tensors
        for (uint8_t i = 0; i < num_indices && offset < Size; ++i) {
            torch::Tensor index = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert to long type for indexing
            if (index.scalar_type() != torch::kLong) {
                index = index.to(torch::kLong);
            }
            
            // Clamp indices to valid range for the tensor dimension
            if (tensor.dim() > i && tensor.size(i) > 0) {
                index = index.abs().fmod(tensor.size(i));
            }
            
            indices.push_back(index);
        }
        
        // Create values tensor
        torch::Tensor values;
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Default values tensor if we've run out of data
            values = torch::ones({1}, tensor.options());
        }
        
        // Get accumulate flag
        bool accumulate = false;
        if (offset < Size) {
            accumulate = Data[offset++] & 0x1;
        }
        
        // Try different variants of index_put
        if (indices.empty()) {
            // If no indices were created, use empty list
            c10::List<torch::optional<torch::Tensor>> empty_indices;
            try {
                tensor.index_put_(empty_indices, values, accumulate);
            } catch (const std::exception&) {
                // Expected to fail in some cases
            }
        } else {
            // Convert indices to a c10::List
            c10::List<torch::optional<torch::Tensor>> optional_indices;
            for (const auto& idx : indices) {
                optional_indices.push_back(idx);
            }
            
            // Try index_put_ with the indices
            try {
                tensor.index_put_(optional_indices, values, accumulate);
            } catch (const std::exception&) {
                // Expected to fail for shape mismatches
            }
            
            // Try non-inplace version if we have enough data
            if (offset < Size && (Data[offset++] & 0x1)) {
                try {
                    torch::Tensor result = tensor.index_put(optional_indices, values, accumulate);
                    (void)result; // Suppress unused warning
                } catch (const std::exception&) {
                    // Expected to fail for shape mismatches
                }
            }
            
            // Try with some None indices if we have multiple indices
            if (indices.size() > 1 && offset < Size && (Data[offset++] & 0x1)) {
                c10::List<torch::optional<torch::Tensor>> mixed_indices;
                mixed_indices.push_back(torch::nullopt);
                for (size_t i = 1; i < indices.size(); ++i) {
                    mixed_indices.push_back(indices[i]);
                }
                try {
                    tensor.index_put_(mixed_indices, values, accumulate);
                } catch (const std::exception&) {
                    // Expected to fail for shape mismatches
                }
            }
        }
        
        // Try with empty values tensor
        if (offset < Size && (Data[offset++] & 0x1) && !indices.empty()) {
            c10::List<torch::optional<torch::Tensor>> optional_indices;
            for (const auto& idx : indices) {
                optional_indices.push_back(idx);
            }
            
            torch::Tensor empty_values = torch::empty({0}, tensor.options());
            try {
                tensor.index_put_(optional_indices, empty_values, accumulate);
            } catch (const std::exception&) {
                // Expected to fail in some cases
            }
        }
        
        // Try with boolean mask if we have data left
        if (offset < Size && (Data[offset++] & 0x1) && tensor.numel() > 0) {
            // Create boolean mask deterministically from fuzzer data
            torch::Tensor mask = torch::zeros_like(tensor, torch::kBool);
            
            // Use fuzzer data to determine which elements to set true
            if (offset < Size) {
                int64_t numel = tensor.numel();
                torch::Tensor flat_mask = mask.flatten();
                
                // Set some elements based on fuzzer data
                int64_t num_to_set = std::min(static_cast<int64_t>(Size - offset), numel);
                num_to_set = std::min(num_to_set, static_cast<int64_t>(16)); // Limit iterations
                
                for (int64_t i = 0; i < num_to_set && offset < Size; ++i) {
                    int64_t idx = Data[offset++] % numel;
                    flat_mask[idx] = true;
                }
                
                mask = flat_mask.reshape(tensor.sizes());
            }
            
            try {
                c10::List<torch::optional<torch::Tensor>> mask_indices;
                mask_indices.push_back(mask);
                
                // Values need to match the number of true elements in mask
                int64_t num_true = mask.sum().item<int64_t>();
                if (num_true > 0) {
                    torch::Tensor mask_values = torch::ones({num_true}, tensor.options());
                    tensor.index_put_(mask_indices, mask_values, accumulate);
                }
            } catch (const std::exception&) {
                // May fail for some shapes/dtypes
            }
        }
        
        // Test with scalar value (converted to tensor)
        if (offset < Size && (Data[offset++] & 0x1) && !indices.empty()) {
            c10::List<torch::optional<torch::Tensor>> optional_indices;
            for (const auto& idx : indices) {
                optional_indices.push_back(idx);
            }
            
            // Convert scalar to tensor - index_put_ with c10::List requires a Tensor, not Scalar
            torch::Tensor scalar_tensor = torch::tensor(1.0, tensor.options());
            try {
                tensor.index_put_(optional_indices, scalar_tensor);
            } catch (const std::exception&) {
                // May fail for some configurations
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