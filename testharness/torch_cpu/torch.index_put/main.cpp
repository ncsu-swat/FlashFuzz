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
            
            indices.push_back(index);
        }
        
        // Create values tensor
        torch::Tensor values;
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Default values tensor if we've run out of data
            values = torch::ones_like(tensor);
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
            tensor.index_put_(empty_indices, values, accumulate);
        } else {
            // Convert indices to a c10::List
            c10::List<torch::optional<torch::Tensor>> optional_indices;
            for (const auto& idx : indices) {
                optional_indices.push_back(idx);
            }
            
            // Try index_put with the indices
            tensor.index_put_(optional_indices, values, accumulate);
            
            // Try non-inplace version if we have enough data
            if (offset < Size && (Data[offset++] & 0x1)) {
                torch::Tensor result = tensor.index_put(optional_indices, values, accumulate);
            }
            
            // Try with some None indices if we have multiple indices
            if (indices.size() > 1 && offset < Size && (Data[offset++] & 0x1)) {
                c10::List<torch::optional<torch::Tensor>> mixed_indices = optional_indices;
                mixed_indices[0] = torch::nullopt;
                tensor.index_put_(mixed_indices, values, accumulate);
            }
        }
        
        // Try with empty values tensor
        if (offset < Size && (Data[offset++] & 0x1)) {
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
        if (offset < Size && (Data[offset++] & 0x1)) {
            torch::Tensor mask = torch::zeros_like(tensor, torch::kBool);
            if (tensor.numel() > 0) {
                // Set some elements to true
                int64_t num_true = std::max(int64_t(1), tensor.numel() / 2);
                mask.index_put_({torch::randperm(tensor.numel()).slice(0, num_true)}, torch::ones({num_true}, torch::kBool));
            }
            
            try {
                c10::List<torch::optional<torch::Tensor>> mask_indices;
                mask_indices.push_back(mask);
                tensor.index_put_(mask_indices, values, accumulate);
            } catch (const std::exception&) {
                // May fail for some shapes/dtypes
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
