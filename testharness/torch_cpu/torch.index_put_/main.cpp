#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create the input tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create indices for indexing
        std::vector<std::optional<torch::Tensor>> indices;
        
        // Determine number of indices to create (between 1 and 3)
        uint8_t num_indices = 1;
        if (offset < Size) {
            num_indices = (Data[offset++] % 3) + 1;
        }
        
        // Create indices tensors
        for (uint8_t i = 0; i < num_indices && offset < Size; ++i) {
            // Create an index tensor with appropriate dimensions
            torch::Tensor index;
            
            // Try to create an index tensor
            if (offset < Size - 2) {
                index = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Convert to long for indexing
                if (index.scalar_type() != torch::kLong) {
                    index = index.to(torch::kLong);
                }
                
                indices.push_back(index);
            } else {
                // If not enough data, create a simple index
                if (tensor.dim() > 0) {
                    indices.push_back(torch::tensor({0}, torch::kLong));
                } else {
                    break; // Can't index a scalar tensor
                }
            }
        }
        
        // If no indices were created but tensor has dimensions, create a default index
        if (indices.empty() && tensor.dim() > 0) {
            indices.push_back(torch::tensor({0}, torch::kLong));
        }
        
        // Create values tensor to put at the indexed locations
        torch::Tensor values;
        if (offset < Size - 2) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to match dtype with the original tensor
            if (values.scalar_type() != tensor.scalar_type()) {
                try {
                    values = values.to(tensor.scalar_type());
                } catch (...) {
                    // If conversion fails, create a simple tensor with matching dtype
                    values = torch::ones({1}, tensor.options());
                }
            }
        } else {
            // Create a simple values tensor with matching dtype
            values = torch::ones({1}, tensor.options());
        }
        
        // Get accumulate flag (boolean)
        bool accumulate = false;
        if (offset < Size) {
            accumulate = Data[offset++] & 1;
        }
        
        // Clone tensor to avoid modifying the original
        torch::Tensor tensor_copy = tensor.clone();
        
        // Apply index_put_ operation
        tensor_copy.index_put_(c10::List<std::optional<torch::Tensor>>(indices), values, accumulate);
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}