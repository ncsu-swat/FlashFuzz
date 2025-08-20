#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 10) {
            return 0;
        }
        
        // Create the input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the index tensor (must be 1D and integer type)
        torch::Tensor index_tensor;
        if (offset < Size) {
            // Try to create index tensor from remaining data
            torch::Tensor temp_index = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure index tensor is 1D and long type
            if (temp_index.dim() > 0) {
                // Flatten to 1D if needed
                index_tensor = temp_index.flatten().to(torch::kLong);
            } else {
                // Create a simple index tensor if we couldn't get a valid one
                index_tensor = torch::tensor({0, 1}, torch::kLong);
            }
        } else {
            // Default index tensor if we've consumed all data
            index_tensor = torch::tensor({0, 1}, torch::kLong);
        }
        
        // Create the source tensor to add
        torch::Tensor source_tensor;
        if (offset < Size) {
            source_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default source tensor with same dtype as input
            source_tensor = torch::ones_like(input_tensor);
        }
        
        // Get a dimension to index along
        int64_t dim = 0;
        if (offset < Size && input_tensor.dim() > 0) {
            // Use a byte from the input to determine the dimension
            dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), input_tensor.dim());
        }
        
        // Ensure indices are within bounds of the dimension
        if (input_tensor.dim() > 0 && input_tensor.size(dim) > 0) {
            index_tensor = index_tensor % input_tensor.size(dim);
            // Handle negative indices by making them positive
            index_tensor = torch::where(index_tensor < 0, index_tensor + input_tensor.size(dim), index_tensor);
        }
        
        // Get alpha value for scaling
        float alpha = 1.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Apply index_add operation
        torch::Tensor result;
        
        // Try different variants of index_add
        if (offset < Size && Data[offset] % 3 == 0) {
            // Variant 1: Using index_add_ (in-place)
            result = input_tensor.clone();
            result.index_add_(dim, index_tensor, source_tensor, alpha);
        } else if (offset < Size && Data[offset] % 3 == 1) {
            // Variant 2: Using index_add (out-of-place)
            result = input_tensor.index_add(dim, index_tensor, source_tensor, alpha);
        } else {
            // Variant 3: Using torch::index_add function
            result = torch::index_add(input_tensor, dim, index_tensor, source_tensor, alpha);
        }
        
        // Ensure the operation completed by accessing a value
        if (result.numel() > 0) {
            float val = result.item<float>();
            (void)val; // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}