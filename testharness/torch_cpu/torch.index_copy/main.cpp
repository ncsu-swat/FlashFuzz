#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 10) {
            return 0;
        }
        
        // Create the source tensor
        torch::Tensor src = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the tensor to be modified (destination)
        torch::Tensor self = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create index tensor (must be 1D LongTensor)
        torch::Tensor index;
        try {
            // Create a raw index tensor
            torch::Tensor raw_index = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert to 1D LongTensor
            if (raw_index.dim() == 0) {
                // If scalar, convert to 1D tensor with one element
                index = raw_index.reshape({1}).to(torch::kInt64);
            } else {
                // Flatten and convert to LongTensor
                index = raw_index.flatten().to(torch::kInt64);
            }
        } catch (const std::exception&) {
            // If index creation fails, create a simple index tensor
            index = torch::tensor({0}, torch::kInt64);
        }
        
        // Get a dimension to index along
        int64_t dim = 0;
        if (offset < Size && self.dim() > 0) {
            dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), static_cast<int64_t>(self.dim()));
        }
        
        // Try different variants of index_copy
        try {
            // Variant 1: Using index_copy_ (in-place)
            torch::Tensor result1 = self.clone();
            result1.index_copy_(dim, index, src);
        } catch (const std::exception&) {
            // Ignore exceptions from the operation
        }
        
        try {
            // Variant 2: Using index_copy (out-of-place)
            torch::Tensor result2 = self.index_copy(dim, index, src);
        } catch (const std::exception&) {
            // Ignore exceptions from the operation
        }
        
        try {
            // Variant 3: Using functional form
            torch::Tensor result3 = torch::index_copy(self, dim, index, src);
        } catch (const std::exception&) {
            // Ignore exceptions from the operation
        }
        
        // Try with negative dimension
        if (self.dim() > 0) {
            try {
                int64_t neg_dim = -1;
                torch::Tensor result4 = self.index_copy(neg_dim, index, src);
            } catch (const std::exception&) {
                // Ignore exceptions from the operation
            }
        }
        
        // Try with out-of-bounds indices
        try {
            torch::Tensor out_of_bounds_index = torch::tensor({-100, 100}, torch::kInt64);
            torch::Tensor result5 = self.index_copy(dim, out_of_bounds_index, src);
        } catch (const std::exception&) {
            // Ignore exceptions from the operation
        }
        
        // Try with empty index tensor
        try {
            torch::Tensor empty_index = torch::tensor({}, torch::kInt64);
            torch::Tensor result6 = self.index_copy(dim, empty_index, src);
        } catch (const std::exception&) {
            // Ignore exceptions from the operation
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
