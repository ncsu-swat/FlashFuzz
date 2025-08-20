#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to fill along
        int64_t dim = 0;
        if (input_tensor.dim() > 0) {
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&dim, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                // Use modulo to ensure dim is within valid range
                dim = dim % input_tensor.dim();
                if (dim < 0) {
                    dim += input_tensor.dim();
                }
            }
        }
        
        // Create index tensor (indices to fill)
        torch::Tensor index_tensor;
        if (offset < Size) {
            index_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert index tensor to long type for indexing
            if (index_tensor.scalar_type() != torch::kLong) {
                index_tensor = index_tensor.to(torch::kLong);
            }
        } else {
            // Create a simple index tensor if we don't have enough data
            if (input_tensor.dim() > 0 && input_tensor.size(dim) > 0) {
                index_tensor = torch::tensor({0}, torch::kLong);
            } else {
                index_tensor = torch::tensor({}, torch::kLong);
            }
        }
        
        // Get a value to fill with
        torch::Scalar value;
        if (offset + sizeof(float) <= Size) {
            float val;
            std::memcpy(&val, Data + offset, sizeof(float));
            offset += sizeof(float);
            value = val;
        } else {
            value = 1.0f;
        }
        
        // Apply index_fill operation
        torch::Tensor result;
        
        // Try different variants of index_fill
        if (offset < Size && Data[offset] % 2 == 0) {
            // Use the in-place version
            result = input_tensor.clone();
            result.index_fill_(dim, index_tensor, value);
        } else {
            // Use the out-of-place version
            result = input_tensor.index_fill(dim, index_tensor, value);
        }
        
        // Try to access elements of the result to ensure it's valid
        if (result.numel() > 0) {
            auto item = result.item();
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}