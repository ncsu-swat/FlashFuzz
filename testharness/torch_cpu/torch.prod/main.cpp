#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a dimension value for prod if there's data left
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor has dimensions, ensure dim is within valid range
            if (input_tensor.dim() > 0) {
                // Allow negative dimensions for testing negative indexing
                dim = dim % (2 * input_tensor.dim()) - input_tensor.dim();
            }
            
            // Get keepdim flag if there's data left
            if (offset < Size) {
                keepdim = Data[offset++] & 0x1;
            }
        }
        
        // Test different variants of prod
        
        // Variant 1: prod over all dimensions
        torch::Tensor result1 = torch::prod(input_tensor);
        
        // Variant 2: prod over specific dimension with keepdim option
        torch::Tensor result2;
        if (input_tensor.dim() > 0) {
            result2 = torch::prod(input_tensor, dim, keepdim);
        }
        
        // Variant 3: prod with dtype specified
        torch::ScalarType dtype = fuzzer_utils::parseDataType(offset < Size ? Data[offset++] : 0);
        torch::Tensor result3;
        try {
            result3 = torch::prod(input_tensor, dtype);
        } catch (...) {
            // Some dtype combinations might not be supported
        }
        
        // Variant 4: prod with dimension, keepdim, and dtype
        torch::Tensor result4;
        if (input_tensor.dim() > 0) {
            try {
                result4 = torch::prod(input_tensor, dim, keepdim, dtype);
            } catch (...) {
                // Some dtype combinations might not be supported
            }
        }
        
        // Variant 5: out variant
        if (input_tensor.dim() > 0) {
            try {
                torch::Tensor out = torch::empty_like(input_tensor);
                torch::prod_out(out, input_tensor, dim, keepdim);
            } catch (...) {
                // Out variant might have shape compatibility requirements
            }
        }
        
        // Variant 6: Test with empty tensor
        if (offset + 1 < Size) {
            try {
                std::vector<int64_t> empty_shape = {0};
                torch::Tensor empty_tensor = torch::empty(empty_shape);
                torch::Tensor empty_result = torch::prod(empty_tensor);
            } catch (...) {
                // Empty tensor might cause issues
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