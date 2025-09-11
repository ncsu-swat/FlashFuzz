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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for norm operation if we have more data
        double p = 2.0; // Default p-norm
        int64_t dim = -1; // Default dimension
        bool keepdim = false;
        
        // Parse p value if we have enough data
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse dim value if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure dim is within valid range for the tensor
            if (input.dim() > 0) {
                dim = dim % (2 * input.dim()) - input.dim();
            } else {
                dim = 0;
            }
        }
        
        // Parse keepdim value if we have enough data
        if (offset < Size) {
            keepdim = Data[offset] & 0x1;
            offset++;
        }
        
        // Try different variants of torch::norm
        
        // Variant 1: Basic norm with default parameters
        torch::Tensor result1 = torch::norm(input);
        
        // Variant 2: Norm with specified p
        torch::Tensor result2 = torch::norm(input, p);
        
        // Variant 3: Norm with specified p and dim
        torch::Tensor result3 = torch::norm(input, p, {dim}, keepdim);
        
        // Variant 4: Frobenius norm
        torch::Tensor result4 = torch::frobenius_norm(input);
        
        // Variant 5: Nuclear norm
        if (input.dim() >= 2) {
            try {
                torch::Tensor result5 = torch::nuclear_norm(input);
            } catch (const std::exception&) {
                // Nuclear norm may not be supported for all tensor types
            }
        }
        
        // Variant 6: Norm with specified dim and keepdim but default p
        torch::Tensor result6 = torch::norm(input, 2.0, {dim}, keepdim);
        
        // Variant 7: Norm with infinity
        torch::Tensor result7 = torch::norm(input, INFINITY);
        
        // Variant 8: Norm with negative infinity
        torch::Tensor result8 = torch::norm(input, -INFINITY);
        
        // Variant 9: Norm with 0 (count of non-zero elements)
        torch::Tensor result9 = torch::norm(input, 0.0);
        
        // Variant 10: Norm with negative p value
        if (p > 0) {
            try {
                torch::Tensor result10 = torch::norm(input, -p);
            } catch (const std::exception&) {
                // Negative p may not be supported
            }
        }
        
        // Variant 11: Norm with very large p value
        try {
            torch::Tensor result11 = torch::norm(input, 1e10);
        } catch (const std::exception&) {
            // Very large p may cause numerical issues
        }
        
        // Variant 12: Norm with very small positive p value
        try {
            torch::Tensor result12 = torch::norm(input, 1e-10);
        } catch (const std::exception&) {
            // Very small p may cause numerical issues
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
