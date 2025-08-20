#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to apply logcumsumexp along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor has dimensions, make sure dim is within valid range
            if (input.dim() > 0) {
                // Allow negative dimensions for testing edge cases
                dim = dim % (2 * input.dim());
                if (dim >= input.dim()) {
                    dim -= 2 * input.dim();
                }
            }
        }
        
        // Apply logcumsumexp operation
        torch::Tensor result = torch::logcumsumexp(input, dim);
        
        // Try with out parameter if we have enough data
        if (offset < Size) {
            // Create output tensor with same shape as expected result
            torch::Tensor out = torch::empty_like(result);
            torch::logcumsumexp_out(out, input, dim);
        }
        
        // Try with dimname if we have enough data
        if (offset < Size && input.has_names()) {
            auto names = input.names();
            if (!names.empty()) {
                torch::Tensor result_dimname = torch::logcumsumexp(input, names[0]);
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