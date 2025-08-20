#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for the input tensor and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for diag_embed if there's data left
        int64_t offset_param = 0;
        int64_t dim1_param = 0;
        int64_t dim2_param = -1;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&offset_param, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim1_param, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim2_param, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply diag_embed operation
        torch::Tensor result;
        
        // Try different combinations of parameters
        if (offset % 3 == 0) {
            // Use all three parameters
            result = torch::diag_embed(input, offset_param, dim1_param, dim2_param);
        } else if (offset % 3 == 1) {
            // Use two parameters
            result = torch::diag_embed(input, offset_param, dim1_param);
        } else {
            // Use just one parameter
            result = torch::diag_embed(input, offset_param);
        }
        
        // Force evaluation of the result
        result.sum().item<float>();
        
        // Try another variant with default parameters
        torch::Tensor result2 = torch::diag_embed(input);
        result2.sum().item<float>();
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}