#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Early exit if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse normalized_shape
        uint8_t normalized_shape_rank = 0;
        if (offset < Size) {
            normalized_shape_rank = fuzzer_utils::parseRank(Data[offset++]);
        }
        
        std::vector<int64_t> normalized_shape;
        if (normalized_shape_rank > 0 && offset < Size) {
            normalized_shape = fuzzer_utils::parseShape(Data, offset, Size, normalized_shape_rank);
        }
        
        // Parse weight and bias options
        bool has_weight = false;
        double eps = 1e-5;
        
        if (offset < Size) {
            has_weight = Data[offset++] & 0x1;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is positive and not too small to avoid numerical issues
            if (eps <= 0 || !std::isfinite(eps)) {
                eps = 1e-5;
            }
        }
        
        // Create weight tensor if needed
        torch::Tensor weight;
        
        if (has_weight) {
            if (normalized_shape.empty()) {
                // Default to a single dimension if normalized_shape is empty
                normalized_shape = {1};
            }
            weight = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Apply rms_norm operation
        torch::Tensor output;
        
        if (normalized_shape.empty() && input.dim() > 0) {
            // If normalized_shape is empty, use the last dimension of input
            int64_t last_dim = input.dim() - 1;
            normalized_shape = {input.size(last_dim)};
        }
        
        if (has_weight) {
            output = torch::rms_norm(input, normalized_shape, weight, eps);
        } else {
            output = torch::rms_norm(input, normalized_shape, {}, eps);
        }
        
        // Ensure the output is used to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}