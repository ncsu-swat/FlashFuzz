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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse q value (quantile value between 0 and 1)
        float q = 0.5f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&q, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure q is between 0 and 1
            q = std::abs(q);
            q = q - std::floor(q);
        }
        
        // Parse dim value
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse keepdim value
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Parse interpolation mode
        std::string interpolation = "linear";
        if (offset < Size) {
            uint8_t interp_selector = Data[offset++] % 4;
            switch (interp_selector) {
                case 0: interpolation = "linear"; break;
                case 1: interpolation = "lower"; break;
                case 2: interpolation = "higher"; break;
                case 3: interpolation = "midpoint"; break;
            }
        }
        
        // Try different variants of quantile
        try {
            // Variant 1: Basic quantile with scalar q
            torch::Tensor result1 = torch::quantile(input_tensor, q);
            
            // Variant 2: Quantile with specified dimension
            torch::Tensor result2 = torch::quantile(input_tensor, q, dim, keepdim);
            
            // Variant 3: Quantile with interpolation mode
            torch::Tensor result3 = torch::quantile(input_tensor, q, c10::nullopt, false, interpolation);
            
            // Variant 4: Full quantile with all parameters
            torch::Tensor result4 = torch::quantile(input_tensor, q, dim, keepdim, interpolation);
            
            // Variant 5: Try with tensor q instead of scalar q
            std::vector<float> q_values = {0.25f, 0.5f, 0.75f};
            torch::Tensor q_tensor = torch::tensor(q_values);
            torch::Tensor result5 = torch::quantile(input_tensor, q_tensor, dim, keepdim, interpolation);
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and not considered bugs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}