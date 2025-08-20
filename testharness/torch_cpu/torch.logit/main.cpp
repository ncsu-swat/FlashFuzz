#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract eps parameter if we have more data
        double eps = 1e-6; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is within a reasonable range
            eps = std::abs(eps);
            if (eps == 0.0) {
                eps = 1e-6;
            }
        }
        
        // Apply logit operation with default eps
        torch::Tensor result1 = torch::logit(input);
        
        // Apply logit operation with custom eps
        torch::Tensor result2 = torch::logit(input, eps);
        
        // Apply logit operation with None eps (using std::nullopt)
        torch::Tensor result3 = torch::logit(input, std::nullopt);
        
        // Apply in-place logit operation
        torch::Tensor input_copy = input.clone();
        input_copy.logit_();
        
        // Apply in-place logit operation with custom eps
        torch::Tensor input_copy2 = input.clone();
        input_copy2.logit_(eps);
        
        // Apply in-place logit operation with None eps
        torch::Tensor input_copy3 = input.clone();
        input_copy3.logit_(std::nullopt);
        
        // Try with different dtypes if input is floating point
        if (input.is_floating_point()) {
            // Convert to different dtypes and test
            auto dtypes = {torch::kFloat, torch::kDouble, torch::kHalf};
            for (auto dtype : dtypes) {
                if (torch::can_cast(input.scalar_type(), dtype)) {
                    torch::Tensor converted = input.to(dtype);
                    torch::Tensor result = torch::logit(converted, eps);
                }
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