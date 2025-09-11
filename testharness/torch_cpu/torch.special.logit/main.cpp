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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.logit operation
        // logit(x) = log(x / (1 - x))
        // The operation expects input values in range (0, 1)
        
        // Try different variants of the operation
        
        // 1. Default version
        torch::Tensor result1 = torch::special::logit(input);
        
        // 2. With eps parameter (clamps values to [eps, 1-eps])
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure eps is positive and small
            float eps = std::abs(eps_raw);
            eps = std::fmod(eps, 0.5f);
            
            torch::Tensor result2 = torch::special::logit(input, std::optional<double>(eps));
        }
        
        // 3. Try with out parameter
        torch::Tensor out = torch::empty_like(input);
        torch::special::logit_out(out, input);
        
        // 4. Try with different input types
        if (offset < Size) {
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
            torch::Tensor input_cast = input.to(dtype);
            
            try {
                torch::Tensor result3 = torch::special::logit(input_cast);
            } catch (const std::exception&) {
                // Some dtypes might not be supported, that's fine
            }
        }
        
        // 5. Try with non-contiguous tensor
        if (input.dim() > 1 && input.size(0) > 1 && input.size(1) > 1) {
            torch::Tensor transposed = input.transpose(0, 1);
            if (!transposed.is_contiguous()) {
                torch::Tensor result4 = torch::special::logit(transposed);
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
