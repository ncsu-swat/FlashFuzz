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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dim parameter if we have more data
        std::optional<int64_t> dim = std::nullopt;
        if (offset + sizeof(int64_t) <= Size) {
            int64_t dim_val;
            std::memcpy(&dim_val, Data + offset, sizeof(int64_t));
            dim = dim_val;
            offset += sizeof(int64_t);
        }
        
        // Call count_nonzero with different parameter combinations
        torch::Tensor result;
        
        // Decide which variant to call based on remaining data
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 2;
            
            switch (variant) {
                case 0:
                    // Basic count_nonzero without additional parameters
                    result = torch::count_nonzero(input_tensor);
                    break;
                    
                case 1:
                    // count_nonzero with dim parameter
                    result = torch::count_nonzero(input_tensor, dim);
                    break;
            }
        } else {
            // Default to basic count_nonzero if no more data
            result = torch::count_nonzero(input_tensor);
        }
        
        // Access result to ensure computation is performed
        if (result.defined()) {
            auto numel = result.numel();
            if (numel > 0) {
                auto item = result.item();
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
