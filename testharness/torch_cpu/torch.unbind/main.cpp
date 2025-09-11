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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get dimension to unbind along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply unbind operation
        std::vector<torch::Tensor> result;
        
        // Try different variants of unbind
        if (offset < Size) {
            uint8_t variant = Data[offset++];
            
            switch (variant % 2) {
                case 0:
                    // Basic unbind with dimension
                    result = torch::unbind(input_tensor, dim);
                    break;
                    
                case 1:
                    // Unbind with default dimension (0)
                    result = torch::unbind(input_tensor);
                    break;
            }
        } else {
            // Default case if we don't have enough data for variant
            result = torch::unbind(input_tensor, dim);
        }
        
        // Perform some operations on the result to ensure it's used
        if (!result.empty()) {
            for (auto& tensor : result) {
                auto sizes = tensor.sizes();
                auto numel = tensor.numel();
                auto dtype = tensor.dtype();
                
                // Simple operation to ensure tensor is valid
                if (numel > 0) {
                    tensor = tensor + 1;
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
