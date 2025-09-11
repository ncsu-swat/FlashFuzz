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
        
        // Extract dimension parameter if we have more data
        int64_t dim = -1;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Get keepdim parameter if available
            if (offset < Size) {
                keepdim = Data[offset++] & 0x1;
            }
        }
        
        // Apply torch.mean in different ways to test various code paths
        torch::Tensor result;
        
        // Test different variants of mean
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 4;
            
            switch (variant) {
                case 0:
                    // Mean over all dimensions
                    result = torch::mean(input_tensor);
                    break;
                    
                case 1:
                    // Mean over specific dimension
                    result = torch::mean(input_tensor, dim, keepdim);
                    break;
                    
                case 2:
                    // Mean with dtype specified
                    if (offset < Size) {
                        auto dtype_selector = Data[offset++];
                        auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                        result = torch::mean(input_tensor, dtype);
                    } else {
                        result = torch::mean(input_tensor);
                    }
                    break;
                    
                case 3:
                    // Mean with dimension and dtype
                    if (offset < Size) {
                        auto dtype_selector = Data[offset++];
                        auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                        result = torch::mean(input_tensor, dim, keepdim, dtype);
                    } else {
                        result = torch::mean(input_tensor, dim, keepdim);
                    }
                    break;
            }
        } else {
            // Default case if no variant byte available
            result = torch::mean(input_tensor);
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
