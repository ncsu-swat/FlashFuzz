#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for dtype selection
        if (Size < 1) {
            return 0;
        }
        
        // Parse dtype selector from the first byte
        uint8_t dtype_selector = Data[0];
        offset++;
        
        // Map the selector to a valid ScalarType
        torch::ScalarType dtype;
        switch (dtype_selector % 4) {
            case 0:
                dtype = torch::ScalarType::Float;
                break;
            case 1:
                dtype = torch::ScalarType::BFloat16;
                break;
            case 2:
                dtype = torch::ScalarType::Half;
                break;
            case 3:
                dtype = torch::ScalarType::Double;
                break;
            default:
                dtype = torch::ScalarType::Float;
        }
        
        // Set the autocast XLA dtype
        torch::autocast::set_autocast_xla_dtype(dtype);
        
        // Create a tensor to test if the setting affects tensor operations
        if (offset < Size) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Test if autocast setting affects operations
                torch::Tensor result = tensor + tensor;
                
                // Reset to default
                torch::autocast::set_autocast_xla_dtype(torch::ScalarType::Float);
            } catch (const std::exception& e) {
                // Reset to default even if tensor operations fail
                torch::autocast::set_autocast_xla_dtype(torch::ScalarType::Float);
            }
        } else {
            // Reset to default
            torch::autocast::set_autocast_xla_dtype(torch::ScalarType::Float);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
