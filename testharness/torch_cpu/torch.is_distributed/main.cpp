#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Call torch.is_distributed on the tensor
        bool is_distributed = torch::is_distributed(tensor);
        
        // Try with different tensor properties if we have more data
        if (offset + 1 < Size) {
            // Use the next byte to determine if we should try with a different tensor
            uint8_t tensor_modifier = Data[offset++];
            
            if (tensor_modifier % 2 == 1) {
                // Create another tensor and test
                torch::Tensor another_tensor = tensor.clone();
                bool is_distributed_clone = torch::is_distributed(another_tensor);
            }
        }
        
        // Try with tensor operations if we have more data
        if (offset + 1 < Size) {
            uint8_t operation_selector = Data[offset++];
            
            try {
                torch::Tensor modified_tensor;
                switch (operation_selector % 4) {
                    case 0:
                        modified_tensor = tensor.detach();
                        break;
                    case 1:
                        modified_tensor = tensor.contiguous();
                        break;
                    case 2:
                        modified_tensor = tensor.cpu();
                        break;
                    case 3:
                        modified_tensor = tensor.view(-1);
                        break;
                }
                
                bool is_distributed_modified = torch::is_distributed(modified_tensor);
            } catch (const c10::Error& e) {
                // Expected for some tensor operations
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