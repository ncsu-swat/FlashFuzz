#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to use with iteration
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test basic tensor iteration using standard C++ iterators
        auto tensor_data = tensor.data_ptr<float>();
        auto numel = tensor.numel();
        
        // Test basic operations on tensor elements
        if (numel > 0) {
            // Access first element
            auto first_element = tensor_data[0];
            
            // Try to iterate through the elements
            int count = 0;
            int max_iterations = std::min(100, static_cast<int>(numel)); // Prevent infinite loops
            
            for (int i = 0; i < max_iterations; ++i) {
                auto element = tensor_data[i];
                count++;
            }
        }
        
        // Test other operations if there's more data
        if (offset + 1 < Size) {
            uint8_t op_selector = Data[offset++];
            
            // Test different operations based on the selector
            switch (op_selector % 3) {
                case 0: {
                    // Test tensor copy
                    auto tensor_copy = tensor.clone();
                    auto copy_data = tensor_copy.data_ptr<float>();
                    if (tensor_copy.numel() > 0) {
                        auto element = copy_data[0];
                    }
                    break;
                }
                case 1: {
                    // Test tensor view operations
                    if (tensor.numel() > 1) {
                        auto reshaped = tensor.view({-1});
                        auto reshaped_data = reshaped.data_ptr<float>();
                        if (reshaped.numel() > 0) {
                            auto element = reshaped_data[0];
                        }
                    }
                    break;
                }
                case 2: {
                    // Test with different tensor if we have more data
                    if (offset < Size) {
                        torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                        auto another_data = another_tensor.data_ptr<float>();
                        
                        // Try to access elements
                        if (another_tensor.numel() > 0) {
                            auto element = another_data[0];
                        }
                    }
                    break;
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