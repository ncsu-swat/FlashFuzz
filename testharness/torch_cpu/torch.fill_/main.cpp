#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the fill value
        if (Size < 1) {
            return 0;
        }
        
        // Create a tensor to fill
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a value to fill the tensor with
        // Use the next byte if available, otherwise use the first byte
        double fill_value = 0.0;
        if (offset < Size) {
            // Use the next byte as a basis for the fill value
            fill_value = static_cast<double>(Data[offset++]);
        } else if (Size > 0) {
            // Fallback to using the first byte
            fill_value = static_cast<double>(Data[0]);
        }
        
        // Try different scalar types for the fill value
        if (offset < Size) {
            uint8_t scalar_type_selector = Data[offset++];
            
            // Apply fill_ with different scalar types based on the selector
            switch (scalar_type_selector % 5) {
                case 0: {
                    // Fill with double
                    tensor.fill_(fill_value);
                    break;
                }
                case 1: {
                    // Fill with int
                    int int_value = static_cast<int>(fill_value);
                    tensor.fill_(int_value);
                    break;
                }
                case 2: {
                    // Fill with bool
                    bool bool_value = (static_cast<int>(fill_value) % 2) == 1;
                    tensor.fill_(bool_value);
                    break;
                }
                case 3: {
                    // Fill with float
                    float float_value = static_cast<float>(fill_value);
                    tensor.fill_(float_value);
                    break;
                }
                case 4: {
                    // Fill with a tensor scalar
                    auto scalar_tensor = torch::tensor(fill_value);
                    tensor.fill_(scalar_tensor);
                    break;
                }
            }
        } else {
            // Default case: just fill with the double value
            tensor.fill_(fill_value);
        }
        
        // Try to access elements to ensure the operation completed
        if (tensor.numel() > 0) {
            auto item = tensor.item();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}