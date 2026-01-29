#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
                    // Fill with a tensor scalar (0-dim tensor)
                    auto scalar_tensor = torch::tensor(fill_value);
                    tensor.fill_(scalar_tensor);
                    break;
                }
            }
        } else {
            // Default case: just fill with the double value
            tensor.fill_(fill_value);
        }
        
        // Verify the operation completed by checking the tensor is valid
        // Use sum() instead of item() since item() only works on single-element tensors
        if (tensor.numel() > 0) {
            auto sum_result = tensor.sum();
            (void)sum_result;  // Suppress unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}