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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create the elements tensor
        torch::Tensor elements = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the test_elements tensor
        torch::Tensor test_elements;
        if (offset < Size) {
            test_elements = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a simple tensor
            test_elements = torch::tensor({1, 2, 3});
        }
        
        // Get a boolean value for invert parameter
        bool invert = false;
        if (offset < Size) {
            invert = Data[offset++] & 0x01;
        }
        
        // Test torch::isin with different configurations
        
        // Basic usage
        torch::Tensor result1 = torch::isin(elements, test_elements);
        
        // With invert parameter
        torch::Tensor result2 = torch::isin(elements, test_elements, invert);
        
        // With assume_unique parameter
        bool assume_unique = false;
        if (offset < Size) {
            assume_unique = Data[offset++] & 0x01;
        }
        torch::Tensor result3 = torch::isin(elements, test_elements, invert, assume_unique);
        
        // Test with empty tensors
        if (elements.numel() > 0) {
            torch::Tensor empty_tensor = torch::empty({0}, elements.options());
            torch::Tensor result4 = torch::isin(elements, empty_tensor);
        }
        
        if (test_elements.numel() > 0) {
            torch::Tensor empty_tensor = torch::empty({0}, test_elements.options());
            torch::Tensor result5 = torch::isin(empty_tensor, test_elements);
        }
        
        // Test with scalar tensors
        if (offset + 1 < Size) {
            int64_t scalar_value = static_cast<int64_t>(Data[offset++]);
            torch::Tensor scalar_tensor = torch::tensor(scalar_value);
            torch::Tensor result6 = torch::isin(elements, scalar_tensor);
            torch::Tensor result7 = torch::isin(scalar_tensor, elements);
        }
        
        // Test with different dtypes
        if (elements.numel() > 0 && test_elements.numel() > 0) {
            // Try to convert to different dtypes if possible
            try {
                torch::Tensor elements_float = elements.to(torch::kFloat);
                torch::Tensor test_elements_float = test_elements.to(torch::kFloat);
                torch::Tensor result8 = torch::isin(elements_float, test_elements_float);
            } catch (...) {
                // Conversion might fail, that's okay
            }
            
            try {
                torch::Tensor elements_int = elements.to(torch::kInt);
                torch::Tensor test_elements_int = test_elements.to(torch::kInt);
                torch::Tensor result9 = torch::isin(elements_int, test_elements_int);
            } catch (...) {
                // Conversion might fail, that's okay
            }
        }
        
        // Test with reshaped tensors
        if (elements.dim() > 1 && elements.numel() > 0) {
            try {
                torch::Tensor flattened = elements.flatten();
                torch::Tensor result10 = torch::isin(flattened, test_elements);
            } catch (...) {
                // Reshape might fail, that's okay
            }
        }
        
        // Test with tensors of different dimensions
        if (elements.dim() > 0 && test_elements.dim() > 0) {
            try {
                torch::Tensor unsqueezed_elements = elements.unsqueeze(0);
                torch::Tensor result11 = torch::isin(unsqueezed_elements, test_elements);
            } catch (...) {
                // Operation might fail, that's okay
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
