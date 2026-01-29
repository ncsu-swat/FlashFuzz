#include "fuzzer_utils.h"
#include <iostream>

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
            test_elements = torch::tensor({1, 2, 3});
        }
        
        // Get boolean values for parameters
        bool assume_unique = false;
        bool invert = false;
        if (offset < Size) {
            assume_unique = Data[offset++] & 0x01;
        }
        if (offset < Size) {
            invert = Data[offset++] & 0x01;
        }
        
        // Basic usage
        torch::Tensor result1 = torch::isin(elements, test_elements);
        
        // With assume_unique and invert parameters (correct parameter order)
        torch::Tensor result2 = torch::isin(elements, test_elements, assume_unique, invert);
        
        // Test with empty tensors
        try {
            if (elements.numel() > 0) {
                torch::Tensor empty_tensor = torch::empty({0}, elements.options());
                torch::Tensor result3 = torch::isin(elements, empty_tensor);
            }
        } catch (...) {
            // Empty tensor operations might fail
        }
        
        try {
            if (test_elements.numel() > 0) {
                torch::Tensor empty_tensor = torch::empty({0}, test_elements.options());
                torch::Tensor result4 = torch::isin(empty_tensor, test_elements);
            }
        } catch (...) {
            // Empty tensor operations might fail
        }
        
        // Test with scalar tensors
        if (offset < Size) {
            try {
                int64_t scalar_value = static_cast<int64_t>(Data[offset++]);
                torch::Tensor scalar_tensor = torch::tensor(scalar_value);
                torch::Tensor result5 = torch::isin(elements, scalar_tensor);
                torch::Tensor result6 = torch::isin(scalar_tensor, elements);
            } catch (...) {
                // Scalar operations might fail
            }
        }
        
        // Test with different dtypes
        if (elements.numel() > 0 && test_elements.numel() > 0) {
            try {
                torch::Tensor elements_float = elements.to(torch::kFloat);
                torch::Tensor test_elements_float = test_elements.to(torch::kFloat);
                torch::Tensor result7 = torch::isin(elements_float, test_elements_float);
            } catch (...) {
                // Conversion might fail
            }
            
            try {
                torch::Tensor elements_int = elements.to(torch::kInt);
                torch::Tensor test_elements_int = test_elements.to(torch::kInt);
                torch::Tensor result8 = torch::isin(elements_int, test_elements_int);
            } catch (...) {
                // Conversion might fail
            }
            
            try {
                torch::Tensor elements_long = elements.to(torch::kLong);
                torch::Tensor test_elements_long = test_elements.to(torch::kLong);
                torch::Tensor result9 = torch::isin(elements_long, test_elements_long, true, false);
            } catch (...) {
                // Conversion might fail
            }
        }
        
        // Test with reshaped tensors
        if (elements.dim() > 1 && elements.numel() > 0) {
            try {
                torch::Tensor flattened = elements.flatten();
                torch::Tensor result10 = torch::isin(flattened, test_elements);
            } catch (...) {
                // Reshape might fail
            }
        }
        
        // Test with tensors of different dimensions
        if (elements.dim() > 0 && test_elements.dim() > 0) {
            try {
                torch::Tensor unsqueezed_elements = elements.unsqueeze(0);
                torch::Tensor result11 = torch::isin(unsqueezed_elements, test_elements);
            } catch (...) {
                // Operation might fail
            }
        }
        
        // Test with contiguous and non-contiguous tensors
        if (elements.dim() >= 2 && elements.numel() > 0) {
            try {
                torch::Tensor transposed = elements.transpose(0, 1);
                torch::Tensor result12 = torch::isin(transposed, test_elements);
            } catch (...) {
                // Transpose might fail for 1D tensors
            }
        }
        
        // Test with cloned tensors to ensure data independence
        try {
            torch::Tensor elements_clone = elements.clone();
            torch::Tensor test_elements_clone = test_elements.clone();
            torch::Tensor result13 = torch::isin(elements_clone, test_elements_clone, false, true);
        } catch (...) {
            // Clone operations might fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}