#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some bytes for basic parameters
        if (Size < 10) return 0;

        // Extract boolean flags
        bool assume_unique = extractBool(Data, Size, offset);
        bool invert = extractBool(Data, Size, offset);
        
        // Extract configuration for elements and test_elements
        uint8_t config = extractUint8(Data, Size, offset) % 4;
        
        torch::Tensor elements, test_elements;
        torch::Scalar elements_scalar, test_elements_scalar;
        bool elements_is_scalar = false, test_elements_is_scalar = false;
        
        // Configure based on extracted config (ensure at least one is tensor)
        switch (config) {
            case 0: // Both tensors
                elements = generateRandomTensor(Data, Size, offset);
                test_elements = generateRandomTensor(Data, Size, offset);
                break;
            case 1: // elements is scalar, test_elements is tensor
                elements_scalar = extractScalar(Data, Size, offset);
                elements_is_scalar = true;
                test_elements = generateRandomTensor(Data, Size, offset);
                break;
            case 2: // elements is tensor, test_elements is scalar
                elements = generateRandomTensor(Data, Size, offset);
                test_elements_scalar = extractScalar(Data, Size, offset);
                test_elements_is_scalar = true;
                break;
            case 3: // Both tensors (fallback to avoid both scalars)
                elements = generateRandomTensor(Data, Size, offset);
                test_elements = generateRandomTensor(Data, Size, offset);
                break;
        }

        torch::Tensor result;
        
        // Call torch::isin with different combinations
        if (elements_is_scalar && !test_elements_is_scalar) {
            result = torch::isin(elements_scalar, test_elements, assume_unique, invert);
        } else if (!elements_is_scalar && test_elements_is_scalar) {
            result = torch::isin(elements, test_elements_scalar, assume_unique, invert);
        } else {
            result = torch::isin(elements, test_elements, assume_unique, invert);
        }

        // Verify result properties
        if (!elements_is_scalar) {
            // Result should have same shape as elements
            if (result.sizes() != elements.sizes()) {
                std::cerr << "Result shape mismatch with elements" << std::endl;
            }
        }
        
        // Result should be boolean tensor
        if (result.dtype() != torch::kBool) {
            std::cerr << "Result should be boolean tensor" << std::endl;
        }

        // Test edge cases with empty tensors
        if (offset < Size - 5) {
            torch::Tensor empty_elements = torch::empty({0}, torch::kFloat32);
            torch::Tensor empty_test = torch::empty({0}, torch::kFloat32);
            
            auto empty_result1 = torch::isin(empty_elements, test_elements_is_scalar ? test_elements_scalar : test_elements, assume_unique, invert);
            auto empty_result2 = torch::isin(elements_is_scalar ? elements_scalar : elements, empty_test, assume_unique, invert);
        }

        // Test with different dtypes if we have remaining data
        if (offset < Size - 10) {
            try {
                auto int_elements = elements_is_scalar ? elements_scalar : elements.to(torch::kInt32);
                auto int_test = test_elements_is_scalar ? test_elements_scalar : test_elements.to(torch::kInt32);
                
                if (elements_is_scalar && !test_elements_is_scalar) {
                    torch::isin(elements_scalar, int_test, assume_unique, invert);
                } else if (!elements_is_scalar && test_elements_is_scalar) {
                    torch::isin(int_elements, test_elements_scalar, assume_unique, invert);
                } else {
                    torch::isin(int_elements, int_test, assume_unique, invert);
                }
            } catch (...) {
                // Some dtype conversions might fail, that's okay
            }
        }

        // Test with very large tensors to stress memory
        if (offset < Size - 5) {
            uint8_t size_factor = extractUint8(Data, Size, offset) % 10 + 1;
            try {
                torch::Tensor large_elements = torch::randint(0, 100, {size_factor * 100}, torch::kInt32);
                torch::Tensor large_test = torch::randint(0, 50, {size_factor * 50}, torch::kInt32);
                torch::isin(large_elements, large_test, assume_unique, invert);
            } catch (...) {
                // Memory allocation might fail for very large tensors
            }
        }

        // Test with duplicate values when assume_unique is true
        if (offset < Size - 5) {
            torch::Tensor dup_elements = torch::tensor({1, 2, 2, 3, 3, 3});
            torch::Tensor dup_test = torch::tensor({2, 3, 3, 4});
            torch::isin(dup_elements, dup_test, true, invert); // assume_unique=true with duplicates
        }

        // Access result to ensure computation
        if (result.numel() > 0) {
            result.sum();
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}