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
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip complex tensors as less_equal doesn't support them
        if (tensor1.is_complex()) {
            tensor1 = torch::real(tensor1);
        }
        
        // Create second tensor if we have more data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (tensor2.is_complex()) {
                tensor2 = torch::real(tensor2);
            }
        } else {
            tensor2 = tensor1.clone();
            if (tensor2.numel() > 0) {
                if (tensor2.is_floating_point()) {
                    tensor2 = tensor2 + 0.5;
                } else {
                    tensor2 = tensor2 + 1;
                }
            }
        }
        
        // Test 1: Basic less_equal with two tensors (with broadcasting)
        try {
            torch::Tensor result = torch::less_equal(tensor1, tensor2);
            // Verify result is boolean
            if (result.numel() > 0) {
                (void)result.scalar_type();
            }
        } catch (...) {
            // Ignore broadcasting/shape errors
        }
        
        // Test 2: Tensor-scalar comparison
        if (tensor1.numel() > 0) {
            // Use different scalar values
            try {
                torch::Tensor result = torch::less_equal(tensor1, 0.0);
            } catch (...) {
            }
            
            try {
                torch::Tensor result = torch::less_equal(tensor1, 1);
            } catch (...) {
            }
            
            try {
                torch::Tensor result = torch::less_equal(tensor1, -1.5);
            } catch (...) {
            }
        }
        
        // Test 3: Use le (alias for less_equal)
        try {
            torch::Tensor result = torch::le(tensor1, tensor2);
        } catch (...) {
        }
        
        // Test 4: Test with scalar from data if we have single-element tensor
        if (tensor2.numel() == 1) {
            try {
                torch::Scalar scalar_value;
                if (tensor2.is_floating_point()) {
                    scalar_value = tensor2.item<double>();
                } else {
                    scalar_value = tensor2.item<int64_t>();
                }
                torch::Tensor result = torch::less_equal(tensor1, scalar_value);
            } catch (...) {
            }
        }
        
        // Test 5: Test with empty tensors
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor result = torch::less_equal(empty_tensor, empty_tensor);
        } catch (...) {
        }
        
        // Test 6: Test with different dtypes
        if (tensor1.numel() > 0 && tensor2.numel() > 0) {
            try {
                torch::Tensor t1_float = tensor1.to(torch::kFloat);
                torch::Tensor t2_int = tensor2.to(torch::kInt64);
                torch::Tensor result = torch::less_equal(t1_float, t2_int);
            } catch (...) {
            }
        }
        
        // Test 7: Test inplace variant if available (le_)
        try {
            torch::Tensor tensor1_copy = tensor1.clone();
            tensor1_copy.le_(tensor2);
        } catch (...) {
        }
        
        // Test 8: Output tensor variant
        try {
            torch::Tensor out = torch::empty_like(tensor1, torch::kBool);
            torch::less_equal_out(out, tensor1, tensor2);
        } catch (...) {
        }
        
        // Test 9: Test with contiguous and non-contiguous tensors
        if (tensor1.dim() >= 2 && tensor1.size(0) > 1 && tensor1.size(1) > 1) {
            try {
                torch::Tensor non_contig = tensor1.transpose(0, 1);
                torch::Tensor result = torch::less_equal(non_contig, tensor2);
            } catch (...) {
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}