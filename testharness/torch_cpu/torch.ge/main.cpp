#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor if we have more data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, use a scalar tensor for comparison
            uint8_t scalar_value = Size > 0 ? Data[Size - 1] : 0;
            tensor2 = torch::tensor(static_cast<float>(scalar_value), tensor1.options());
        }
        
        // Extract scalar value from remaining data
        double scalar_val = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scalar_val, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize scalar to avoid NaN/Inf issues
            if (std::isnan(scalar_val) || std::isinf(scalar_val)) {
                scalar_val = 0.0;
            }
        }
        
        // 1. Element-wise comparison (tensor >= tensor) with broadcasting handling
        try {
            torch::Tensor result1 = torch::ge(tensor1, tensor2);
        } catch (const std::exception &) {
            // Broadcasting may fail for incompatible shapes - expected
        }
        
        // 2. Tensor >= Scalar (always works)
        torch::Tensor result2 = torch::ge(tensor1, scalar_val);
        
        // 3. Create scalar tensor for scalar >= tensor comparison
        torch::Tensor scalar_tensor = torch::tensor(scalar_val, tensor1.options());
        torch::Tensor result3 = torch::ge(scalar_tensor, tensor1);
        
        // 4. In-place version with scalar (always works)
        torch::Tensor tensor_copy = tensor1.clone();
        tensor_copy.ge_(scalar_val);
        
        // 5. In-place version (tensor >= tensor) - may fail with broadcasting
        try {
            torch::Tensor tensor_copy2 = tensor1.clone();
            tensor_copy2.ge_(tensor2);
        } catch (const std::exception &) {
            // In-place broadcasting restrictions - expected
        }
        
        // 6. Operator overload versions
        try {
            torch::Tensor result4 = tensor1 >= tensor2;
        } catch (const std::exception &) {
            // Broadcasting may fail - expected
        }
        torch::Tensor result5 = tensor1 >= scalar_val;
        torch::Tensor result6 = scalar_tensor >= tensor1;
        
        // 7. Test with different output types
        torch::Tensor result7 = torch::ge(tensor1, scalar_val).to(torch::kFloat32);
        
        // 8. Test with empty tensors
        try {
            torch::Tensor empty1 = torch::empty({0}, tensor1.options());
            torch::Tensor empty2 = torch::empty({0}, tensor1.options());
            torch::Tensor empty_result = torch::ge(empty1, empty2);
        } catch (const std::exception &) {
            // Empty tensor handling may vary - expected
        }
        
        // 9. Test ge_out with same-shape tensors to avoid broadcast shape calculation
        try {
            // Create output tensor matching tensor1's shape (for tensor vs scalar comparison)
            torch::Tensor out_tensor = torch::empty(tensor1.sizes(), torch::TensorOptions().dtype(torch::kBool));
            torch::ge_out(out_tensor, tensor1, scalar_tensor);
        } catch (const std::exception &) {
            // Shape mismatch or other failure - expected
        }
        
        // 10. Test ge_out with two tensors of same shape
        try {
            // Create tensor2_same with same shape as tensor1 for ge_out test
            torch::Tensor tensor2_same = torch::rand(tensor1.sizes(), tensor1.options());
            torch::Tensor out_tensor2 = torch::empty(tensor1.sizes(), torch::TensorOptions().dtype(torch::kBool));
            torch::ge_out(out_tensor2, tensor1, tensor2_same);
        } catch (const std::exception &) {
            // May fail - expected
        }
        
        // 11. Test with 0-dimensional (scalar) tensors
        try {
            torch::Tensor scalar_t1 = torch::tensor(1.5);
            torch::Tensor scalar_t2 = torch::tensor(2.5);
            torch::Tensor scalar_result = torch::ge(scalar_t1, scalar_t2);
        } catch (const std::exception &) {
            // Scalar tensor comparison - expected to work but catch just in case
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}