#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor for subtraction
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor for subtraction if we have more data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, use a scalar value for subtraction
            uint8_t scalar_value = 1;
            if (Size > 0) {
                scalar_value = Data[0]; // Use first byte as scalar
            }
            tensor2 = torch::tensor(scalar_value, tensor1.options());
        }
        
        // Try different variants of subtraction
        try {
            // Basic subtraction
            torch::Tensor result1 = torch::subtract(tensor1, tensor2);
        } catch (const std::exception&) {
            // Continue with other tests even if this fails
        }
        
        try {
            // Subtraction with alpha parameter
            double alpha = 1.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&alpha, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            torch::Tensor result2 = torch::subtract(tensor1, tensor2, alpha);
        } catch (const std::exception&) {
            // Continue with other tests even if this fails
        }
        
        try {
            // Inplace subtraction
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.subtract_(tensor2);
        } catch (const std::exception&) {
            // Continue with other tests even if this fails
        }
        
        try {
            // Operator-based subtraction
            torch::Tensor result3 = tensor1 - tensor2;
        } catch (const std::exception&) {
            // Continue with other tests even if this fails
        }
        
        try {
            // Scalar subtraction
            double scalar = 0.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            torch::Tensor result4 = torch::subtract(tensor1, scalar);
        } catch (const std::exception&) {
            // Continue with other tests even if this fails
        }
        
        // Test out_variant
        try {
            torch::Tensor out = torch::empty_like(tensor1);
            torch::subtract_out(out, tensor1, tensor2);
        } catch (const std::exception&) {
            // Continue with other tests even if this fails
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}