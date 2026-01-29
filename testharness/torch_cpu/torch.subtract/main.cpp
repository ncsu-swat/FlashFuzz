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
            tensor2 = torch::tensor(static_cast<float>(scalar_value), tensor1.options());
        }
        
        // Try different variants of subtraction
        try {
            // Basic subtraction using torch::sub (subtract is an alias)
            torch::Tensor result1 = torch::sub(tensor1, tensor2);
            (void)result1;
        } catch (const std::exception&) {
            // Continue with other tests even if this fails (shape mismatch, etc.)
        }
        
        try {
            // Subtraction with alpha parameter: result = input - alpha * other
            double alpha = 1.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&alpha, Data + offset, sizeof(double));
                offset += sizeof(double);
                // Clamp alpha to reasonable range to avoid inf/nan issues
                if (std::isnan(alpha) || std::isinf(alpha)) {
                    alpha = 1.0;
                }
            }
            torch::Tensor result2 = torch::sub(tensor1, tensor2, torch::Scalar(alpha));
            (void)result2;
        } catch (const std::exception&) {
            // Continue with other tests even if this fails
        }
        
        try {
            // Inplace subtraction
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.sub_(tensor2);
        } catch (const std::exception&) {
            // Continue with other tests even if this fails
        }
        
        try {
            // Operator-based subtraction
            torch::Tensor result3 = tensor1 - tensor2;
            (void)result3;
        } catch (const std::exception&) {
            // Continue with other tests even if this fails
        }
        
        try {
            // Scalar subtraction
            float scalar = 0.0f;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&scalar, Data + offset, sizeof(float));
                offset += sizeof(float);
                // Handle nan/inf
                if (std::isnan(scalar) || std::isinf(scalar)) {
                    scalar = 1.0f;
                }
            }
            torch::Tensor result4 = torch::sub(tensor1, torch::Scalar(scalar));
            (void)result4;
        } catch (const std::exception&) {
            // Continue with other tests even if this fails
        }
        
        // Test out_variant using torch::sub_out
        try {
            torch::Tensor out = torch::empty_like(tensor1);
            torch::sub_out(out, tensor1, tensor2);
        } catch (const std::exception&) {
            // Continue with other tests even if this fails
        }
        
        // Test inplace with alpha
        try {
            torch::Tensor tensor_copy = tensor1.clone();
            double alpha = 2.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&alpha, Data + offset, sizeof(double));
                if (std::isnan(alpha) || std::isinf(alpha)) {
                    alpha = 2.0;
                }
            }
            tensor_copy.sub_(tensor2, torch::Scalar(alpha));
        } catch (const std::exception&) {
            // Continue with other tests even if this fails
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}