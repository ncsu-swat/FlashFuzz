#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For isnan, isinf

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
        
        // Need at least some bytes for tensor creation
        if (Size < 6)
            return 0;
            
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create tensor1 with same shape as input for guaranteed compatibility
        torch::Tensor tensor1;
        if (offset < Size) {
            tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure broadcastable by reshaping to input's shape if needed
            try {
                tensor1 = tensor1.expand_as(input).clone();
            } catch (...) {
                tensor1 = torch::ones_like(input);
            }
        } else {
            tensor1 = torch::ones_like(input);
        }
        
        // Create tensor2 with same shape as input for guaranteed compatibility
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure broadcastable by reshaping to input's shape if needed
            try {
                tensor2 = tensor2.expand_as(input).clone();
            } catch (...) {
                tensor2 = torch::ones_like(input);
            }
        } else {
            tensor2 = torch::ones_like(input);
        }
        
        // Parse value for alpha, with sanitization
        double alpha = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize alpha to avoid NaN/Inf issues
            if (std::isnan(alpha) || std::isinf(alpha)) {
                alpha = 1.0;
            }
            // Clamp to reasonable range
            alpha = std::max(-1e10, std::min(1e10, alpha));
        }
        
        // Ensure all tensors are floating point for addcmul
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        if (!tensor1.is_floating_point()) {
            tensor1 = tensor1.to(torch::kFloat32);
        }
        if (!tensor2.is_floating_point()) {
            tensor2 = tensor2.to(torch::kFloat32);
        }
        
        // Apply addcmul operation: out = input + alpha * tensor1 * tensor2
        torch::Tensor result = torch::addcmul(input, tensor1, tensor2, alpha);
        
        // Try in-place version
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.addcmul_(tensor1, tensor2, alpha);
        } catch (...) {
            // In-place may fail due to dtype/shape requirements
        }
        
        // Try the out= variant
        try {
            torch::Tensor output = torch::empty_like(input);
            torch::addcmul_out(output, input, tensor1, tensor2, alpha);
        } catch (...) {
            // May fail for some configurations
        }
        
        // Try with different alpha values from remaining data
        if (offset + sizeof(float) <= Size) {
            float alpha2_f;
            std::memcpy(&alpha2_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (!std::isnan(alpha2_f) && !std::isinf(alpha2_f)) {
                double alpha2 = static_cast<double>(alpha2_f);
                alpha2 = std::max(-1e6, std::min(1e6, alpha2));
                torch::Tensor result2 = torch::addcmul(input, tensor1, tensor2, alpha2);
            }
        }
        
        // Try with Scalar value parameter
        if (offset < Size) {
            int8_t scalar_val = static_cast<int8_t>(Data[offset]);
            offset++;
            torch::Scalar value_scalar(static_cast<double>(scalar_val));
            torch::Tensor result3 = torch::addcmul(input, tensor1, tensor2, value_scalar);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}