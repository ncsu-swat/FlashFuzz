#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout
#include <cmath>          // For std::abs, std::isfinite

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
        
        // Need at least a few bytes for tensor creation and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if not already
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Get alpha parameter for ELU from the remaining data
        double alpha = 1.0;
        if (offset + sizeof(float) <= Size) {
            float alpha_f;
            std::memcpy(&alpha_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(alpha_f)) {
                alpha = static_cast<double>(alpha_f);
                // Clamp alpha to reasonable range
                alpha = std::max(-10.0, std::min(10.0, alpha));
            }
        }
        
        // Get scale for quantization
        double scale = 0.1;
        if (offset + sizeof(float) <= Size) {
            float scale_f;
            std::memcpy(&scale_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(scale_f) && scale_f != 0.0f) {
                scale = static_cast<double>(std::abs(scale_f));
                // Clamp scale to reasonable range
                scale = std::max(1e-6, std::min(1e6, scale));
            }
        }
        
        // Get zero_point for quantization
        int64_t zero_point = 128;
        if (offset + sizeof(uint8_t) <= Size) {
            zero_point = static_cast<int64_t>(Data[offset]);
            offset += sizeof(uint8_t);
        }
        
        // Get output scale and zero_point
        double output_scale = scale;
        if (offset + sizeof(float) <= Size) {
            float oscale_f;
            std::memcpy(&oscale_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(oscale_f) && oscale_f != 0.0f) {
                output_scale = static_cast<double>(std::abs(oscale_f));
                output_scale = std::max(1e-6, std::min(1e6, output_scale));
            }
        }
        
        int64_t output_zero_point = 128;
        if (offset + sizeof(uint8_t) <= Size) {
            output_zero_point = static_cast<int64_t>(Data[offset]);
            offset += sizeof(uint8_t);
        }
        
        // Quantize the input tensor
        torch::Tensor quantized_input;
        try {
            quantized_input = torch::quantize_per_tensor(
                input_tensor, scale, zero_point, torch::kQUInt8);
        } catch (...) {
            // If quantization fails with these parameters, return early
            return 0;
        }
        
        // Apply quantized ELU using at::quantized_elu
        // The C++ API for quantized ELU takes: input, scale, zero_point, alpha
        try {
            auto output = at::quantized_elu(
                quantized_input,
                output_scale,
                output_zero_point,
                alpha
            );
            
            // Verify output is valid by dequantizing
            auto dequantized = output.dequantize();
            
            // Additional verification: check output shape matches input
            if (output.sizes() != quantized_input.sizes()) {
                std::cerr << "Shape mismatch in quantized ELU output" << std::endl;
            }
        } catch (...) {
            // Expected failures for certain parameter combinations
            // Try with default alpha
            try {
                auto output = at::quantized_elu(
                    quantized_input,
                    output_scale,
                    output_zero_point,
                    1.0  // default alpha
                );
                auto dequantized = output.dequantize();
            } catch (...) {
                // Still failed, silently ignore
            }
        }
        
        // Also test the non-quantized ELU path for comparison
        try {
            auto float_output = torch::elu(input_tensor, alpha);
        } catch (...) {
            // Silently ignore failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}