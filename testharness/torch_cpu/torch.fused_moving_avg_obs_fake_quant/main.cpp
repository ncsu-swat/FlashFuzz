#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For std::memcpy
#include <cmath>          // For std::abs, std::floor

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
        
        // Early exit if not enough data
        if (Size < 20) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor X = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Need more data for parameters
        if (offset + 16 > Size) {
            return 0;
        }
        
        // Create observer_on tensor (scalar tensor with 0 or 1)
        torch::Tensor observer_on = torch::tensor(1);
        
        // Create fake_quant_on tensor (scalar tensor with 0 or 1)
        torch::Tensor fake_quant_on = torch::tensor(1);
        
        // Create running_min tensor (mutable)
        torch::Tensor running_min = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        // Create running_max tensor (mutable)
        torch::Tensor running_max = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        // Create scale tensor (mutable)
        torch::Tensor scale = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        // Create zero_point tensor (mutable)
        torch::Tensor zero_point = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract averaging constant
        double averaging_const = 0.01;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&averaging_const, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure averaging_const is valid (between 0 and 1)
            if (!std::isfinite(averaging_const)) {
                averaging_const = 0.01;
            } else {
                averaging_const = std::abs(averaging_const);
                if (averaging_const > 1.0) {
                    averaging_const = averaging_const - std::floor(averaging_const);
                }
                if (averaging_const == 0.0) {
                    averaging_const = 0.01;
                }
            }
        }
        
        // Extract quant_min and quant_max
        int64_t quant_min = 0;
        int64_t quant_max = 255;
        if (offset + 2 * sizeof(int64_t) <= Size) {
            std::memcpy(&quant_min, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&quant_max, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Normalize to reasonable quantization ranges
            quant_min = quant_min % 256;
            quant_max = quant_max % 256;
            
            // Ensure quant_max > quant_min
            if (quant_max <= quant_min) {
                quant_max = quant_min + 1;
            }
        }
        
        // Extract ch_axis (channel axis for per-channel quantization)
        int64_t ch_axis = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&ch_axis, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Normalize to valid axis range (-4 to 3 typically)
            ch_axis = ch_axis % 4;
        }
        
        // Extract per_row_fake_quant and symmetric_quant flags
        bool per_row_fake_quant = false;
        bool symmetric_quant = false;
        if (offset < Size) {
            uint8_t flags = Data[offset++];
            per_row_fake_quant = (flags & 0x01) != 0;
            symmetric_quant = (flags & 0x02) != 0;
        }
        
        // Ensure running_min <= running_max
        try {
            auto min_val = torch::min(running_min, running_max);
            auto max_val = torch::max(running_min, running_max);
            running_min = min_val.clone();
            running_max = max_val.clone();
        } catch (...) {
            // Shape mismatch, use defaults
            running_min = torch::tensor(-1.0f);
            running_max = torch::tensor(1.0f);
        }
        
        // Ensure scale is positive
        try {
            scale = torch::abs(scale) + 0.001f;
        } catch (...) {
            scale = torch::tensor(0.1f);
        }
        
        // Ensure zero_point is an integer tensor
        try {
            zero_point = zero_point.to(torch::kInt32);
        } catch (...) {
            zero_point = torch::tensor(0, torch::kInt32);
        }
        
        // Call fused_moving_avg_obs_fake_quant with the correct signature
        // Signature: (self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, 
        //             averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant)
        try {
            torch::Tensor output = torch::fused_moving_avg_obs_fake_quant(
                X,
                observer_on,
                fake_quant_on,
                running_min,
                running_max,
                scale,
                zero_point,
                averaging_const,
                quant_min,
                quant_max,
                ch_axis,
                per_row_fake_quant,
                symmetric_quant
            );
            
            // Ensure the output is used to prevent optimization
            if (output.defined() && output.numel() > 0) {
                volatile float sum = output.sum().item<float>();
                (void)sum;
            }
        } catch (...) {
            // Expected failures for invalid parameter combinations
        }
        
        // Test with observer and fake_quant off
        try {
            torch::Tensor observer_off = torch::tensor(0);
            torch::Tensor fake_quant_off = torch::tensor(0);
            
            // Need fresh copies since the tensors are mutable
            torch::Tensor running_min_copy = running_min.clone();
            torch::Tensor running_max_copy = running_max.clone();
            torch::Tensor scale_copy = scale.clone();
            torch::Tensor zero_point_copy = zero_point.clone();
            
            torch::Tensor output2 = torch::fused_moving_avg_obs_fake_quant(
                X,
                observer_off,
                fake_quant_off,
                running_min_copy,
                running_max_copy,
                scale_copy,
                zero_point_copy,
                averaging_const,
                quant_min,
                quant_max,
                ch_axis,
                !per_row_fake_quant,
                !symmetric_quant
            );
            
            if (output2.defined() && output2.numel() > 0) {
                volatile float sum = output2.sum().item<float>();
                (void)sum;
            }
        } catch (...) {
            // Expected failures
        }
        
        // Test with standard int8 quantization ranges
        try {
            torch::Tensor running_min_copy = running_min.clone();
            torch::Tensor running_max_copy = running_max.clone();
            torch::Tensor scale_copy = scale.clone();
            torch::Tensor zero_point_copy = zero_point.clone();
            
            torch::Tensor output3 = torch::fused_moving_avg_obs_fake_quant(
                X,
                observer_on,
                fake_quant_on,
                running_min_copy,
                running_max_copy,
                scale_copy,
                zero_point_copy,
                0.1,  // standard averaging constant
                -128, // int8 min
                127,  // int8 max
                0,    // ch_axis = 0
                true,
                true
            );
            
            if (output3.defined() && output3.numel() > 0) {
                volatile float sum = output3.sum().item<float>();
                (void)sum;
            }
        } catch (...) {
            // Expected failures
        }
        
        // Test with uint8 quantization range
        try {
            torch::Tensor running_min_copy = running_min.clone();
            torch::Tensor running_max_copy = running_max.clone();
            torch::Tensor scale_copy = scale.clone();
            torch::Tensor zero_point_copy = zero_point.clone();
            
            torch::Tensor output4 = torch::fused_moving_avg_obs_fake_quant(
                X,
                observer_on,
                fake_quant_on,
                running_min_copy,
                running_max_copy,
                scale_copy,
                zero_point_copy,
                0.01,
                0,    // uint8 min
                255,  // uint8 max
                1,    // ch_axis = 1
                false,
                false
            );
            
            if (output4.defined() && output4.numel() > 0) {
                volatile float sum = output4.sum().item<float>();
                (void)sum;
            }
        } catch (...) {
            // Expected failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}