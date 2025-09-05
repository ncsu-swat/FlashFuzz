#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Minimum size check for basic parsing
        if (Size < 20) {
            return 0;
        }

        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data for additional parameters
        if (offset + 16 >= Size) {
            return 0;
        }

        // Create observer_on tensor (for running min/max statistics)
        // Make it a 1D tensor with 2 elements [min, max]
        torch::Tensor observer_on = torch::zeros({2}, input.options());
        if (offset + 2 < Size) {
            float min_val, max_val;
            std::memcpy(&min_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&max_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure min <= max
            if (min_val > max_val) {
                std::swap(min_val, max_val);
            }
            observer_on[0] = min_val;
            observer_on[1] = max_val;
        }

        // Parse fake quant parameters
        int quant_min = -128;
        int quant_max = 127;
        if (offset + 2 < Size) {
            quant_min = static_cast<int>(Data[offset++]);
            quant_max = static_cast<int>(Data[offset++]);
            
            // Ensure valid quantization range
            if (quant_min >= quant_max) {
                quant_max = quant_min + 1;
            }
            // Clamp to reasonable values
            quant_min = std::max(-256, std::min(255, quant_min));
            quant_max = std::max(quant_min + 1, std::min(256, quant_max));
        }

        // Parse averaging constant (momentum for moving average)
        double averaging_const = 0.01;
        if (offset + sizeof(float) <= Size) {
            float tmp;
            std::memcpy(&tmp, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp to [0, 1] range
            averaging_const = std::max(0.0, std::min(1.0, static_cast<double>(std::abs(tmp))));
        }

        // Parse channel axis (for per-channel quantization)
        int64_t ch_axis = 0;
        if (offset < Size) {
            ch_axis = static_cast<int64_t>(Data[offset++] % std::max(1L, static_cast<long>(input.dim())));
        }

        // Parse per_row_fake_quant flag
        bool per_row_fake_quant = false;
        if (offset < Size) {
            per_row_fake_quant = (Data[offset++] & 1) != 0;
        }

        // Parse symmetric_quant flag
        bool symmetric_quant = false;
        if (offset < Size) {
            symmetric_quant = (Data[offset++] & 1) != 0;
        }

        // Create scale and zero_point tensors based on quantization type
        torch::Tensor scale, zero_point;
        
        if (per_row_fake_quant && input.dim() > 0) {
            // Per-channel quantization
            int64_t num_channels = (ch_axis < input.dim()) ? input.size(ch_axis) : 1;
            scale = torch::ones({num_channels}, torch::kFloat32);
            zero_point = torch::zeros({num_channels}, torch::kInt32);
            
            // Optionally populate with fuzzed values
            if (offset + num_channels * sizeof(float) <= Size) {
                for (int64_t i = 0; i < num_channels && offset < Size; ++i) {
                    float s;
                    std::memcpy(&s, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    scale[i] = std::abs(s) + 0.001f; // Ensure positive scale
                }
            }
        } else {
            // Per-tensor quantization
            scale = torch::tensor(1.0f);
            zero_point = torch::tensor(0, torch::kInt32);
            
            if (offset + sizeof(float) <= Size) {
                float s;
                std::memcpy(&s, Data + offset, sizeof(float));
                offset += sizeof(float);
                scale = torch::tensor(std::abs(s) + 0.001f);
            }
        }

        // Create fake_quant_on tensor (enable/disable flag)
        torch::Tensor fake_quant_on = torch::ones({1}, torch::kInt32);
        if (offset < Size) {
            fake_quant_on = torch::tensor({static_cast<int>(Data[offset++] & 1)}, torch::kInt32);
        }

        // Create running_min and running_max tensors for observer state
        torch::Tensor running_min, running_max;
        if (per_row_fake_quant && input.dim() > 0) {
            int64_t num_channels = (ch_axis < input.dim()) ? input.size(ch_axis) : 1;
            running_min = torch::full({num_channels}, std::numeric_limits<float>::max());
            running_max = torch::full({num_channels}, std::numeric_limits<float>::lowest());
        } else {
            running_min = torch::tensor(std::numeric_limits<float>::max());
            running_max = torch::tensor(std::numeric_limits<float>::lowest());
        }

        // Try different tensor layouts/strides
        if (offset < Size && (Data[offset++] & 1) && input.dim() >= 2) {
            input = input.transpose(0, 1);
        }

        // Make tensor non-contiguous in some cases
        if (offset < Size && (Data[offset++] & 1) && input.numel() > 1) {
            auto sizes = input.sizes().vec();
            auto strides = input.strides().vec();
            if (!strides.empty()) {
                strides[0] *= 2;
                input = input.as_strided(sizes, strides);
            }
        }

        // Call the fused_moving_avg_obs_fake_quant operation
        // Note: The actual C++ API might differ slightly from Python
        // This attempts to match the expected signature based on PyTorch quantization ops
        
        // The operation typically returns a tuple of (output, mask)
        // where mask indicates which values were clamped
        auto result = torch::fused_moving_avg_obs_fake_quant(
            input,
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

        // Access the result to ensure computation happens
        if (result.defined()) {
            auto output_tensor = result;
            
            // Trigger computation
            if (output_tensor.numel() > 0) {
                output_tensor.sum().item<float>();
            }
            
            // Additional operations to increase coverage
            if (offset < Size && (Data[offset++] & 1)) {
                output_tensor.mean();
            }
            if (offset < Size && (Data[offset++] & 1)) {
                output_tensor.std();
            }
        }

        // Try with different tensor types if we have more data
        if (offset + 10 < Size) {
            // Try with different dtype
            auto dtype_selector = Data[offset++];
            if (dtype_selector % 4 == 0) {
                input = input.to(torch::kFloat64);
            } else if (dtype_selector % 4 == 1) {
                input = input.to(torch::kFloat16);
            } else if (dtype_selector % 4 == 2 && torch::cuda::is_available()) {
                input = input.cuda();
            }
            
            // Call again with modified input
            try {
                auto result2 = torch::fused_moving_avg_obs_fake_quant(
                    input,
                    observer_on.to(input.device()),
                    fake_quant_on.to(input.device()),
                    running_min.to(input.device()),
                    running_max.to(input.device()),
                    scale.to(input.device()),
                    zero_point.to(input.device()),
                    averaging_const,
                    quant_min,
                    quant_max,
                    ch_axis,
                    per_row_fake_quant,
                    symmetric_quant
                );
                
                if (result2.defined() && result2.numel() > 0) {
                    result2.sum().item<float>();
                }
            } catch (const c10::Error& e) {
                // Silently ignore C10 errors for edge cases
            }
        }

    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors - these are expected for invalid inputs
        return 0;
    }
    catch (const std::bad_alloc &e)
    {
        // Memory allocation failures - expected for large tensors
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}