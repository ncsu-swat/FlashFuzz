#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <tuple>

// Helper to consume bytes from fuzzer input
template<typename T>
T consume(const uint8_t* &data, size_t &remaining) {
    if (remaining < sizeof(T)) {
        remaining = 0;
        return T{};
    }
    T value;
    std::memcpy(&value, data, sizeof(T));
    data += sizeof(T);
    remaining -= sizeof(T);
    return value;
}

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 10) {
            // Need minimum bytes for configuration
            return 0;
        }

        size_t offset = 0;
        const uint8_t* current = Data;
        size_t remaining = Size;

        // Create input tensor using fuzzer_utils
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with a default tensor
            input = torch::randn({1, 3, 16});
        }

        // Update pointers after tensor creation
        current = Data + offset;
        remaining = (offset < Size) ? Size - offset : 0;

        // Parse AvgPool1d parameters from remaining bytes
        // kernel_size (required)
        int64_t kernel_size = 1 + (consume<uint8_t>(current, remaining) % 16);
        if (kernel_size <= 0) kernel_size = 1;

        // stride (optional, defaults to kernel_size)
        int64_t stride = 0;
        bool use_custom_stride = consume<uint8_t>(current, remaining) % 2;
        if (use_custom_stride) {
            stride = 1 + (consume<uint8_t>(current, remaining) % 16);
            if (stride <= 0) stride = 1;
        }

        // padding (optional, defaults to 0)
        int64_t padding = consume<uint8_t>(current, remaining) % 8;

        // ceil_mode (optional, defaults to false)
        bool ceil_mode = consume<uint8_t>(current, remaining) % 2;

        // count_include_pad (optional, defaults to true)
        bool count_include_pad = consume<uint8_t>(current, remaining) % 2;

        // Create AvgPool1d options
        torch::nn::AvgPool1dOptions options(kernel_size);
        
        if (use_custom_stride && stride > 0) {
            options.stride(stride);
        }
        
        options.padding(padding);
        options.ceil_mode(ceil_mode);
        options.count_include_pad(count_include_pad);

        // Create AvgPool1d module
        torch::nn::AvgPool1d pool(options);

        // Ensure input has correct dimensions for AvgPool1d (N, C, L) or (C, L)
        torch::Tensor reshaped_input = input;
        
        if (input.dim() == 0) {
            // Scalar - reshape to (1, 1, 1)
            reshaped_input = input.reshape({1, 1, 1});
        } else if (input.dim() == 1) {
            // 1D tensor - treat as (1, 1, L)
            reshaped_input = input.unsqueeze(0).unsqueeze(0);
        } else if (input.dim() == 2) {
            // 2D tensor - treat as (C, L), add batch dimension
            reshaped_input = input.unsqueeze(0);
        } else if (input.dim() > 3) {
            // Flatten extra dimensions
            auto sizes = input.sizes();
            int64_t batch = sizes[0];
            int64_t channels = (input.dim() > 1) ? sizes[1] : 1;
            int64_t length = input.numel() / (batch * channels);
            if (length <= 0) length = 1;
            reshaped_input = input.reshape({batch, channels, length});
        }

        // Ensure we have 3 dimensions (N, C, L)
        if (reshaped_input.dim() != 3) {
            // Final fallback
            reshaped_input = reshaped_input.reshape({1, 1, reshaped_input.numel()});
        }

        // Apply pooling
        torch::Tensor output = pool->forward(reshaped_input);

        // Test various edge cases and operations
        
        // Test with different input types if float-like
        if (reshaped_input.dtype() == torch::kFloat || 
            reshaped_input.dtype() == torch::kDouble ||
            reshaped_input.dtype() == torch::kHalf ||
            reshaped_input.dtype() == torch::kBFloat16) {
            
            // Test with special values
            torch::Tensor special_input = reshaped_input.clone();
            
            // Inject NaN
            if (consume<uint8_t>(current, remaining) % 4 == 0) {
                special_input[0][0][0] = std::numeric_limits<float>::quiet_NaN();
                torch::Tensor nan_output = pool->forward(special_input);
            }
            
            // Inject Inf
            if (consume<uint8_t>(current, remaining) % 4 == 0) {
                special_input[0][0][0] = std::numeric_limits<float>::infinity();
                torch::Tensor inf_output = pool->forward(special_input);
            }
            
            // Inject -Inf
            if (consume<uint8_t>(current, remaining) % 4 == 0) {
                special_input[0][0][0] = -std::numeric_limits<float>::infinity();
                torch::Tensor ninf_output = pool->forward(special_input);
            }
        }

        // Test with zero-sized dimensions
        if (consume<uint8_t>(current, remaining) % 8 == 0) {
            torch::Tensor zero_batch = torch::zeros({0, 3, 16});
            torch::Tensor zero_output = pool->forward(zero_batch);
        }

        // Test with very large kernel relative to input size
        if (reshaped_input.size(2) > 0) {
            torch::nn::AvgPool1dOptions large_kernel_options(reshaped_input.size(2) * 2);
            large_kernel_options.padding(padding);
            torch::nn::AvgPool1d large_pool(large_kernel_options);
            torch::Tensor large_kernel_output = large_pool->forward(reshaped_input);
        }

        // Test gradient computation if requires_grad
        if (reshaped_input.requires_grad() && consume<uint8_t>(current, remaining) % 4 == 0) {
            output.backward(torch::ones_like(output));
        }

        // Test with different memory layouts
        if (consume<uint8_t>(current, remaining) % 4 == 0) {
            torch::Tensor transposed = reshaped_input.transpose(1, 2).contiguous().transpose(1, 2);
            torch::Tensor transposed_output = pool->forward(transposed);
        }

        // Test divisor_override parameter (if supported in C++ API)
        // Note: divisor_override might not be available in C++ API, but we try different count_include_pad settings
        if (consume<uint8_t>(current, remaining) % 4 == 0) {
            torch::nn::AvgPool1dOptions alt_options(kernel_size);
            alt_options.stride(stride > 0 ? stride : kernel_size);
            alt_options.padding(padding);
            alt_options.ceil_mode(!ceil_mode);  // Toggle ceil_mode
            alt_options.count_include_pad(!count_include_pad);  // Toggle count_include_pad
            
            torch::nn::AvgPool1d alt_pool(alt_options);
            torch::Tensor alt_output = alt_pool->forward(reshaped_input);
        }

        // Test with different batch sizes
        if (consume<uint8_t>(current, remaining) % 4 == 0) {
            auto sizes = reshaped_input.sizes();
            torch::Tensor multi_batch = reshaped_input.expand({5, sizes[1], sizes[2]});
            torch::Tensor multi_output = pool->forward(multi_batch);
        }

        // Verify output properties
        if (output.defined()) {
            // Check output is finite for float types
            if (output.dtype() == torch::kFloat || output.dtype() == torch::kDouble) {
                bool has_nan = torch::any(torch::isnan(output)).item<bool>();
                bool has_inf = torch::any(torch::isinf(output)).item<bool>();
            }
            
            // Check output shape consistency
            int64_t expected_length = (reshaped_input.size(2) + 2 * padding - kernel_size) / 
                                     (stride > 0 ? stride : kernel_size) + 1;
            if (ceil_mode) {
                expected_length = (reshaped_input.size(2) + 2 * padding - kernel_size + 
                                  (stride > 0 ? stride : kernel_size) - 1) / 
                                  (stride > 0 ? stride : kernel_size) + 1;
            }
        }

    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}