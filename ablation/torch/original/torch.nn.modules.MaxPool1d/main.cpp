#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>

// Helper function to safely extract values from fuzzer data
template<typename T>
T extractValue(const uint8_t* data, size_t& offset, size_t size, T min_val, T max_val) {
    if (offset + sizeof(T) > size) {
        offset = size;
        return min_val;
    }
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    
    // Ensure value is within bounds
    if (value < min_val) value = min_val;
    if (value > max_val) value = max_val;
    return value;
}

// Helper to extract a bounded integer from a single byte
int64_t extractBoundedInt(const uint8_t* data, size_t& offset, size_t size, 
                          int64_t min_val, int64_t max_val) {
    if (offset >= size) return min_val;
    uint8_t byte = data[offset++];
    int64_t range = max_val - min_val + 1;
    return min_val + (byte % range);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Minimum size check - we need at least a few bytes for parameters and tensor
        if (Size < 10) {
            return 0;
        }

        // Extract MaxPool1d parameters from fuzzer input
        // kernel_size: must be > 0, typically 1-10
        int64_t kernel_size = extractBoundedInt(Data, offset, Size, 1, 10);
        
        // stride: must be > 0, default is kernel_size if not specified
        // Use a flag to determine if we should use default or custom stride
        bool use_default_stride = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        int64_t stride = use_default_stride ? kernel_size : extractBoundedInt(Data, offset, Size, 1, 10);
        
        // padding: must be >= 0 and <= kernel_size / 2
        int64_t max_padding = kernel_size / 2;
        int64_t padding = extractBoundedInt(Data, offset, Size, 0, max_padding);
        
        // dilation: must be > 0, typically 1-5
        int64_t dilation = extractBoundedInt(Data, offset, Size, 1, 5);
        
        // return_indices: boolean flag
        bool return_indices = (offset < Size) ? (Data[offset++] % 2 == 1) : false;
        
        // ceil_mode: boolean flag
        bool ceil_mode = (offset < Size) ? (Data[offset++] % 2 == 1) : false;

#ifdef DEBUG_FUZZ
        std::cout << "MaxPool1d parameters:" << std::endl;
        std::cout << "  kernel_size: " << kernel_size << std::endl;
        std::cout << "  stride: " << stride << std::endl;
        std::cout << "  padding: " << padding << std::endl;
        std::cout << "  dilation: " << dilation << std::endl;
        std::cout << "  return_indices: " << return_indices << std::endl;
        std::cout << "  ceil_mode: " << ceil_mode << std::endl;
#endif

        // Create MaxPool1d module with options
        torch::nn::MaxPool1dOptions pool_options(kernel_size);
        pool_options.stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .ceil_mode(ceil_mode);
        
        torch::nn::MaxPool1d pool_module(pool_options);

        // Create input tensor from remaining fuzzer data
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, create a small random tensor
            input = torch::randn({2, 3, 10});
        }

        // Ensure input tensor has appropriate dimensions for MaxPool1d
        // MaxPool1d expects either (N, C, L) or (C, L)
        if (input.dim() < 2 || input.dim() > 3) {
            // Reshape to valid dimensions
            int64_t total_elements = input.numel();
            if (total_elements == 0) {
                input = torch::randn({2, 3, 10});
            } else if (input.dim() == 1) {
                // Make it 2D: (C=1, L)
                input = input.unsqueeze(0);
            } else if (input.dim() > 3) {
                // Flatten extra dimensions
                auto sizes = input.sizes();
                int64_t batch = (input.dim() > 0) ? sizes[0] : 1;
                int64_t channels = (input.dim() > 1) ? sizes[1] : 1;
                int64_t length = total_elements / (batch * channels);
                if (length < 1) length = 1;
                input = input.reshape({batch, channels, length});
            }
        }

        // Ensure the last dimension is large enough for the pooling operation
        if (input.dim() == 2) {
            // Shape is (C, L)
            int64_t min_length = dilation * (kernel_size - 1) + 1;
            if (input.size(-1) < min_length) {
                // Pad or recreate tensor with minimum required size
                input = torch::randn({input.size(0), min_length + padding * 2});
            }
        } else if (input.dim() == 3) {
            // Shape is (N, C, L)
            int64_t min_length = dilation * (kernel_size - 1) + 1;
            if (input.size(-1) < min_length) {
                input = torch::randn({input.size(0), input.size(1), min_length + padding * 2});
            }
        }

#ifdef DEBUG_FUZZ
        std::cout << "Input tensor shape: " << input.sizes() << std::endl;
        std::cout << "Input tensor dtype: " << input.dtype() << std::endl;
#endif

        // Apply MaxPool1d
        if (return_indices) {
            // When return_indices is true, forward returns a tuple
            auto result = pool_module->forward_with_indices(input);
            torch::Tensor output = std::get<0>(result);
            torch::Tensor indices = std::get<1>(result);
            
#ifdef DEBUG_FUZZ
            std::cout << "Output shape: " << output.sizes() << std::endl;
            std::cout << "Indices shape: " << indices.sizes() << std::endl;
#endif
            
            // Perform some basic validation
            if (output.dim() != input.dim()) {
                std::cerr << "Warning: Output dimension mismatch" << std::endl;
            }
            
            // Check that indices are within valid range
            if (indices.numel() > 0) {
                auto max_idx = indices.max().item<int64_t>();
                auto min_idx = indices.min().item<int64_t>();
                if (min_idx < 0 || max_idx >= input.size(-1)) {
                    std::cerr << "Warning: Indices out of range" << std::endl;
                }
            }
        } else {
            // Regular forward without indices
            torch::Tensor output = pool_module->forward(input);
            
#ifdef DEBUG_FUZZ
            std::cout << "Output shape: " << output.sizes() << std::endl;
#endif
            
            // Validate output
            if (output.dim() != input.dim()) {
                std::cerr << "Warning: Output dimension mismatch" << std::endl;
            }
            
            // Test with different input variations
            // Test with requires_grad
            if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
                input.requires_grad_(true);
                torch::Tensor grad_output = pool_module->forward(input);
                
                // Compute gradients if possible
                if (grad_output.numel() > 0) {
                    auto sum = grad_output.sum();
                    sum.backward();
                    
                    if (input.grad().defined()) {
#ifdef DEBUG_FUZZ
                        std::cout << "Gradient computed successfully" << std::endl;
#endif
                    }
                }
            }
        }
        
        // Test edge cases
        // Test with different memory layouts (non-contiguous tensors)
        if (input.is_contiguous() && input.numel() > 1) {
            torch::Tensor transposed = input.transpose(-1, -2);
            if (transposed.dim() == 3 && transposed.size(-1) >= dilation * (kernel_size - 1) + 1) {
                try {
                    torch::Tensor output_transposed = pool_module->forward(transposed.contiguous());
#ifdef DEBUG_FUZZ
                    std::cout << "Non-contiguous test passed" << std::endl;
#endif
                } catch (...) {
                    // Silently ignore if this specific test fails
                }
            }
        }
        
        // Test with zero-sized dimensions (if applicable)
        if (input.dim() == 3 && input.size(0) > 0) {
            torch::Tensor zero_batch = torch::empty({0, input.size(1), input.size(2)}, input.options());
            try {
                torch::Tensor zero_output = pool_module->forward(zero_batch);
#ifdef DEBUG_FUZZ
                std::cout << "Zero-batch test passed, output shape: " << zero_output.sizes() << std::endl;
#endif
            } catch (...) {
                // Expected to potentially fail with zero batch
            }
        }
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
#ifdef DEBUG_FUZZ
        std::cout << "PyTorch error caught: " << e.what() << std::endl;
#endif
        return 0; // Continue fuzzing
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // Discard this input
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1; // Discard this input
    }
    
    return 0; // Keep the input
}