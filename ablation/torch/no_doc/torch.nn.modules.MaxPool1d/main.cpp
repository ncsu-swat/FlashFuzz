#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <vector>

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

        // Create input tensor with diverse properties
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try a simpler approach
            uint8_t rank = consume<uint8_t>(current, remaining) % 3 + 1; // 1-3 dimensions
            std::vector<int64_t> shape;
            
            if (rank == 1) {
                shape = {1 + (consume<uint8_t>(current, remaining) % 128)};
            } else if (rank == 2) {
                shape = {1 + (consume<uint8_t>(current, remaining) % 32),
                        1 + (consume<uint8_t>(current, remaining) % 128)};
            } else {
                shape = {1 + (consume<uint8_t>(current, remaining) % 16),
                        1 + (consume<uint8_t>(current, remaining) % 32),
                        1 + (consume<uint8_t>(current, remaining) % 128)};
            }
            
            input = torch::randn(shape);
        }

        // Update pointers after tensor creation
        current = Data + offset;
        remaining = (offset < Size) ? Size - offset : 0;

        // Parse MaxPool1d parameters from remaining bytes
        int64_t kernel_size = 1 + (consume<uint8_t>(current, remaining) % 16);
        
        // Stride can be 0 (defaults to kernel_size), or any positive value
        int64_t stride_raw = consume<uint8_t>(current, remaining);
        int64_t stride = (stride_raw == 0) ? kernel_size : 1 + (stride_raw % 16);
        
        // Padding can be 0 or positive
        int64_t padding = consume<uint8_t>(current, remaining) % (kernel_size / 2 + 1);
        
        // Dilation must be positive
        int64_t dilation = 1 + (consume<uint8_t>(current, remaining) % 4);
        
        // Return indices flag
        bool return_indices = consume<uint8_t>(current, remaining) & 1;
        
        // Ceil mode flag
        bool ceil_mode = consume<uint8_t>(current, remaining) & 1;

        // Create MaxPool1d module with diverse configurations
        torch::nn::MaxPool1dOptions options(kernel_size);
        options.stride(stride);
        options.padding(padding);
        options.dilation(dilation);
        options.return_indices(return_indices);
        options.ceil_mode(ceil_mode);

        torch::nn::MaxPool1d pool(options);

        // Handle different input shapes
        if (input.dim() == 1) {
            // Add batch and channel dimensions if needed
            input = input.unsqueeze(0).unsqueeze(0);
        } else if (input.dim() == 2) {
            // Add batch dimension
            input = input.unsqueeze(0);
        } else if (input.dim() > 3) {
            // Flatten extra dimensions
            auto sizes = input.sizes();
            int64_t batch = sizes[0];
            int64_t channels = (sizes.size() > 1) ? sizes[1] : 1;
            int64_t length = 1;
            for (size_t i = 2; i < sizes.size(); ++i) {
                length *= sizes[i];
            }
            input = input.reshape({batch, channels, length});
        }

        // Ensure input has correct shape (N, C, L)
        if (input.dim() != 3) {
            // Fallback to safe shape
            input = torch::randn({1, 1, 10});
        }

        // Test with different data types
        uint8_t dtype_selector = consume<uint8_t>(current, remaining);
        if (dtype_selector % 4 == 0 && input.dtype() != torch::kFloat32) {
            input = input.to(torch::kFloat32);
        } else if (dtype_selector % 4 == 1 && input.dtype() != torch::kFloat64) {
            input = input.to(torch::kFloat64);
        } else if (dtype_selector % 4 == 2 && torch::cuda::is_available()) {
            try {
                input = input.to(torch::kCUDA);
            } catch (...) {
                // CUDA operation failed, continue with CPU
            }
        }

        // Apply MaxPool1d
        if (return_indices) {
            auto result = pool->forward_with_indices(input);
            torch::Tensor output = std::get<0>(result);
            torch::Tensor indices = std::get<1>(result);
            
            // Perform some operations to ensure computation
            auto sum_output = output.sum();
            auto max_indices = indices.max();
            
            // Test backward pass if possible
            if (output.requires_grad() && remaining > 0) {
                try {
                    output.sum().backward();
                } catch (...) {
                    // Backward pass failed, continue
                }
            }
        } else {
            torch::Tensor output = pool->forward(input);
            
            // Perform operations to ensure computation
            auto sum = output.sum();
            auto mean = output.mean();
            
            // Test edge cases with special values
            if (remaining > 0 && consume<uint8_t>(current, remaining) % 4 == 0) {
                // Test with NaN/Inf values
                auto special_input = input.clone();
                if (special_input.numel() > 0) {
                    special_input.view(-1)[0] = std::numeric_limits<float>::quiet_NaN();
                    if (special_input.numel() > 1) {
                        special_input.view(-1)[1] = std::numeric_limits<float>::infinity();
                    }
                    try {
                        auto special_output = pool->forward(special_input);
                    } catch (...) {
                        // Special value handling failed, continue
                    }
                }
            }
            
            // Test with zero-sized dimensions if configured
            if (remaining > 0 && consume<uint8_t>(current, remaining) % 8 == 0) {
                try {
                    torch::Tensor zero_input = torch::randn({1, 1, 0});
                    auto zero_output = pool->forward(zero_input);
                } catch (...) {
                    // Zero-sized operation failed, expected for some configurations
                }
            }
        }

        // Test module serialization/deserialization
        if (remaining > 0 && consume<uint8_t>(current, remaining) % 4 == 0) {
            try {
                std::stringstream stream;
                torch::save(pool, stream);
                torch::nn::MaxPool1d loaded_pool(kernel_size);
                torch::load(loaded_pool, stream);
                
                // Test loaded module
                auto loaded_output = loaded_pool->forward(input);
            } catch (...) {
                // Serialization test failed, continue
            }
        }

        // Test with different memory layouts
        if (remaining > 0 && consume<uint8_t>(current, remaining) % 3 == 0) {
            try {
                // Test with non-contiguous tensor
                auto transposed = input.transpose(1, 2);
                auto non_contiguous_output = pool->forward(transposed.transpose(1, 2));
            } catch (...) {
                // Non-contiguous operation failed, continue
            }
        }

    }
    catch (const c10::Error& e)
    {
        // PyTorch-specific errors
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}