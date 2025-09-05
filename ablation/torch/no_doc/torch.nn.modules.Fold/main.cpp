#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes and create size parameters
template<typename T>
T consumeValue(const uint8_t* data, size_t& offset, size_t size, T min_val, T max_val) {
    if (offset + sizeof(T) > size) {
        offset = size;
        return min_val;
    }
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    // Constrain to reasonable range
    value = std::abs(value) % (max_val - min_val + 1) + min_val;
    return value;
}

// Helper to create size vectors for kernel, stride, padding, dilation
std::vector<int64_t> consumeSizeVector(const uint8_t* data, size_t& offset, size_t size, int dims, int min_val, int max_val) {
    std::vector<int64_t> result;
    for (int i = 0; i < dims; ++i) {
        result.push_back(consumeValue<int64_t>(data, offset, size, min_val, max_val));
    }
    return result;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Minimum size check for basic parameters
        if (Size < 10) {
            return 0;
        }

        // Consume dimension choice (1D or 2D fold)
        int dims = (Data[offset++] % 2) + 1;  // 1 or 2 dimensions
        
        // Create input tensor - Fold expects (N, C×∏(kernel_size), L) for 1D or (N, C×∏(kernel_size), H×W) for 2D
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (...) {
            // If tensor creation fails, create a default one
            if (dims == 1) {
                input = torch::randn({2, 12, 8}); // N=2, C*kernel=12, L=8
            } else {
                input = torch::randn({2, 18, 12}); // N=2, C*kernel=18, L=12 (will be reshaped)
            }
        }
        
        // Ensure input has at least 3 dimensions
        if (input.dim() < 3) {
            // Reshape to have 3 dimensions
            int64_t total_elements = input.numel();
            if (total_elements < 8) {
                input = torch::randn({1, 4, 2});
            } else {
                int64_t batch = 1;
                int64_t channels = std::min(total_elements / 2, int64_t(16));
                int64_t length = total_elements / channels;
                input = input.reshape({batch, channels, length});
            }
        }
        
        // Get actual input dimensions
        int64_t batch_size = input.size(0);
        int64_t channels_x_kernel = input.size(1);
        int64_t input_length = input.size(2);
        
        // Consume kernel size
        auto kernel_size = consumeSizeVector(Data, offset, Size, dims, 1, 5);
        
        // Calculate channels from input (channels * prod(kernel_size) = channels_x_kernel)
        int64_t kernel_prod = 1;
        for (auto k : kernel_size) {
            kernel_prod *= k;
        }
        
        // Ensure valid channel count
        if (kernel_prod == 0 || channels_x_kernel % kernel_prod != 0) {
            // Adjust to make it divisible
            kernel_prod = std::max(int64_t(1), kernel_prod);
            channels_x_kernel = ((channels_x_kernel / kernel_prod) + 1) * kernel_prod;
            if (dims == 1) {
                input = torch::randn({batch_size, channels_x_kernel, input_length});
            } else {
                input = torch::randn({batch_size, channels_x_kernel, input_length});
            }
        }
        
        // Consume stride
        auto stride = consumeSizeVector(Data, offset, Size, dims, 1, 3);
        
        // Consume padding  
        auto padding = consumeSizeVector(Data, offset, Size, dims, 0, 3);
        
        // Consume dilation
        auto dilation = consumeSizeVector(Data, offset, Size, dims, 1, 2);
        
        // Calculate output_size based on the fold formula
        // For unfold: L_out = (L_in + 2*padding - dilation*(kernel-1) - 1) / stride + 1
        // For fold (inverse): L_in = (L_out - 1) * stride - 2*padding + dilation*(kernel-1) + 1
        std::vector<int64_t> output_size;
        
        if (dims == 1) {
            // 1D case
            int64_t out_length = consumeValue<int64_t>(Data, offset, Size, 4, 32);
            output_size = {out_length};
        } else {
            // 2D case  
            int64_t out_height = consumeValue<int64_t>(Data, offset, Size, 4, 16);
            int64_t out_width = consumeValue<int64_t>(Data, offset, Size, 4, 16);
            output_size = {out_height, out_width};
        }
        
        // Create Fold module based on dimensions
        if (dims == 1) {
            // torch::nn::Fold1d
            auto fold_options = torch::nn::Fold1dOptions(output_size, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation);
                
            auto fold = torch::nn::Fold1d(fold_options);
            
            // Apply fold operation
            torch::Tensor output;
            try {
                output = fold->forward(input);
                
                // Additional operations to increase coverage
                if (offset < Size && Data[offset++] % 2 == 0) {
                    // Try backward pass
                    if (output.requires_grad()) {
                        auto loss = output.sum();
                        loss.backward();
                    }
                }
                
                // Check output properties
                if (offset < Size && Data[offset++] % 3 == 0) {
                    auto min_val = output.min();
                    auto max_val = output.max();
                    auto mean_val = output.mean();
                }
                
            } catch (const c10::Error& e) {
                // Specific PyTorch errors are expected for invalid configurations
                return 0;
            }
            
        } else {
            // torch::nn::Fold2d
            auto fold_options = torch::nn::Fold2dOptions(output_size, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation);
                
            auto fold = torch::nn::Fold2d(fold_options);
            
            // Apply fold operation
            torch::Tensor output;
            try {
                output = fold->forward(input);
                
                // Additional operations
                if (offset < Size && Data[offset++] % 2 == 0) {
                    // Try with different input shapes
                    auto input2 = torch::randn({batch_size * 2, channels_x_kernel, input_length * 2});
                    auto output2 = fold->forward(input2);
                }
                
                // Test edge cases
                if (offset < Size && Data[offset++] % 3 == 0) {
                    // Zero input
                    auto zero_input = torch::zeros_like(input);
                    auto zero_output = fold->forward(zero_input);
                    
                    // Ones input  
                    auto ones_input = torch::ones_like(input);
                    auto ones_output = fold->forward(ones_input);
                }
                
            } catch (const c10::Error& e) {
                // Expected for invalid configurations
                return 0;
            }
        }
        
        // Try creating fold with different data types
        if (offset < Size && Data[offset++] % 4 == 0) {
            auto dtype_selector = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            auto typed_input = input.to(dtype);
            
            if (dims == 1) {
                auto fold = torch::nn::Fold1d(torch::nn::Fold1dOptions(output_size, kernel_size));
                try {
                    auto output = fold->forward(typed_input);
                } catch (...) {
                    // Some dtypes may not be supported
                }
            } else {
                auto fold = torch::nn::Fold2d(torch::nn::Fold2dOptions(output_size, kernel_size));
                try {
                    auto output = fold->forward(typed_input);
                } catch (...) {
                    // Some dtypes may not be supported
                }
            }
        }
        
        // Test with batch size variations
        if (offset < Size && Data[offset++] % 3 == 0) {
            std::vector<int64_t> batch_sizes = {0, 1, 3, 7};
            for (auto bs : batch_sizes) {
                if (bs == 0 && input.size(0) == 0) continue;
                
                torch::Tensor batch_input;
                if (bs == 0) {
                    batch_input = torch::randn({0, channels_x_kernel, input_length});
                } else {
                    batch_input = input.narrow(0, 0, std::min(bs, input.size(0)));
                }
                
                try {
                    if (dims == 1) {
                        auto fold = torch::nn::Fold1d(torch::nn::Fold1dOptions(output_size, kernel_size));
                        auto output = fold->forward(batch_input);
                    } else {
                        auto fold = torch::nn::Fold2d(torch::nn::Fold2dOptions(output_size, kernel_size));
                        auto output = fold->forward(batch_input);
                    }
                } catch (...) {
                    // Expected for some configurations
                }
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}