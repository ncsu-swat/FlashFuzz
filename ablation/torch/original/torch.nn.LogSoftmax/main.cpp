#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes: 2 for tensor creation + 1 for dim parameter
        if (Size < 3) {
            return 0;
        }

        // Create input tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse dimension parameter if we have data left
        int64_t dim = 0;
        if (offset < Size) {
            // Use remaining byte(s) to determine dimension
            uint8_t dim_byte = Data[offset++];
            
            // For non-scalar tensors, compute valid dimension range
            if (input.dim() > 0) {
                // Dimension can be in range [-rank, rank-1]
                // Map byte to this range
                int64_t rank = input.dim();
                int64_t total_range = 2 * rank;  // From -rank to rank-1
                dim = (dim_byte % total_range) - rank;
                
                // Ensure dim is in valid range
                if (dim >= rank) {
                    dim = rank - 1;
                }
            }
        }

#ifdef DEBUG_FUZZ
        std::cout << "Input tensor shape: " << input.sizes() 
                  << ", dtype: " << input.dtype() 
                  << ", dim parameter: " << dim << std::endl;
#endif

        // Create LogSoftmax module
        torch::nn::LogSoftmax log_softmax(torch::nn::LogSoftmaxOptions(dim));
        
        // Apply LogSoftmax
        torch::Tensor output = log_softmax(input);
        
#ifdef DEBUG_FUZZ
        std::cout << "Output tensor shape: " << output.sizes() 
                  << ", dtype: " << output.dtype() << std::endl;
#endif

        // Additional operations to increase coverage
        
        // Test with different tensor configurations
        if (offset + 1 < Size) {
            uint8_t config_byte = Data[offset++];
            
            // Test with requires_grad
            if (config_byte & 0x01) {
                if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble ||
                    input.dtype() == torch::kHalf || input.dtype() == torch::kBFloat16) {
                    input.requires_grad_(true);
                    torch::Tensor grad_output = log_softmax(input);
                    
                    // Compute gradients if possible
                    if (grad_output.requires_grad()) {
                        auto sum_output = grad_output.sum();
                        if (sum_output.requires_grad()) {
                            sum_output.backward();
                        }
                    }
                }
            }
            
            // Test with different memory layouts
            if (config_byte & 0x02) {
                // Make tensor non-contiguous if possible
                if (input.dim() >= 2) {
                    input = input.transpose(0, input.dim() - 1);
                    torch::Tensor transposed_output = log_softmax(input);
                }
            }
            
            // Test with view/reshape operations
            if (config_byte & 0x04) {
                if (input.numel() > 0) {
                    // Try to reshape to different valid shape
                    std::vector<int64_t> new_shape;
                    if (input.numel() == 1) {
                        new_shape = {1};
                    } else if (input.numel() % 2 == 0) {
                        new_shape = {2, input.numel() / 2};
                    } else {
                        new_shape = {1, input.numel()};
                    }
                    
                    torch::Tensor reshaped = input.reshape(new_shape);
                    
                    // Apply LogSoftmax with appropriate dimension
                    int64_t new_dim = (reshaped.dim() > 0) ? (dim_byte % reshaped.dim()) : 0;
                    torch::nn::LogSoftmax reshaped_log_softmax(torch::nn::LogSoftmaxOptions(new_dim));
                    torch::Tensor reshaped_output = reshaped_log_softmax(reshaped);
                }
            }
            
            // Test with slicing operations
            if (config_byte & 0x08) {
                if (input.dim() > 0 && input.size(0) > 1) {
                    // Slice along first dimension
                    torch::Tensor sliced = input.slice(0, 0, input.size(0) / 2);
                    torch::Tensor sliced_output = log_softmax(sliced);
                }
            }
            
            // Test with different dimensions for multi-dimensional tensors
            if (config_byte & 0x10 && input.dim() > 1) {
                for (int64_t d = -input.dim(); d < input.dim(); ++d) {
                    torch::nn::LogSoftmax multi_dim_log_softmax(torch::nn::LogSoftmaxOptions(d));
                    torch::Tensor multi_output = multi_dim_log_softmax(input);
                }
            }
        }
        
        // Test edge cases based on tensor properties
        if (input.numel() == 0) {
            // Empty tensor - should handle gracefully
            torch::Tensor empty_output = log_softmax(input);
        }
        
        if (input.numel() == 1) {
            // Single element tensor
            torch::Tensor single_output = log_softmax(input);
            // For single element, log_softmax should return log(1) = 0
        }
        
        // Test with extreme values if we have floating point type
        if (offset + 1 < Size) {
            uint8_t extreme_byte = Data[offset++];
            if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
                if (extreme_byte & 0x01) {
                    // Test with infinity values
                    if (input.numel() > 0) {
                        input.data_ptr<float>()[0] = std::numeric_limits<float>::infinity();
                        torch::Tensor inf_output = log_softmax(input);
                    }
                }
                if (extreme_byte & 0x02) {
                    // Test with negative infinity
                    if (input.numel() > 0) {
                        input.data_ptr<float>()[0] = -std::numeric_limits<float>::infinity();
                        torch::Tensor neg_inf_output = log_softmax(input);
                    }
                }
                if (extreme_byte & 0x04) {
                    // Test with NaN values
                    if (input.numel() > 0) {
                        input.data_ptr<float>()[0] = std::numeric_limits<float>::quiet_NaN();
                        torch::Tensor nan_output = log_softmax(input);
                    }
                }
            }
        }
        
        // Verify output properties
        // LogSoftmax output should have same shape as input
        if (output.sizes() != input.sizes()) {
            std::cerr << "Unexpected: Output shape differs from input shape" << std::endl;
        }
        
        // Values should be in range [-inf, 0]
        if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
            auto max_val = output.max();
            if (max_val.item<float>() > 1e-6) {  // Small epsilon for numerical errors
                std::cerr << "Unexpected: LogSoftmax output > 0" << std::endl;
            }
        }
        
        // Test batch processing if we have enough data
        if (offset + 4 < Size && input.dim() >= 1) {
            uint32_t batch_size_raw;
            std::memcpy(&batch_size_raw, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            
            int64_t batch_size = (batch_size_raw % 8) + 1;  // Limit batch size to [1, 8]
            
            // Create batch by repeating input
            std::vector<torch::Tensor> batch_tensors;
            for (int64_t i = 0; i < batch_size; ++i) {
                batch_tensors.push_back(input.clone());
            }
            
            if (!batch_tensors.empty()) {
                torch::Tensor batched = torch::stack(batch_tensors);
                // Adjust dimension for batched tensor
                int64_t batch_dim = (batched.dim() > 0) ? (dim % batched.dim()) : 0;
                torch::nn::LogSoftmax batch_log_softmax(torch::nn::LogSoftmaxOptions(batch_dim));
                torch::Tensor batch_output = batch_log_softmax(batched);
            }
        }
    }
    catch (const c10::Error &e)
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