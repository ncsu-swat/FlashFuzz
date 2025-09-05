#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Minimum size check - we need at least enough bytes for basic tensor metadata
        // We need: input, hidden, weight_ih, weight_hh, bias_ih (optional), bias_hh (optional)
        // Each tensor needs at least 2 bytes (dtype + rank)
        if (Size < 8) {
            return 0; // Not enough data to create meaningful tensors
        }

        // Parse configuration byte for optional parameters
        bool use_bias = (offset < Size) ? (Data[offset++] & 0x01) : false;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            // Try with minimal input
            input = torch::randn({1, 1});
        }
        
        // Create hidden state tensor
        torch::Tensor hidden = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            // Match dimensions with input if possible
            int64_t batch_size = input.size(0) > 0 ? input.size(0) : 1;
            hidden = torch::randn({batch_size, 1});
        }
        
        // Determine dimensions
        int64_t batch_size = input.size(0);
        int64_t input_size = input.size(-1);
        int64_t hidden_size = hidden.size(-1);
        
        // Ensure tensors are 2D (batch_size x features)
        if (input.dim() != 2) {
            input = input.reshape({batch_size > 0 ? batch_size : 1, 
                                  input.numel() / (batch_size > 0 ? batch_size : 1)});
        }
        if (hidden.dim() != 2) {
            hidden = hidden.reshape({batch_size > 0 ? batch_size : 1,
                                    hidden.numel() / (batch_size > 0 ? batch_size : 1)});
        }
        
        // Update dimensions after reshape
        batch_size = input.size(0);
        input_size = input.size(1);
        hidden_size = hidden.size(1);
        
        // Create weight matrices
        torch::Tensor weight_ih, weight_hh;
        
        if (offset < Size) {
            weight_ih = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure correct shape for weight_ih
            if (weight_ih.numel() != hidden_size * input_size) {
                weight_ih = torch::randn({hidden_size, input_size}, input.options());
            } else {
                weight_ih = weight_ih.reshape({hidden_size, input_size}).to(input.dtype());
            }
        } else {
            weight_ih = torch::randn({hidden_size, input_size}, input.options());
        }
        
        if (offset < Size) {
            weight_hh = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure correct shape for weight_hh
            if (weight_hh.numel() != hidden_size * hidden_size) {
                weight_hh = torch::randn({hidden_size, hidden_size}, input.options());
            } else {
                weight_hh = weight_hh.reshape({hidden_size, hidden_size}).to(input.dtype());
            }
        } else {
            weight_hh = torch::randn({hidden_size, hidden_size}, input.options());
        }
        
        // Create bias tensors if needed
        torch::Tensor bias_ih, bias_hh;
        
        if (use_bias) {
            if (offset < Size) {
                bias_ih = fuzzer_utils::createTensor(Data, Size, offset);
                if (bias_ih.numel() != hidden_size) {
                    bias_ih = torch::randn({hidden_size}, input.options());
                } else {
                    bias_ih = bias_ih.reshape({hidden_size}).to(input.dtype());
                }
            } else {
                bias_ih = torch::randn({hidden_size}, input.options());
            }
            
            if (offset < Size) {
                bias_hh = fuzzer_utils::createTensor(Data, Size, offset);
                if (bias_hh.numel() != hidden_size) {
                    bias_hh = torch::randn({hidden_size}, input.options());
                } else {
                    bias_hh = bias_hh.reshape({hidden_size}).to(input.dtype());
                }
            } else {
                bias_hh = torch::randn({hidden_size}, input.options());
            }
        }
        
        // Ensure all tensors have compatible dtypes
        auto target_dtype = input.dtype();
        hidden = hidden.to(target_dtype);
        weight_ih = weight_ih.to(target_dtype);
        weight_hh = weight_hh.to(target_dtype);
        if (use_bias) {
            bias_ih = bias_ih.to(target_dtype);
            bias_hh = bias_hh.to(target_dtype);
        }
        
        // Ensure batch sizes match
        if (hidden.size(0) != batch_size) {
            hidden = hidden.expand({batch_size, hidden_size}).contiguous();
        }
        
        // Call rnn_tanh_cell with different configurations
        torch::Tensor output;
        
        try {
            if (use_bias) {
                output = torch::rnn_tanh_cell(input, hidden, weight_ih, weight_hh, bias_ih, bias_hh);
            } else {
                output = torch::rnn_tanh_cell(input, hidden, weight_ih, weight_hh, {}, {});
            }
            
            // Additional operations to increase coverage
            if (output.defined()) {
                // Test backward pass
                if (output.requires_grad()) {
                    auto loss = output.sum();
                    loss.backward();
                }
                
                // Test with different memory formats
                if (offset < Size && Data[offset % Size] & 0x02) {
                    auto output_channels_last = output.contiguous(torch::MemoryFormat::ChannelsLast);
                }
                
                // Test detach and clone
                auto detached = output.detach();
                auto cloned = output.clone();
                
                // Test in-place operations
                if (offset < Size && Data[offset % Size] & 0x04) {
                    output.add_(1.0);
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific errors - these are expected for invalid inputs
            return 0;
        }
        
        // Try with requires_grad enabled
        if (offset < Size && Data[offset % Size] & 0x08) {
            input.requires_grad_(true);
            hidden.requires_grad_(true);
            weight_ih.requires_grad_(true);
            weight_hh.requires_grad_(true);
            
            try {
                if (use_bias) {
                    bias_ih.requires_grad_(true);
                    bias_hh.requires_grad_(true);
                    output = torch::rnn_tanh_cell(input, hidden, weight_ih, weight_hh, bias_ih, bias_hh);
                } else {
                    output = torch::rnn_tanh_cell(input, hidden, weight_ih, weight_hh, {}, {});
                }
                
                if (output.defined() && output.requires_grad()) {
                    auto loss = output.mean();
                    loss.backward();
                }
            } catch (const c10::Error& e) {
                // Expected for some invalid configurations
                return 0;
            }
        }
        
        // Test with different tensor properties
        if (offset < Size && Data[offset % Size] & 0x10) {
            // Try with non-contiguous tensors
            auto input_t = input.t().t(); // Makes it potentially non-contiguous
            auto hidden_t = hidden.t().t();
            
            try {
                if (use_bias) {
                    output = torch::rnn_tanh_cell(input_t, hidden_t, weight_ih, weight_hh, bias_ih, bias_hh);
                } else {
                    output = torch::rnn_tanh_cell(input_t, hidden_t, weight_ih, weight_hh, {}, {});
                }
            } catch (const c10::Error& e) {
                return 0;
            }
        }
        
        // Test edge cases with zero-sized tensors
        if (offset < Size && Data[offset % Size] & 0x20) {
            try {
                auto empty_input = torch::empty({0, input_size}, input.options());
                auto empty_hidden = torch::empty({0, hidden_size}, hidden.options());
                
                if (use_bias) {
                    output = torch::rnn_tanh_cell(empty_input, empty_hidden, weight_ih, weight_hh, bias_ih, bias_hh);
                } else {
                    output = torch::rnn_tanh_cell(empty_input, empty_hidden, weight_ih, weight_hh, {}, {});
                }
            } catch (const c10::Error& e) {
                return 0;
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}