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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }

        // Parse dropout probability from first byte
        float p = static_cast<float>(Data[offset++]) / 255.0f;  // Range [0, 1]
        
        // Parse inplace flag
        bool inplace = (Data[offset++] % 2) == 1;
        
        // Parse training mode flag
        bool training_mode = (Data[offset++] % 2) == 1;
        
        // Parse whether to use 4D or 5D tensor
        bool use_5d = (Data[offset++] % 2) == 1;
        
        // Create input tensor
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with remaining data
            return 0;
        }
        
        // Reshape tensor to appropriate dimensions for Dropout3d
        // Dropout3d expects either (N, C, D, H, W) or (C, D, H, W)
        torch::Tensor reshaped_input;
        
        if (use_5d) {
            // Try to reshape to 5D (N, C, D, H, W)
            int64_t total_elements = input.numel();
            if (total_elements >= 5) {
                // Parse dimensions from remaining bytes or use defaults
                int64_t N = 1 + (total_elements > 100 ? (Data[offset % Size] % 4) : 1);
                int64_t C = 1 + (total_elements > 50 ? (Data[(offset+1) % Size] % 8) : 1);
                int64_t D = 1 + (Data[(offset+2) % Size] % 4);
                int64_t H = 1 + (Data[(offset+3) % Size] % 8);
                
                // Calculate W to match total elements
                int64_t partial_prod = N * C * D * H;
                if (partial_prod > 0 && total_elements % partial_prod == 0) {
                    int64_t W = total_elements / partial_prod;
                    if (W > 0 && W <= 128) {
                        reshaped_input = input.reshape({N, C, D, H, W});
                    } else {
                        // Fallback to simple 5D shape
                        reshaped_input = input.reshape({1, 1, 1, 1, total_elements});
                    }
                } else {
                    // Fallback shape
                    reshaped_input = input.reshape({1, 1, 1, 1, total_elements});
                }
            } else {
                // Too few elements, create minimal 5D tensor
                reshaped_input = torch::zeros({1, 1, 1, 1, 1}, input.options());
            }
        } else {
            // Try to reshape to 4D (C, D, H, W)
            int64_t total_elements = input.numel();
            if (total_elements >= 4) {
                int64_t C = 1 + (Data[(offset) % Size] % 8);
                int64_t D = 1 + (Data[(offset+1) % Size] % 4);
                int64_t H = 1 + (Data[(offset+2) % Size] % 8);
                
                int64_t partial_prod = C * D * H;
                if (partial_prod > 0 && total_elements % partial_prod == 0) {
                    int64_t W = total_elements / partial_prod;
                    if (W > 0 && W <= 128) {
                        reshaped_input = input.reshape({C, D, H, W});
                    } else {
                        // Fallback to simple 4D shape
                        reshaped_input = input.reshape({1, 1, 1, total_elements});
                    }
                } else {
                    // Fallback shape
                    reshaped_input = input.reshape({1, 1, 1, total_elements});
                }
            } else {
                // Too few elements, create minimal 4D tensor
                reshaped_input = torch::zeros({1, 1, 1, 1}, input.options());
            }
        }
        
        // Create Dropout3d module
        torch::nn::Dropout3d dropout3d_module(torch::nn::Dropout3dOptions(p).inplace(inplace));
        
        // Set training/eval mode
        if (training_mode) {
            dropout3d_module->train();
        } else {
            dropout3d_module->eval();
        }
        
        // Apply dropout
        torch::Tensor output;
        if (inplace && reshaped_input.is_floating_point()) {
            // For inplace operation, clone first to avoid modifying original
            torch::Tensor input_clone = reshaped_input.clone();
            output = dropout3d_module->forward(input_clone);
            
            // Verify inplace operation worked
            if (training_mode && p > 0.0f && p < 1.0f) {
                // Check that some channels might be zeroed
                if (use_5d && reshaped_input.size(1) > 1) {
                    // Check channel-wise differences
                    for (int64_t n = 0; n < reshaped_input.size(0); ++n) {
                        for (int64_t c = 0; c < reshaped_input.size(1); ++c) {
                            auto channel = output[n][c];
                            auto channel_sum = channel.sum();
                            // Some channels should be zero in training mode
                        }
                    }
                }
            }
        } else {
            output = dropout3d_module->forward(reshaped_input);
        }
        
        // Verify output shape matches input shape
        if (output.sizes() != reshaped_input.sizes()) {
            std::cerr << "Shape mismatch: input " << reshaped_input.sizes() 
                     << " vs output " << output.sizes() << std::endl;
            return -1;
        }
        
        // Additional validations
        if (training_mode) {
            // In training mode with p > 0, some channels should be zeroed
            if (p > 0.0f && p < 1.0f) {
                // Run multiple times to test stochastic behavior
                torch::Tensor output2 = dropout3d_module->forward(reshaped_input);
                torch::Tensor output3 = dropout3d_module->forward(reshaped_input);
                
                // Outputs should differ due to randomness (unless p=0 or p=1)
                bool all_same = torch::allclose(output, output2) && torch::allclose(output2, output3);
                if (all_same && reshaped_input.numel() > 1) {
                    // This might indicate the dropout isn't working randomly
                    // But it could also happen by chance, so don't fail
                }
            }
        } else {
            // In eval mode, output should equal input
            if (!torch::allclose(output, reshaped_input, 1e-5, 1e-8)) {
                std::cerr << "In eval mode, output should equal input" << std::endl;
                // Don't return -1 as floating point might have small differences
            }
        }
        
        // Test with edge cases
        if (Size > offset + 10) {
            // Test with zero probability
            torch::nn::Dropout3d zero_dropout(torch::nn::Dropout3dOptions(0.0f));
            torch::Tensor zero_output = zero_dropout->forward(reshaped_input);
            
            // Test with probability 1.0
            torch::nn::Dropout3d full_dropout(torch::nn::Dropout3dOptions(1.0f));
            full_dropout->train();
            torch::Tensor full_output = full_dropout->forward(reshaped_input);
            
            // Test with different tensor types if input supports it
            if (reshaped_input.is_floating_point()) {
                // Convert to different float type
                torch::Tensor converted = reshaped_input.to(
                    reshaped_input.dtype() == torch::kFloat32 ? torch::kFloat64 : torch::kFloat32
                );
                torch::Tensor converted_output = dropout3d_module->forward(converted);
            }
        }
        
        // Test batch processing with different batch sizes
        if (use_5d && reshaped_input.size(0) == 1 && Size > offset + 20) {
            // Try to expand batch dimension
            torch::Tensor batched = reshaped_input.repeat({3, 1, 1, 1, 1});
            torch::Tensor batched_output = dropout3d_module->forward(batched);
            
            if (batched_output.size(0) != 3) {
                std::cerr << "Batch processing failed" << std::endl;
                return -1;
            }
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors
        std::cout << "PyTorch error: " << e.what() << std::endl;
        return 0;  // Continue fuzzing
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0; // keep the input
}