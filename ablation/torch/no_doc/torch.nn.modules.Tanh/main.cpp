#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Minimum size check - we need at least some bytes for tensor creation
        if (Size < 3) {
            // Still try to create something minimal
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            auto input = torch::zeros({1}, options);
            torch::nn::Tanh tanh_module;
            auto output = tanh_module->forward(input);
            return 0;
        }

        // Create Tanh module
        torch::nn::Tanh tanh_module;
        
        // Parse number of tensors to test (1-5)
        uint8_t num_tensors = 1;
        if (offset < Size) {
            num_tensors = (Data[offset++] % 5) + 1;
        }
        
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                // Create input tensor from fuzzer data
                torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Test different configurations based on fuzzer input
                if (offset < Size) {
                    uint8_t config_byte = Data[offset++];
                    
                    // Test with different tensor properties
                    if (config_byte & 0x01) {
                        // Test with requires_grad
                        input = input.requires_grad_(true);
                    }
                    
                    if (config_byte & 0x02) {
                        // Test with non-contiguous tensor
                        if (input.dim() > 1 && input.size(0) > 1 && input.size(1) > 1) {
                            input = input.transpose(0, 1);
                        }
                    }
                    
                    if (config_byte & 0x04) {
                        // Test with different device (if CUDA available)
                        if (torch::cuda::is_available()) {
                            input = input.to(torch::kCUDA);
                        }
                    }
                    
                    if (config_byte & 0x08) {
                        // Test with sparse tensor (if applicable)
                        if (input.dim() == 2 && input.dtype() == torch::kFloat32) {
                            try {
                                auto indices = torch::tensor({{0, 1}, {1, 0}}).t();
                                auto values = torch::tensor({1.0f, 2.0f});
                                auto sparse_input = torch::sparse_coo_tensor(indices, values, input.sizes());
                                // Tanh doesn't support sparse directly, but test conversion
                                input = sparse_input.to_dense();
                            } catch (...) {
                                // Sparse creation failed, continue with dense
                            }
                        }
                    }
                    
                    if (config_byte & 0x10) {
                        // Test with extreme values
                        if (offset < Size) {
                            uint8_t extreme_type = Data[offset++] % 6;
                            switch (extreme_type) {
                                case 0: input.fill_(std::numeric_limits<float>::infinity()); break;
                                case 1: input.fill_(-std::numeric_limits<float>::infinity()); break;
                                case 2: input.fill_(std::numeric_limits<float>::quiet_NaN()); break;
                                case 3: input.fill_(std::numeric_limits<float>::max()); break;
                                case 4: input.fill_(std::numeric_limits<float>::min()); break;
                                case 5: input.fill_(std::numeric_limits<float>::epsilon()); break;
                            }
                        }
                    }
                }
                
                // Apply Tanh forward pass
                torch::Tensor output = tanh_module->forward(input);
                
                // Additional operations to increase coverage
                if (offset < Size) {
                    uint8_t post_op = Data[offset++];
                    
                    if (post_op & 0x01) {
                        // Test backward pass if gradient enabled
                        if (input.requires_grad()) {
                            auto loss = output.sum();
                            loss.backward();
                        }
                    }
                    
                    if (post_op & 0x02) {
                        // Test in-place operation
                        auto input_clone = input.clone();
                        input_clone.tanh_();
                    }
                    
                    if (post_op & 0x04) {
                        // Test with different memory formats
                        if (input.dim() == 4) {
                            try {
                                auto channels_last_input = input.to(torch::MemoryFormat::ChannelsLast);
                                auto cl_output = tanh_module->forward(channels_last_input);
                            } catch (...) {
                                // Memory format conversion failed, continue
                            }
                        }
                    }
                    
                    if (post_op & 0x08) {
                        // Test module serialization/deserialization
                        try {
                            std::stringstream stream;
                            torch::save(tanh_module, stream);
                            torch::nn::Tanh loaded_module;
                            torch::load(loaded_module, stream);
                            auto loaded_output = loaded_module->forward(input);
                        } catch (...) {
                            // Serialization failed, continue
                        }
                    }
                    
                    if (post_op & 0x10) {
                        // Test with batched input
                        if (input.dim() >= 1) {
                            auto batched = input.unsqueeze(0).repeat({3, 1});
                            for (int j = 1; j < input.dim(); ++j) {
                                batched = batched.narrow(j + 1, 0, input.size(j));
                            }
                            auto batch_output = tanh_module->forward(batched);
                        }
                    }
                }
                
                // Verify output properties
                if (!output.defined()) {
                    std::cerr << "Output tensor is not defined" << std::endl;
                    return -1;
                }
                
                // Check output shape matches input shape
                if (output.sizes() != input.sizes()) {
                    std::cerr << "Shape mismatch: input " << input.sizes() 
                             << " vs output " << output.sizes() << std::endl;
                }
                
                // Verify tanh bounds (-1, 1)
                if (output.dtype().isFloatingPoint() && !output.any(output != output).item<bool>()) {
                    // Only check bounds if not NaN
                    auto min_val = output.min().item<float>();
                    auto max_val = output.max().item<float>();
                    if (min_val < -1.0f - 1e-6f || max_val > 1.0f + 1e-6f) {
                        std::cerr << "Tanh output out of bounds: [" << min_val << ", " << max_val << "]" << std::endl;
                    }
                }
                
            } catch (const c10::Error& e) {
                // PyTorch-specific errors, continue fuzzing
                continue;
            } catch (const std::bad_alloc& e) {
                // Memory allocation failed, likely due to large tensor request
                continue;
            }
        }
        
        // Test edge cases with manually crafted tensors
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Test with zero-dimensional tensor (scalar)
            auto scalar = torch::tensor(3.14f);
            auto scalar_out = tanh_module->forward(scalar);
            
            // Test with empty tensor
            auto empty = torch::empty({0, 5});
            auto empty_out = tanh_module->forward(empty);
            
            // Test with single element tensor
            auto single = torch::tensor({1.0f});
            auto single_out = tanh_module->forward(single);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}