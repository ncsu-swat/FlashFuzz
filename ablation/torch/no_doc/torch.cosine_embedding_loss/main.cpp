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
        if (Size < 10) {
            return 0;
        }

        // Create input1 tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create input2 tensor with same shape as input1 if possible
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
            // Try to match shapes - if they don't match, reshape or expand
            if (input2.numel() > 0 && input1.numel() > 0) {
                if (input2.sizes() != input1.sizes()) {
                    // Try to reshape input2 to match input1's shape
                    if (input2.numel() == input1.numel()) {
                        input2 = input2.reshape(input1.sizes());
                    } else {
                        // Create a new tensor with matching shape
                        input2 = torch::randn_like(input1);
                    }
                }
            }
        } else {
            // Create random tensor matching input1
            input2 = torch::randn_like(input1);
        }

        // Create target tensor
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Target should contain 1 or -1 values
            // Convert to appropriate shape and values
            if (target.numel() > 0 && input1.dim() > 0) {
                // For cosine_embedding_loss, target should be 1D with size matching batch dimension
                int64_t batch_size = input1.size(0);
                if (target.numel() >= batch_size) {
                    target = target.flatten().slice(0, 0, batch_size);
                } else {
                    // Expand or repeat to match batch size
                    target = target.flatten();
                    if (target.numel() > 0) {
                        target = target.repeat({(batch_size + target.numel() - 1) / target.numel()})
                                      .slice(0, 0, batch_size);
                    } else {
                        target = torch::ones({batch_size});
                    }
                }
                // Convert to 1 or -1
                target = torch::where(target > 0, torch::ones_like(target), -torch::ones_like(target));
            } else if (input1.dim() == 0) {
                // Scalar case
                target = torch::ones({1});
            } else {
                target = torch::ones({1});
            }
        } else {
            // Create default target
            if (input1.dim() > 0) {
                int64_t batch_size = input1.size(0);
                target = torch::ones({batch_size});
                // Mix of 1 and -1 based on remaining data
                if (offset < Size) {
                    uint8_t selector = Data[offset++];
                    if (selector % 2 == 0) {
                        target = -target;
                    } else if (selector % 3 == 0) {
                        // Mix of 1 and -1
                        for (int64_t i = 0; i < batch_size; i++) {
                            if (i % 2 == 0) target[i] = -1;
                        }
                    }
                }
            } else {
                target = torch::ones({1});
            }
        }

        // Parse margin parameter
        double margin = 0.0;
        if (offset + sizeof(float) <= Size) {
            float margin_raw;
            std::memcpy(&margin_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp margin to reasonable range
            margin = std::fmax(-1.0, std::fmin(1.0, static_cast<double>(margin_raw)));
        } else if (offset < Size) {
            // Use single byte for margin
            margin = (Data[offset++] / 255.0) * 2.0 - 1.0; // Range [-1, 1]
        }

        // Parse reduction type
        torch::Reduction::Reduction reduction = torch::Reduction::Mean;
        if (offset < Size) {
            uint8_t reduction_selector = Data[offset++];
            switch (reduction_selector % 3) {
                case 0:
                    reduction = torch::Reduction::None;
                    break;
                case 1:
                    reduction = torch::Reduction::Mean;
                    break;
                case 2:
                    reduction = torch::Reduction::Sum;
                    break;
            }
        }

        // Ensure tensors have compatible types for the operation
        if (input1.dtype() != input2.dtype()) {
            // Convert to common dtype (prefer floating point for cosine similarity)
            if (!input1.is_floating_point()) {
                input1 = input1.to(torch::kFloat32);
            }
            input2 = input2.to(input1.dtype());
        }
        
        // Ensure target is proper dtype (typically float for loss functions)
        if (!target.is_floating_point()) {
            target = target.to(torch::kFloat32);
        }
        
        // Match dtypes between inputs and target
        if (input1.dtype() != target.dtype()) {
            target = target.to(input1.dtype());
        }

        // Handle edge cases for dimensions
        if (input1.dim() == 0 && input2.dim() == 0) {
            // Scalar inputs - expand to 1D
            input1 = input1.unsqueeze(0);
            input2 = input2.unsqueeze(0);
        }
        
        // Ensure inputs are at least 1D for cosine_embedding_loss
        if (input1.dim() == 0) {
            input1 = input1.reshape({1, 1});
            input2 = input2.reshape({1, 1});
        } else if (input1.dim() == 1) {
            // 1D tensors should be treated as batch of 1
            input1 = input1.unsqueeze(0);
            input2 = input2.unsqueeze(0);
        }

        // Call cosine_embedding_loss with different configurations
        torch::Tensor result;
        
        // Try the main operation
        result = torch::cosine_embedding_loss(input1, input2, target, 
                                             torch::cosine_embedding_loss_options()
                                                 .margin(margin)
                                                 .reduction(reduction));

        // Try with different options to increase coverage
        if (offset < Size && Data[offset++] % 4 == 0) {
            // Try with different margin
            double alt_margin = (Data[offset % Size] / 255.0);
            auto alt_result = torch::cosine_embedding_loss(input1, input2, target,
                                                          torch::cosine_embedding_loss_options()
                                                              .margin(alt_margin)
                                                              .reduction(torch::Reduction::None));
            
            // Try manual reduction on the alt_result
            if (alt_result.numel() > 0) {
                if (offset < Size && Data[offset++] % 2 == 0) {
                    auto mean_result = alt_result.mean();
                } else {
                    auto sum_result = alt_result.sum();
                }
            }
        }

        // Additional operations to explore more paths
        if (result.numel() > 0) {
            // Check some properties
            bool is_finite = torch::isfinite(result).all().item<bool>();
            
            // Try backward pass if result requires grad
            if (offset < Size && Data[offset++] % 3 == 0) {
                input1.requires_grad_(true);
                input2.requires_grad_(true);
                
                auto loss = torch::cosine_embedding_loss(input1, input2, target,
                                                        torch::cosine_embedding_loss_options()
                                                            .margin(margin)
                                                            .reduction(torch::Reduction::Mean));
                if (loss.requires_grad() && loss.numel() == 1) {
                    loss.backward();
                }
            }
        }

        // Test edge cases explicitly
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            if (edge_case % 5 == 0) {
                // Test with zero tensors
                auto zero1 = torch::zeros_like(input1);
                auto zero2 = torch::zeros_like(input2);
                try {
                    auto edge_result = torch::cosine_embedding_loss(zero1, zero2, target,
                                                                   torch::cosine_embedding_loss_options()
                                                                       .margin(0.5));
                } catch (const c10::Error& e) {
                    // Expected for zero vectors (undefined cosine similarity)
                }
            } else if (edge_case % 5 == 1) {
                // Test with very large values
                auto large1 = input1 * 1e10;
                auto large2 = input2 * 1e10;
                try {
                    auto edge_result = torch::cosine_embedding_loss(large1, large2, target,
                                                                   torch::cosine_embedding_loss_options()
                                                                       .margin(margin));
                } catch (const c10::Error& e) {
                    // Handle potential overflow
                }
            } else if (edge_case % 5 == 2) {
                // Test with NaN/Inf values
                if (input1.numel() > 0) {
                    input1[0] = std::numeric_limits<float>::quiet_NaN();
                    try {
                        auto edge_result = torch::cosine_embedding_loss(input1, input2, target,
                                                                       torch::cosine_embedding_loss_options());
                    } catch (const c10::Error& e) {
                        // Handle NaN propagation
                    }
                }
            }
        }
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors - these are expected for invalid operations
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}