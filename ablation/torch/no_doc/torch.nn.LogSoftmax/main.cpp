#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <vector>
#include <memory>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }

        // Parse dimension for LogSoftmax
        int64_t dim = 0;
        if (offset < Size) {
            uint8_t dim_byte = Data[offset++];
            // Allow negative dimensions too for testing edge cases
            dim = static_cast<int64_t>(static_cast<int8_t>(dim_byte));
        }

        // Create input tensor from fuzzer data
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with remaining data
            if (offset >= Size) {
                return 0;
            }
            // Create a simple fallback tensor
            std::vector<int64_t> shape = {2, 3};
            input = torch::randn(shape);
        }

        // Create LogSoftmax module with various configurations
        torch::nn::LogSoftmaxOptions options(dim);
        torch::nn::LogSoftmax log_softmax(options);

        // Test forward pass
        torch::Tensor output;
        try {
            output = log_softmax->forward(input);
        } catch (const c10::Error& e) {
            // PyTorch errors are expected for invalid dimensions
            return 0;
        }

        // Additional testing with different tensor properties
        if (offset < Size) {
            uint8_t test_flags = Data[offset++];
            
            // Test with requires_grad
            if (test_flags & 0x01) {
                try {
                    auto grad_input = input.clone().requires_grad_(true);
                    auto grad_output = log_softmax->forward(grad_input);
                    
                    // Test backward pass if gradients are enabled
                    if (grad_input.requires_grad() && (test_flags & 0x02)) {
                        auto grad_out = torch::ones_like(grad_output);
                        grad_output.backward(grad_out);
                    }
                } catch (...) {
                    // Gradient operations may fail, continue
                }
            }

            // Test with different memory layouts
            if (test_flags & 0x04) {
                try {
                    // Test with non-contiguous tensor
                    if (input.dim() >= 2) {
                        auto transposed = input.transpose(0, -1);
                        log_softmax->forward(transposed);
                    }
                } catch (...) {
                    // Continue on error
                }
            }

            // Test with different dimensions
            if (test_flags & 0x08) {
                for (int64_t test_dim = -input.dim(); test_dim < input.dim(); ++test_dim) {
                    try {
                        torch::nn::LogSoftmax test_log_softmax(test_dim);
                        test_log_softmax->forward(input);
                    } catch (...) {
                        // Expected for out-of-range dimensions
                    }
                }
            }

            // Test with zero-sized tensors
            if (test_flags & 0x10) {
                try {
                    std::vector<std::vector<int64_t>> zero_shapes = {
                        {0}, {0, 5}, {5, 0}, {0, 0}, {1, 0, 3}
                    };
                    for (const auto& shape : zero_shapes) {
                        try {
                            auto zero_tensor = torch::zeros(shape, input.options());
                            log_softmax->forward(zero_tensor);
                        } catch (...) {
                            // Continue
                        }
                    }
                } catch (...) {
                    // Continue
                }
            }

            // Test with extreme dimensions
            if (test_flags & 0x20) {
                std::vector<int64_t> extreme_dims = {
                    std::numeric_limits<int64_t>::min(),
                    std::numeric_limits<int64_t>::max(),
                    -1000, 1000
                };
                for (auto extreme_dim : extreme_dims) {
                    try {
                        torch::nn::LogSoftmax extreme_log_softmax(extreme_dim);
                        extreme_log_softmax->forward(input);
                    } catch (...) {
                        // Expected to fail
                    }
                }
            }

            // Test with different tensor types if we have more data
            if (offset < Size && (test_flags & 0x40)) {
                try {
                    // Convert to different types and test
                    std::vector<torch::ScalarType> test_types = {
                        torch::kFloat32, torch::kFloat64, torch::kFloat16,
                        torch::kBFloat16, torch::kInt32, torch::kInt64
                    };
                    
                    for (auto dtype : test_types) {
                        try {
                            auto typed_input = input.to(dtype);
                            log_softmax->forward(typed_input);
                        } catch (...) {
                            // Some types may not be supported
                        }
                    }
                } catch (...) {
                    // Continue
                }
            }

            // Test in-place operations if supported
            if (test_flags & 0x80) {
                try {
                    auto inplace_input = input.clone();
                    // LogSoftmax doesn't have in-place variant in nn module,
                    // but we can test the functional version
                    torch::nn::functional::log_softmax(inplace_input, 
                        torch::nn::functional::LogSoftmaxFuncOptions(dim));
                } catch (...) {
                    // Continue
                }
            }
        }

        // Test with batch processing
        if (offset < Size) {
            uint8_t batch_size = Data[offset++] % 16 + 1;
            try {
                // Create batched input
                std::vector<torch::Tensor> batch;
                for (int i = 0; i < batch_size; ++i) {
                    batch.push_back(input.clone());
                }
                auto batched = torch::stack(batch);
                log_softmax->forward(batched);
            } catch (...) {
                // Batching might fail for certain tensor shapes
            }
        }

        // Verify output properties
        if (output.defined()) {
            // Check that output has same shape as input
            if (output.sizes() != input.sizes()) {
                std::cerr << "Warning: Output shape mismatch" << std::endl;
            }
            
            // Verify log_softmax properties: sum(exp(output), dim) should be ~1
            try {
                auto exp_output = output.exp();
                auto sum_exp = exp_output.sum(dim);
                auto ones = torch::ones_like(sum_exp);
                if (!torch::allclose(sum_exp, ones, 1e-5, 1e-8)) {
                    // This might indicate numerical issues
                }
            } catch (...) {
                // Verification might fail for edge cases
            }
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}