#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>

// Helper to consume values from fuzzer data
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
    if (Size < 4) {
        return 0; // Need minimum bytes for basic parameters
    }

    try
    {
        size_t offset = 0;
        const uint8_t* current_data = Data;
        size_t remaining = Size;

        // Create input tensor with diverse characteristics
        torch::Tensor input;
        
        // Decide tensor creation strategy based on first byte
        uint8_t strategy = consume<uint8_t>(current_data, remaining);
        
        if (strategy % 3 == 0) {
            // Use fuzzer_utils to create tensor
            try {
                input = fuzzer_utils::createTensor(Data, Size, offset);
                current_data = Data + offset;
                remaining = (offset < Size) ? Size - offset : 0;
            } catch (...) {
                // Fallback to simple tensor
                input = torch::randn({2, 3});
            }
        } else if (strategy % 3 == 1) {
            // Create tensor with specific shapes that might trigger edge cases
            uint8_t shape_selector = consume<uint8_t>(current_data, remaining);
            switch (shape_selector % 8) {
                case 0: input = torch::randn({}); break; // scalar
                case 1: input = torch::randn({1}); break; // single element
                case 2: input = torch::randn({0}); break; // empty
                case 3: input = torch::randn({1, 1, 1}); break; // all ones
                case 4: input = torch::randn({100}); break; // 1D
                case 5: input = torch::randn({10, 10}); break; // 2D square
                case 6: input = torch::randn({2, 3, 4, 5}); break; // 4D
                case 7: input = torch::randn({1000, 1}); break; // large 1D-like
            }
        } else {
            // Create tensor with extreme values
            uint8_t value_type = consume<uint8_t>(current_data, remaining);
            int64_t dim1 = 1 + (consume<uint8_t>(current_data, remaining) % 50);
            int64_t dim2 = 1 + (consume<uint8_t>(current_data, remaining) % 50);
            
            switch (value_type % 6) {
                case 0: input = torch::zeros({dim1, dim2}); break;
                case 1: input = torch::ones({dim1, dim2}) * 1e10; break;
                case 2: input = torch::ones({dim1, dim2}) * -1e10; break;
                case 3: input = torch::randn({dim1, dim2}) * 1000; break;
                case 4: input = torch::rand({dim1, dim2}) * 0.001; break;
                case 5: {
                    input = torch::randn({dim1, dim2});
                    // Add some NaN/Inf values
                    if (input.numel() > 0) {
                        input.view(-1)[0] = std::numeric_limits<float>::quiet_NaN();
                        if (input.numel() > 1) {
                            input.view(-1)[1] = std::numeric_limits<float>::infinity();
                        }
                    }
                    break;
                }
            }
        }

        // Ensure tensor is float type (required for quantization)
        if (input.scalar_type() != torch::kFloat && input.scalar_type() != torch::kDouble) {
            input = input.to(torch::kFloat);
        }

        // Parse number of bits for quantization
        int n_bits = 8; // default
        if (remaining >= 1) {
            uint8_t bits_selector = consume<uint8_t>(current_data, remaining);
            // Common quantization bit widths
            switch (bits_selector % 5) {
                case 0: n_bits = 1; break;
                case 1: n_bits = 4; break;
                case 2: n_bits = 8; break;
                case 3: n_bits = 16; break;
                case 4: n_bits = 32; break;
            }
        }

        // Parse whether to use unsigned quantization
        bool use_unsigned = false;
        if (remaining >= 1) {
            use_unsigned = consume<uint8_t>(current_data, remaining) % 2;
        }

        // Parse whether to preserve sparsity
        bool preserve_sparsity = false;
        if (remaining >= 1) {
            preserve_sparsity = consume<uint8_t>(current_data, remaining) % 2;
        }

        // Try different tensor manipulations before calling choose_qparams_optimized
        if (remaining >= 1) {
            uint8_t manipulation = consume<uint8_t>(current_data, remaining);
            switch (manipulation % 8) {
                case 0: break; // no manipulation
                case 1: 
                    if (input.dim() > 0) input = input.transpose(0, -1); 
                    break;
                case 2: 
                    input = input.contiguous(); 
                    break;
                case 3: 
                    if (input.numel() > 0) input = input.view(-1); 
                    break;
                case 4: 
                    if (input.dim() > 0) input = input.squeeze(); 
                    break;
                case 5: 
                    input = input.unsqueeze(0); 
                    break;
                case 6:
                    if (input.device().type() == torch::kCPU && torch::cuda::is_available()) {
                        input = input.cuda();
                    }
                    break;
                case 7:
                    // Make tensor non-contiguous
                    if (input.dim() >= 2 && input.size(0) > 1 && input.size(1) > 1) {
                        input = input.narrow(0, 0, input.size(0) - 1);
                    }
                    break;
            }
        }

        // Call the target function
        try {
            auto [scale, zero_point] = torch::choose_qparams_optimized(
                input, 
                n_bits, 
                use_unsigned, 
                preserve_sparsity
            );

            // Verify outputs are valid
            if (!scale.defined() || !zero_point.defined()) {
                std::cerr << "Undefined output from choose_qparams_optimized" << std::endl;
                return -1;
            }

            // Additional operations to increase coverage
            if (scale.numel() > 0 && zero_point.numel() > 0) {
                // Try to use the quantization parameters
                int qmin = use_unsigned ? 0 : -(1 << (n_bits - 1));
                int qmax = use_unsigned ? (1 << n_bits) - 1 : (1 << (n_bits - 1)) - 1;
                
                // Attempt fake quantization with the chosen parameters
                if (input.numel() > 0 && scale.numel() > 0) {
                    try {
                        auto quantized = torch::fake_quantize_per_tensor_affine(
                            input,
                            scale.item<double>(),
                            zero_point.item<int64_t>(),
                            qmin,
                            qmax
                        );
                        
                        // Verify quantized tensor
                        if (!quantized.defined() || quantized.numel() != input.numel()) {
                            std::cerr << "Invalid quantized tensor" << std::endl;
                        }
                    } catch (...) {
                        // Fake quantization might fail for edge cases, that's ok
                    }
                }
            }

        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
            return 0;
        } catch (const std::runtime_error& e) {
            // Runtime errors from the operation itself
            return 0;
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}