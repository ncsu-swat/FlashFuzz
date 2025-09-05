#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 10) {
        // Need minimum bytes for basic parameters
        return 0;
    }

    try
    {
        size_t offset = 0;

        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(data, size, offset);
        
        // Skip if we don't have enough data left for parameters
        if (offset + 5 > size) {
            return 0;
        }

        // Parse quantile mode: scalar or tensor
        bool use_scalar_q = (offset < size) ? (data[offset++] % 2 == 0) : true;
        
        torch::Tensor q_tensor;
        float q_scalar = 0.5f;
        
        if (use_scalar_q) {
            // Parse scalar quantile value [0, 1]
            if (offset < size) {
                q_scalar = static_cast<float>(data[offset++]) / 255.0f;
            }
        } else {
            // Create a 1D tensor of quantiles
            if (offset < size) {
                uint8_t num_quantiles = (data[offset++] % 10) + 1; // 1-10 quantiles
                std::vector<float> q_values;
                q_values.reserve(num_quantiles);
                
                for (uint8_t i = 0; i < num_quantiles; ++i) {
                    float val = 0.5f; // default
                    if (offset < size) {
                        val = static_cast<float>(data[offset++]) / 255.0f;
                    }
                    q_values.push_back(val);
                }
                q_tensor = torch::tensor(q_values);
            } else {
                q_tensor = torch::tensor({0.25f, 0.5f, 0.75f}); // default quantiles
            }
        }

        // Parse dimension (optional)
        bool use_dim = (offset < size) ? (data[offset++] % 2 == 0) : false;
        c10::optional<int64_t> dim = c10::nullopt;
        
        if (use_dim && input.dim() > 0) {
            if (offset < size) {
                // Valid dimension range: [-input.dim(), input.dim()-1]
                int64_t raw_dim = static_cast<int64_t>(data[offset++]);
                dim = raw_dim % input.dim();
                if (dim.value() < 0) {
                    dim = dim.value() + input.dim();
                }
            }
        }

        // Parse keepdim
        bool keepdim = (offset < size) ? (data[offset++] % 2 == 0) : false;

        // Parse interpolation method
        std::string interpolation = "linear";
        if (offset < size) {
            uint8_t interp_selector = data[offset++] % 5;
            switch (interp_selector) {
                case 0: interpolation = "linear"; break;
                case 1: interpolation = "lower"; break;
                case 2: interpolation = "higher"; break;
                case 3: interpolation = "midpoint"; break;
                case 4: interpolation = "nearest"; break;
            }
        }

        // Call torch.quantile with different parameter combinations
        torch::Tensor result;
        
        try {
            if (use_scalar_q) {
                // Scalar quantile version
                if (dim.has_value()) {
                    result = torch::quantile(input, q_scalar, dim.value(), keepdim, interpolation);
                } else {
                    // No dim specified - flattens the tensor
                    result = torch::quantile(input, q_scalar, c10::nullopt, keepdim, interpolation);
                }
            } else {
                // Tensor quantile version
                if (dim.has_value()) {
                    result = torch::quantile(input, q_tensor, dim.value(), keepdim, interpolation);
                } else {
                    // No dim specified - flattens the tensor
                    result = torch::quantile(input, q_tensor, c10::nullopt, keepdim, interpolation);
                }
            }

            // Additional operations to increase coverage
            if (result.numel() > 0) {
                // Test some properties of the result
                auto min_val = result.min();
                auto max_val = result.max();
                
                // For non-empty input, quantiles should be within input range
                if (input.numel() > 0) {
                    auto input_min = input.min();
                    auto input_max = input.max();
                    
                    // Basic sanity check (may not hold for all interpolation methods exactly)
                    if (min_val.item<float>() < input_min.item<float>() - 1e-5 ||
                        max_val.item<float>() > input_max.item<float>() + 1e-5) {
                        // This could indicate an issue but might be expected for some edge cases
                    }
                }
            }

            // Test edge cases with special values if input contains them
            if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
                // Try quantile on tensor with NaN/Inf values
                if (offset + 2 < size && data[offset] % 10 == 0) {
                    torch::Tensor special_input = input.clone();
                    if (special_input.numel() > 0) {
                        special_input.view(-1)[0] = std::numeric_limits<float>::quiet_NaN();
                        if (special_input.numel() > 1) {
                            special_input.view(-1)[1] = std::numeric_limits<float>::infinity();
                        }
                        
                        try {
                            torch::Tensor special_result;
                            if (use_scalar_q) {
                                special_result = torch::quantile(special_input, q_scalar, dim, keepdim, interpolation);
                            } else {
                                special_result = torch::quantile(special_input, q_tensor, dim, keepdim, interpolation);
                            }
                        } catch (const c10::Error& e) {
                            // Expected for some cases with NaN/Inf
                        }
                    }
                }
            }

            // Test with out parameter
            if (offset + 1 < size && data[offset] % 5 == 0) {
                try {
                    // Pre-allocate output tensor with correct shape
                    torch::Tensor out_tensor;
                    if (use_scalar_q) {
                        // For scalar q, determine output shape
                        if (dim.has_value()) {
                            auto out_shape = input.sizes().vec();
                            if (!keepdim) {
                                out_shape.erase(out_shape.begin() + dim.value());
                            } else {
                                out_shape[dim.value()] = 1;
                            }
                            out_tensor = torch::empty(out_shape, input.options());
                        } else {
                            out_tensor = torch::empty({}, input.options());
                        }
                        torch::quantile_out(out_tensor, input, q_scalar, dim, keepdim, interpolation);
                    } else {
                        // For tensor q, first dimension is quantiles
                        if (dim.has_value()) {
                            auto out_shape = input.sizes().vec();
                            if (!keepdim) {
                                out_shape.erase(out_shape.begin() + dim.value());
                            } else {
                                out_shape[dim.value()] = 1;
                            }
                            out_shape.insert(out_shape.begin(), q_tensor.size(0));
                            out_tensor = torch::empty(out_shape, input.options());
                        } else {
                            out_tensor = torch::empty({q_tensor.size(0)}, input.options());
                        }
                        torch::quantile_out(out_tensor, input, q_tensor, dim, keepdim, interpolation);
                    }
                } catch (const c10::Error& e) {
                    // Output tensor shape mismatch or other issues
                }
            }

        } catch (const c10::Error& e) {
            // PyTorch errors are expected for invalid inputs
            // Continue fuzzing
        }

    }
    catch (const std::exception &e)
    {
        // Unexpected exceptions from fuzzer utils or other sources
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}