#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    if (Size < 10) {
        // Need minimum bytes for basic parsing
        return 0;
    }

    try
    {
        size_t offset = 0;

        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip if we don't have enough data left
        if (offset + 5 > Size) {
            return 0;
        }

        // Parse quantile values
        uint8_t quantile_type = Data[offset++];
        torch::Tensor q;
        
        if (quantile_type % 3 == 0) {
            // Single scalar quantile
            float q_val = static_cast<float>(Data[offset++]) / 255.0f; // Normalize to [0, 1]
            q = torch::tensor(q_val);
        } else if (quantile_type % 3 == 1) {
            // Multiple quantiles
            uint8_t num_quantiles = (Data[offset++] % 10) + 1; // 1-10 quantiles
            std::vector<float> q_values;
            for (int i = 0; i < num_quantiles && offset < Size; i++) {
                float val = static_cast<float>(Data[offset++]) / 255.0f;
                q_values.push_back(val);
            }
            q = torch::tensor(q_values);
        } else {
            // Edge case: empty quantiles tensor or extreme values
            if (Data[offset] % 4 == 0) {
                q = torch::tensor({}); // Empty tensor
                offset++;
            } else if (Data[offset] % 4 == 1) {
                q = torch::tensor({0.0f, 1.0f}); // Boundary values
                offset++;
            } else if (Data[offset] % 4 == 2) {
                q = torch::tensor({0.5f}); // Median
                offset++;
            } else {
                // Random tensor of quantiles
                uint8_t shape_val = Data[offset++];
                int64_t q_size = (shape_val % 5) + 1;
                q = torch::rand({q_size});
                q = q.clamp(0.0, 1.0); // Ensure values are in [0, 1]
            }
        }

        // Parse optional dimension
        c10::optional<int64_t> dim;
        bool has_dim = false;
        if (offset < Size) {
            uint8_t dim_flag = Data[offset++];
            if (dim_flag % 2 == 0) {
                has_dim = true;
                if (offset < Size) {
                    int64_t rank = input.dim();
                    if (rank > 0) {
                        // Parse dimension, handle negative indexing
                        int8_t dim_val = static_cast<int8_t>(Data[offset++]);
                        int64_t actual_dim = dim_val % rank;
                        if (actual_dim < 0) {
                            actual_dim += rank;
                        }
                        dim = actual_dim;
                    } else {
                        // Scalar tensor, no valid dimension
                        has_dim = false;
                    }
                }
            }
        }

        // Parse keepdim flag
        bool keepdim = false;
        if (offset < Size) {
            keepdim = (Data[offset++] % 2) == 0;
        }

        // Parse interpolation method
        std::string interpolation = "linear";
        if (offset < Size) {
            uint8_t interp_selector = Data[offset++];
            switch (interp_selector % 5) {
                case 0: interpolation = "linear"; break;
                case 1: interpolation = "lower"; break;
                case 2: interpolation = "higher"; break;
                case 3: interpolation = "midpoint"; break;
                case 4: interpolation = "nearest"; break;
            }
        }

        // Try different invocation patterns
        torch::Tensor result;
        
        if (offset < Size && Data[offset++] % 4 == 0) {
            // Test with out parameter
            torch::Tensor out;
            if (has_dim) {
                // Pre-allocate output tensor with appropriate shape
                auto expected_shape = input.sizes().vec();
                if (q.numel() > 0) {
                    if (q.dim() == 0) {
                        // Scalar quantile
                        if (keepdim) {
                            expected_shape[dim.value()] = 1;
                        } else {
                            expected_shape.erase(expected_shape.begin() + dim.value());
                        }
                    } else {
                        // Multiple quantiles
                        expected_shape[dim.value()] = q.numel();
                    }
                    out = torch::empty(expected_shape, input.options());
                    result = torch::quantile_out(out, input, q, dim, keepdim, interpolation);
                } else {
                    // Empty quantiles, skip out parameter test
                    result = torch::quantile(input, q, dim, keepdim, interpolation);
                }
            } else {
                // No dimension specified
                if (q.numel() > 0) {
                    std::vector<int64_t> out_shape;
                    if (q.dim() > 0) {
                        out_shape.push_back(q.numel());
                    }
                    out = torch::empty(out_shape, input.options());
                    result = torch::quantile_out(out, input, q, c10::nullopt, keepdim, interpolation);
                } else {
                    result = torch::quantile(input, q, c10::nullopt, keepdim, interpolation);
                }
            }
        } else {
            // Regular invocation
            if (has_dim) {
                result = torch::quantile(input, q, dim, keepdim, interpolation);
            } else {
                result = torch::quantile(input, q, c10::nullopt, keepdim, interpolation);
            }
        }

        // Additional edge case testing
        if (offset < Size && Data[offset++] % 3 == 0) {
            // Test with different tensor types
            if (input.is_floating_point()) {
                // Try converting to integer and back
                auto int_input = input.to(torch::kInt32);
                auto float_input = int_input.to(torch::kFloat32);
                auto result2 = torch::quantile(float_input, q, dim, keepdim, interpolation);
                
                // Test with NaN/Inf values if floating point
                if (offset < Size && Data[offset++] % 4 == 0) {
                    auto test_input = input.clone();
                    if (test_input.numel() > 0) {
                        test_input.view(-1)[0] = std::numeric_limits<float>::quiet_NaN();
                        if (test_input.numel() > 1) {
                            test_input.view(-1)[1] = std::numeric_limits<float>::infinity();
                        }
                        auto result3 = torch::quantile(test_input, q, dim, keepdim, interpolation);
                    }
                }
            }
        }

        // Test with strided tensors
        if (offset < Size && Data[offset++] % 3 == 0 && input.numel() > 1) {
            auto strided = input.as_strided({input.numel()/2}, {2});
            if (has_dim && dim.value() < strided.dim()) {
                auto result4 = torch::quantile(strided, q, dim, keepdim, interpolation);
            } else {
                auto result4 = torch::quantile(strided, q, c10::nullopt, keepdim, interpolation);
            }
        }

        // Test with transposed/permuted tensors
        if (input.dim() >= 2 && offset < Size && Data[offset++] % 2 == 0) {
            auto transposed = input.transpose(0, 1);
            if (has_dim) {
                // Adjust dimension after transpose
                int64_t new_dim = dim.value();
                if (new_dim == 0) new_dim = 1;
                else if (new_dim == 1) new_dim = 0;
                auto result5 = torch::quantile(transposed, q, new_dim, keepdim, interpolation);
            } else {
                auto result5 = torch::quantile(transposed, q, c10::nullopt, keepdim, interpolation);
            }
        }

        // Test with views
        if (input.numel() > 0 && offset < Size && Data[offset++] % 2 == 0) {
            auto view = input.view({-1});
            auto result6 = torch::quantile(view, q, 0, keepdim, interpolation);
        }

    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    }
    catch (const std::runtime_error &e)
    {
        // Runtime errors from tensor creation are expected
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}