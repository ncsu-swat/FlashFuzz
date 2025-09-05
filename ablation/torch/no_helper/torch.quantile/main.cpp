#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 16) {
        return 0;  // Need minimum bytes for basic parameters
    }

    try
    {
        size_t offset = 0;
        
        // Helper to consume bytes
        auto consumeBytes = [&](size_t num_bytes) -> std::vector<uint8_t> {
            if (offset + num_bytes > size) {
                return std::vector<uint8_t>(num_bytes, 0);
            }
            std::vector<uint8_t> result(data + offset, data + offset + num_bytes);
            offset += num_bytes;
            return result;
        };
        
        auto consumeUInt8 = [&]() -> uint8_t {
            if (offset >= size) return 0;
            return data[offset++];
        };
        
        auto consumeInt64 = [&]() -> int64_t {
            if (offset + sizeof(int64_t) > size) return 1;
            int64_t val;
            std::memcpy(&val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            return (std::abs(val) % 100) + 1;  // Bound dimensions
        };
        
        auto consumeFloat = [&]() -> float {
            if (offset + sizeof(float) > size) return 0.5f;
            float val;
            std::memcpy(&val, data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp to [0, 1] for quantile values
            if (std::isnan(val) || std::isinf(val)) return 0.5f;
            return std::max(0.0f, std::min(1.0f, std::abs(val)));
        };
        
        // Determine input tensor properties
        uint8_t num_dims = (consumeUInt8() % 4) + 1;  // 1-4 dimensions
        std::vector<int64_t> shape;
        for (size_t i = 0; i < num_dims; ++i) {
            shape.push_back(consumeInt64());
        }
        
        // Choose dtype for input tensor
        uint8_t dtype_choice = consumeUInt8() % 3;
        torch::Tensor input;
        
        if (dtype_choice == 0) {
            input = torch::randn(shape, torch::kFloat32);
        } else if (dtype_choice == 1) {
            input = torch::randn(shape, torch::kFloat64);
        } else {
            input = torch::randn(shape, torch::kFloat16);
        }
        
        // Add some variation to input values
        if (consumeUInt8() % 2 == 0) {
            float scale = consumeFloat() * 10.0f;
            input = input * scale;
        }
        
        // Determine q parameter (quantile values)
        uint8_t q_type = consumeUInt8() % 3;
        torch::Tensor q_tensor;
        float q_scalar;
        bool use_tensor_q = false;
        
        if (q_type == 0) {
            // Scalar q
            q_scalar = consumeFloat();
        } else if (q_type == 1) {
            // 1D tensor q with random values
            int64_t q_size = (consumeInt64() % 10) + 1;
            std::vector<float> q_values;
            for (int64_t i = 0; i < q_size; ++i) {
                q_values.push_back(consumeFloat());
            }
            q_tensor = torch::tensor(q_values);
            use_tensor_q = true;
        } else {
            // 1D tensor q with edge case values
            std::vector<float> q_values = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
            if (consumeUInt8() % 2 == 0) {
                q_values.push_back(consumeFloat());
            }
            q_tensor = torch::tensor(q_values);
            use_tensor_q = true;
        }
        
        // Determine dim parameter
        bool has_dim = consumeUInt8() % 2 == 0;
        int64_t dim = -1;
        if (has_dim && num_dims > 0) {
            dim = consumeInt64() % num_dims;
            if (consumeUInt8() % 2 == 0) {
                dim = -dim - 1;  // Test negative dimensions
            }
        }
        
        // Determine keepdim
        bool keepdim = consumeUInt8() % 2 == 0;
        
        // Determine interpolation method
        uint8_t interp_choice = consumeUInt8() % 5;
        std::string interpolation;
        switch (interp_choice) {
            case 0: interpolation = "linear"; break;
            case 1: interpolation = "lower"; break;
            case 2: interpolation = "higher"; break;
            case 3: interpolation = "midpoint"; break;
            case 4: interpolation = "nearest"; break;
            default: interpolation = "linear"; break;
        }
        
        // Call torch.quantile with various parameter combinations
        torch::Tensor result;
        
        if (use_tensor_q) {
            if (has_dim) {
                result = torch::quantile(input, q_tensor, dim, keepdim, interpolation);
            } else {
                result = torch::quantile(input, q_tensor, c10::nullopt, keepdim, interpolation);
            }
        } else {
            if (has_dim) {
                result = torch::quantile(input, q_scalar, dim, keepdim, interpolation);
            } else {
                result = torch::quantile(input, q_scalar, c10::nullopt, keepdim, interpolation);
            }
        }
        
        // Test with output tensor
        if (consumeUInt8() % 3 == 0) {
            torch::Tensor out;
            if (use_tensor_q) {
                if (has_dim) {
                    // Pre-allocate output tensor with correct shape
                    std::vector<int64_t> out_shape;
                    out_shape.push_back(q_tensor.size(0));
                    for (int64_t i = 0; i < input.dim(); ++i) {
                        if (i != dim || keepdim) {
                            out_shape.push_back(i == dim && keepdim ? 1 : input.size(i));
                        }
                    }
                    out = torch::empty(out_shape, input.options());
                    torch::quantile_out(out, input, q_tensor, dim, keepdim, interpolation);
                } else {
                    out = torch::empty({q_tensor.size(0)}, input.options());
                    torch::quantile_out(out, input, q_tensor, c10::nullopt, keepdim, interpolation);
                }
            } else {
                if (has_dim) {
                    std::vector<int64_t> out_shape;
                    for (int64_t i = 0; i < input.dim(); ++i) {
                        if (i != dim || keepdim) {
                            out_shape.push_back(i == dim && keepdim ? 1 : input.size(i));
                        }
                    }
                    if (out_shape.empty()) {
                        out = torch::empty({}, input.options());
                    } else {
                        out = torch::empty(out_shape, input.options());
                    }
                    torch::quantile_out(out, input, q_scalar, dim, keepdim, interpolation);
                } else {
                    out = torch::empty({}, input.options());
                    torch::quantile_out(out, input, q_scalar, c10::nullopt, keepdim, interpolation);
                }
            }
        }
        
        // Test edge cases with empty tensors
        if (consumeUInt8() % 10 == 0) {
            torch::Tensor empty_input = torch::empty({0});
            torch::Tensor empty_result = torch::quantile(empty_input, 0.5f);
        }
        
        // Test with special values
        if (consumeUInt8() % 10 == 0) {
            torch::Tensor special_input = torch::tensor({std::numeric_limits<float>::infinity(),
                                                         -std::numeric_limits<float>::infinity(),
                                                         std::numeric_limits<float>::quiet_NaN()});
            torch::Tensor special_result = torch::quantile(special_input, 0.5f);
        }
    }
    catch (const c10::Error &e)
    {
        // PyTorch errors are expected for invalid inputs
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}