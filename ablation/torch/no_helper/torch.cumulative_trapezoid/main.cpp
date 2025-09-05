#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 16) {
        return 0;  // Need minimum bytes for configuration
    }

    try {
        size_t offset = 0;
        
        // Helper lambda to consume bytes
        auto consume_bytes = [&](size_t num_bytes) -> std::vector<uint8_t> {
            if (offset + num_bytes > size) {
                return std::vector<uint8_t>(num_bytes, 0);
            }
            std::vector<uint8_t> result(data + offset, data + offset + num_bytes);
            offset += num_bytes;
            return result;
        };
        
        auto consume_byte = [&]() -> uint8_t {
            if (offset >= size) return 0;
            return data[offset++];
        };
        
        auto consume_int = [&]() -> int {
            if (offset + sizeof(int) > size) return 1;
            int val;
            std::memcpy(&val, data + offset, sizeof(int));
            offset += sizeof(int);
            return val;
        };
        
        auto consume_float = [&]() -> float {
            if (offset + sizeof(float) > size) return 1.0f;
            float val;
            std::memcpy(&val, data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize to avoid NaN/Inf issues
            if (!std::isfinite(val)) val = 1.0f;
            return val;
        };

        // Configuration bytes
        uint8_t config = consume_byte();
        bool use_x_tensor = (config & 0x01);
        bool use_dx = (config & 0x02) && !use_x_tensor;  // dx and x are mutually exclusive
        bool use_float_dtype = (config & 0x04);
        uint8_t num_dims = (consume_byte() % 4) + 1;  // 1-4 dimensions
        
        // Build tensor shape
        std::vector<int64_t> shape;
        for (int i = 0; i < num_dims; ++i) {
            int dim_size = (consume_byte() % 10) + 1;  // 1-10 per dimension
            shape.push_back(dim_size);
        }
        
        // Determine dim parameter
        int dim = consume_int() % num_dims;
        if (consume_byte() & 0x01) {
            dim = -dim - 1;  // Test negative indexing
        }
        
        // Calculate total elements needed
        int64_t total_elements = 1;
        for (auto s : shape) {
            total_elements *= s;
        }
        
        // Create y tensor
        torch::Tensor y;
        if (use_float_dtype) {
            std::vector<float> y_data;
            for (int64_t i = 0; i < total_elements; ++i) {
                y_data.push_back(consume_float());
            }
            y = torch::from_blob(y_data.data(), shape, torch::kFloat32).clone();
        } else {
            std::vector<double> y_data;
            for (int64_t i = 0; i < total_elements; ++i) {
                y_data.push_back(static_cast<double>(consume_float()));
            }
            y = torch::from_blob(y_data.data(), shape, torch::kFloat64).clone();
        }
        
        // Test with different memory layouts
        if (consume_byte() & 0x01) {
            y = y.contiguous();
        } else if (consume_byte() & 0x01) {
            y = y.transpose(0, -1);  // Create non-contiguous tensor
        }
        
        torch::Tensor result;
        
        if (use_x_tensor) {
            // Create x tensor with same shape as y
            torch::Tensor x;
            if (use_float_dtype) {
                std::vector<float> x_data;
                for (int64_t i = 0; i < total_elements; ++i) {
                    x_data.push_back(consume_float());
                }
                x = torch::from_blob(x_data.data(), shape, torch::kFloat32).clone();
            } else {
                std::vector<double> x_data;
                for (int64_t i = 0; i < total_elements; ++i) {
                    x_data.push_back(static_cast<double>(consume_float()));
                }
                x = torch::from_blob(x_data.data(), shape, torch::kFloat64).clone();
            }
            
            // Call with x tensor
            result = torch::cumulative_trapezoid(y, x, dim);
        } else if (use_dx) {
            // Call with dx scalar
            double dx = static_cast<double>(consume_float());
            result = torch::cumulative_trapezoid(y, c10::nullopt, dx, dim);
        } else {
            // Call with default spacing
            result = torch::cumulative_trapezoid(y, c10::nullopt, c10::nullopt, dim);
        }
        
        // Perform some basic operations to ensure result is valid
        if (result.defined()) {
            auto sum = result.sum();
            auto mean = result.mean();
            
            // Test edge cases with the result
            if (consume_byte() & 0x01) {
                auto reshaped = result.reshape({-1});
            }
            if (consume_byte() & 0x01) {
                auto transposed = result.transpose(0, -1);
            }
        }
        
        // Test special cases
        if (offset < size && (consume_byte() & 0x01)) {
            // Test with empty tensor
            torch::Tensor empty_y = torch::empty({0});
            try {
                auto empty_result = torch::cumulative_trapezoid(empty_y);
            } catch (...) {
                // Expected to fail for some configurations
            }
        }
        
        if (offset < size && (consume_byte() & 0x01)) {
            // Test with single element tensor
            torch::Tensor single = torch::ones({1});
            try {
                auto single_result = torch::cumulative_trapezoid(single);
            } catch (...) {
                // Expected to fail for some configurations
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}