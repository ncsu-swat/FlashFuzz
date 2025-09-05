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
        auto consumeBytes = [&](size_t n) -> std::vector<uint8_t> {
            if (offset + n > size) {
                n = (size > offset) ? (size - offset) : 0;
            }
            std::vector<uint8_t> result(data + offset, data + offset + n);
            offset += n;
            return result;
        };
        
        auto consumeByte = [&]() -> uint8_t {
            if (offset >= size) return 0;
            return data[offset++];
        };
        
        auto consumeInt = [&]() -> int {
            if (offset + sizeof(int) > size) return 1;
            int val;
            std::memcpy(&val, data + offset, sizeof(int));
            offset += sizeof(int);
            return (std::abs(val) % 100) + 1;  // Bounded positive value
        };
        
        auto consumeFloat = [&]() -> float {
            if (offset + sizeof(float) > size) return 1.0f;
            float val;
            std::memcpy(&val, data + offset, sizeof(float));
            offset += sizeof(float);
            return val;
        };

        // Determine tensor configuration for input
        uint8_t config1 = consumeByte();
        bool use_scalar_input = (config1 & 1);
        uint8_t dtype_selector1 = (config1 >> 1) & 7;
        
        torch::Tensor input;
        
        if (use_scalar_input) {
            // Create scalar tensor
            float scalar_val = consumeFloat();
            input = torch::tensor(scalar_val);
        } else {
            // Create tensor with shape
            int ndims = (consumeByte() % 4) + 1;
            std::vector<int64_t> shape;
            int64_t total_elements = 1;
            
            for (int i = 0; i < ndims; i++) {
                int dim_size = consumeInt() % 10;  // Keep dimensions small
                shape.push_back(dim_size);
                total_elements *= dim_size;
                if (total_elements > 1000) break;  // Prevent excessive memory
            }
            
            // Select dtype for input
            torch::ScalarType dtype;
            switch (dtype_selector1) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
                case 4: dtype = torch::kInt8; break;
                case 5: dtype = torch::kUInt8; break;
                case 6: dtype = torch::kInt16; break;
                default: dtype = torch::kFloat32; break;
            }
            
            // Create input tensor
            if (total_elements == 0) {
                input = torch::empty(shape, torch::dtype(dtype));
            } else {
                input = torch::randn(shape, torch::dtype(dtype));
                
                // Fill with fuzzed data
                if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                    auto input_accessor = input.flatten();
                    for (int64_t i = 0; i < input_accessor.numel() && offset < size; i++) {
                        float val = consumeFloat();
                        // Include special values occasionally
                        uint8_t special = consumeByte();
                        if (special < 5) val = 0.0f;
                        else if (special < 10) val = std::numeric_limits<float>::infinity();
                        else if (special < 15) val = -std::numeric_limits<float>::infinity();
                        else if (special < 20) val = std::numeric_limits<float>::quiet_NaN();
                        
                        input_accessor[i] = val;
                    }
                } else {
                    auto input_accessor = input.flatten();
                    for (int64_t i = 0; i < input_accessor.numel() && offset < size; i++) {
                        input_accessor[i] = static_cast<int>(consumeInt());
                    }
                }
            }
        }
        
        // Determine configuration for other
        uint8_t config2 = consumeByte();
        bool use_scalar_other = (config2 & 1);
        bool use_same_shape = (config2 & 2) && !use_scalar_input;
        uint8_t dtype_selector2 = (config2 >> 2) & 7;
        
        torch::Tensor other;
        
        if (use_scalar_other) {
            // Create scalar for other
            float scalar_val = consumeFloat();
            // Occasionally use zero or near-zero divisor
            uint8_t zero_chance = consumeByte();
            if (zero_chance < 30) scalar_val = 0.0f;
            else if (zero_chance < 60) scalar_val = 1e-10f;
            
            other = torch::tensor(scalar_val);
        } else {
            // Create tensor for other
            std::vector<int64_t> other_shape;
            
            if (use_same_shape && input.dim() > 0) {
                other_shape = input.sizes().vec();
            } else {
                // Create potentially broadcastable shape
                int ndims = (consumeByte() % 4) + 1;
                int64_t total_elements = 1;
                
                for (int i = 0; i < ndims; i++) {
                    int dim_size = consumeInt() % 10;
                    // Occasionally use size 1 for broadcasting
                    if (consumeByte() < 50) dim_size = 1;
                    other_shape.push_back(dim_size);
                    total_elements *= dim_size;
                    if (total_elements > 1000) break;
                }
            }
            
            // Select dtype for other
            torch::ScalarType dtype;
            switch (dtype_selector2) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
                case 4: dtype = torch::kInt8; break;
                case 5: dtype = torch::kUInt8; break;
                case 6: dtype = torch::kInt16; break;
                default: dtype = torch::kFloat32; break;
            }
            
            // Create other tensor
            int64_t total_other = 1;
            for (auto d : other_shape) total_other *= d;
            
            if (total_other == 0) {
                other = torch::empty(other_shape, torch::dtype(dtype));
            } else {
                other = torch::randn(other_shape, torch::dtype(dtype));
                
                // Fill with fuzzed data, including zeros
                if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                    auto other_accessor = other.flatten();
                    for (int64_t i = 0; i < other_accessor.numel() && offset < size; i++) {
                        float val = consumeFloat();
                        // Higher chance of zeros and special values for divisor
                        uint8_t special = consumeByte();
                        if (special < 40) val = 0.0f;
                        else if (special < 50) val = 1e-10f;
                        else if (special < 55) val = -1e-10f;
                        else if (special < 60) val = std::numeric_limits<float>::infinity();
                        else if (special < 65) val = -std::numeric_limits<float>::infinity();
                        else if (special < 70) val = std::numeric_limits<float>::quiet_NaN();
                        
                        other_accessor[i] = val;
                    }
                } else {
                    auto other_accessor = other.flatten();
                    for (int64_t i = 0; i < other_accessor.numel() && offset < size; i++) {
                        int val = consumeInt();
                        // Include zeros
                        if (consumeByte() < 40) val = 0;
                        other_accessor[i] = val;
                    }
                }
            }
        }
        
        // Decide whether to use out parameter
        bool use_out = consumeByte() < 128;
        
        if (use_out) {
            // Create output tensor
            torch::Tensor out;
            
            // Try to infer output shape (broadcasting rules)
            try {
                auto dummy_result = torch::empty_like(torch::add(input, other));
                out = torch::empty_like(dummy_result);
                
                // Call remainder with out parameter
                torch::remainder_out(out, input, other);
            } catch (...) {
                // If shape inference fails, try without out
                torch::remainder(input, other);
            }
        } else {
            // Call remainder without out parameter
            torch::Tensor result = torch::remainder(input, other);
            
            // Optionally perform some operations on result to trigger more code paths
            if (consumeByte() < 100) {
                auto sum = result.sum();
                auto mean = result.mean();
            }
        }
        
        // Try scalar versions
        if (consumeByte() < 50 && input.numel() > 0) {
            float scalar_divisor = consumeFloat();
            if (consumeByte() < 30) scalar_divisor = 0.0f;  // Test division by zero
            
            torch::Tensor scalar_result = torch::remainder(input, scalar_divisor);
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid operations
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}