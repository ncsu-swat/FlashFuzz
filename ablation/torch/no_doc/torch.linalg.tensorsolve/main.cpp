#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes for various parameters
template<typename T>
T consumeValue(const uint8_t* data, size_t& offset, size_t size, T min_val, T max_val) {
    if (offset + sizeof(T) > size) {
        offset = size;
        return min_val;
    }
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    
    // Map to range [min_val, max_val]
    if (max_val > min_val) {
        value = min_val + std::abs(value) % (max_val - min_val + 1);
    } else {
        value = min_val;
    }
    return value;
}

// Create tensor with controlled properties for tensorsolve
torch::Tensor createTensorsolveCompatibleTensor(const uint8_t* data, size_t& offset, size_t size, 
                                                const std::vector<int64_t>& required_shape,
                                                torch::ScalarType dtype) {
    try {
        // Calculate total elements
        int64_t num_elements = 1;
        for (auto dim : required_shape) {
            num_elements *= dim;
        }
        
        // Parse tensor data
        size_t dtype_size = c10::elementSize(dtype);
        auto tensor_data = fuzzer_utils::parseTensorData(data, offset, size, num_elements, dtype_size);
        
        // Create tensor from blob
        auto options = torch::TensorOptions().dtype(dtype);
        if (num_elements == 0) {
            return torch::empty(required_shape, options);
        }
        
        return torch::from_blob(tensor_data.data(), required_shape, options).clone();
    } catch (...) {
        // Fallback to random tensor
        auto options = torch::TensorOptions().dtype(dtype);
        return torch::randn(required_shape, options);
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) {
        return 0;  // Need minimum bytes for configuration
    }
    
    try {
        size_t offset = 0;
        
        // Parse dtype for both tensors
        uint8_t dtype_selector = consumeValue<uint8_t>(data, offset, size, 0, 255);
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Skip non-floating point types for linalg operations
        if (dtype != torch::kFloat && dtype != torch::kDouble && 
            dtype != torch::kComplexFloat && dtype != torch::kComplexDouble) {
            dtype = torch::kFloat;
        }
        
        // Parse dimensions for the solution tensor x
        // tensorsolve requires: a.shape = a_shape_prefix + x.shape + x.shape
        // where we solve a @ x = b
        
        // Generate x_shape dimensions
        uint8_t x_ndim = consumeValue<uint8_t>(data, offset, size, 1, 3);
        std::vector<int64_t> x_shape;
        for (int i = 0; i < x_ndim; i++) {
            int64_t dim = consumeValue<int64_t>(data, offset, size, 1, 5);
            x_shape.push_back(dim);
        }
        
        // Generate a_shape_prefix dimensions (can be empty)
        uint8_t prefix_ndim = consumeValue<uint8_t>(data, offset, size, 0, 2);
        std::vector<int64_t> a_shape_prefix;
        for (int i = 0; i < prefix_ndim; i++) {
            int64_t dim = consumeValue<int64_t>(data, offset, size, 1, 4);
            a_shape_prefix.push_back(dim);
        }
        
        // Construct full a_shape: prefix + x_shape + x_shape
        std::vector<int64_t> a_shape = a_shape_prefix;
        a_shape.insert(a_shape.end(), x_shape.begin(), x_shape.end());
        a_shape.insert(a_shape.end(), x_shape.begin(), x_shape.end());
        
        // Construct b_shape: prefix + x_shape
        std::vector<int64_t> b_shape = a_shape_prefix;
        b_shape.insert(b_shape.end(), x_shape.begin(), x_shape.end());
        
        // Create tensor a
        torch::Tensor a = createTensorsolveCompatibleTensor(data, offset, size, a_shape, dtype);
        
        // Create tensor b
        torch::Tensor b = createTensorsolveCompatibleTensor(data, offset, size, b_shape, dtype);
        
        // Parse optional dims parameter
        bool use_dims = consumeValue<uint8_t>(data, offset, size, 0, 1);
        
        // Call tensorsolve with different parameter combinations
        torch::Tensor result;
        
        if (use_dims && offset < size) {
            // Parse dims value - it should be <= x_ndim
            int64_t dims_val = consumeValue<int64_t>(data, offset, size, 0, x_ndim);
            
            // Try with explicit dims parameter
            try {
                result = torch::linalg::tensorsolve(a, b, dims_val);
            } catch (const c10::Error& e) {
                // Try without dims on error
                result = torch::linalg::tensorsolve(a, b);
            }
        } else {
            // Call without dims parameter (uses default)
            result = torch::linalg::tensorsolve(a, b);
        }
        
        // Verify result properties
        if (result.defined()) {
            // Check shape matches expected x_shape
            if (result.sizes() != torch::IntArrayRef(x_shape)) {
                // Shape mismatch but continue
            }
            
            // Verify solution by computing a @ result and comparing with b
            try {
                // Reshape result for matrix multiplication if needed
                torch::Tensor verification = torch::tensordot(a, result, x_ndim);
                
                // Check if close to b
                if (b.dtype().isFloatingPoint()) {
                    bool close = torch::allclose(verification, b, 1e-3, 1e-5);
                }
            } catch (...) {
                // Verification failed, but continue
            }
            
            // Test edge cases
            if (offset < size) {
                uint8_t edge_case = consumeValue<uint8_t>(data, offset, size, 0, 3);
                switch (edge_case) {
                    case 0:
                        // Try with transposed a
                        try {
                            auto a_t = a.transpose(-2, -1);
                            torch::linalg::tensorsolve(a_t, b);
                        } catch (...) {}
                        break;
                    case 1:
                        // Try with conjugate (for complex)
                        if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
                            try {
                                auto a_conj = a.conj();
                                torch::linalg::tensorsolve(a_conj, b);
                            } catch (...) {}
                        }
                        break;
                    case 2:
                        // Try with non-contiguous tensors
                        try {
                            auto a_nc = a.transpose(0, -1);
                            auto b_nc = b.transpose(0, -1);
                            torch::linalg::tensorsolve(a_nc, b_nc);
                        } catch (...) {}
                        break;
                    case 3:
                        // Try with views
                        try {
                            auto a_view = a.view(a.sizes());
                            auto b_view = b.view(b.sizes());
                            torch::linalg::tensorsolve(a_view, b_view);
                        } catch (...) {}
                        break;
                }
            }
        }
        
        // Additional tests with malformed inputs
        if (offset < size) {
            uint8_t malform_type = consumeValue<uint8_t>(data, offset, size, 0, 4);
            try {
                switch (malform_type) {
                    case 0:
                        // Mismatched shapes
                        {
                            auto bad_b = torch::randn({2, 3}, torch::TensorOptions().dtype(dtype));
                            torch::linalg::tensorsolve(a, bad_b);
                        }
                        break;
                    case 1:
                        // Empty tensors
                        {
                            auto empty_a = torch::empty({0, 0}, torch::TensorOptions().dtype(dtype));
                            auto empty_b = torch::empty({0}, torch::TensorOptions().dtype(dtype));
                            torch::linalg::tensorsolve(empty_a, empty_b);
                        }
                        break;
                    case 2:
                        // Singular matrix (may not have solution)
                        {
                            auto singular_a = torch::zeros(a.sizes(), torch::TensorOptions().dtype(dtype));
                            torch::linalg::tensorsolve(singular_a, b);
                        }
                        break;
                    case 3:
                        // NaN/Inf values
                        {
                            auto nan_a = a.clone();
                            nan_a[0] = std::numeric_limits<float>::quiet_NaN();
                            torch::linalg::tensorsolve(nan_a, b);
                        }
                        break;
                    case 4:
                        // Different dtypes (should fail or convert)
                        if (dtype != torch::kDouble) {
                            try {
                                auto double_b = b.to(torch::kDouble);
                                torch::linalg::tensorsolve(a, double_b);
                            } catch (...) {}
                        }
                        break;
                }
            } catch (const c10::Error& e) {
                // Expected for malformed inputs
            } catch (const std::exception& e) {
                // Other exceptions from malformed inputs
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch errors are expected for invalid inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}