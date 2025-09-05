#include <torch/torch.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <iostream>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t*& data, size_t& size, T& value) {
    if (size < sizeof(T)) return false;
    std::memcpy(&value, data, sizeof(T));
    data += sizeof(T);
    size -= sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    try {
        if (size < 16) return 0;  // Need minimum bytes for basic parameters
        
        const uint8_t* ptr = data;
        size_t remaining = size;
        
        // Consume configuration bytes
        uint8_t dtype_selector = 0;
        uint8_t use_dims = 0;
        uint8_t b_ndim = 0;
        uint8_t a_extra_ndim = 0;
        uint8_t use_out = 0;
        
        if (!consumeBytes(ptr, remaining, dtype_selector)) return 0;
        if (!consumeBytes(ptr, remaining, use_dims)) return 0;
        if (!consumeBytes(ptr, remaining, b_ndim)) return 0;
        if (!consumeBytes(ptr, remaining, a_extra_ndim)) return 0;
        if (!consumeBytes(ptr, remaining, use_out)) return 0;
        
        // Limit dimensions to reasonable values
        b_ndim = (b_ndim % 4) + 1;  // B.ndim in [1, 4]
        a_extra_ndim = (a_extra_ndim % 3) + 1;  // Extra dims for A in [1, 3]
        
        // Select dtype
        torch::ScalarType dtype;
        switch (dtype_selector % 4) {
            case 0: dtype = torch::kFloat; break;
            case 1: dtype = torch::kDouble; break;
            case 2: dtype = torch::kComplexFloat; break;
            case 3: dtype = torch::kComplexDouble; break;
            default: dtype = torch::kFloat;
        }
        
        // Build B shape
        std::vector<int64_t> b_shape;
        int64_t b_total_size = 1;
        for (int i = 0; i < b_ndim; i++) {
            uint8_t dim_size = 0;
            if (!consumeBytes(ptr, remaining, dim_size)) return 0;
            dim_size = (dim_size % 5) + 1;  // Dimension sizes in [1, 5]
            b_shape.push_back(dim_size);
            b_total_size *= dim_size;
        }
        
        // Build A shape: first b_ndim dimensions match B, then extra dimensions
        // For tensorsolve to work: prod(A.shape[:B.ndim]) == prod(A.shape[B.ndim:])
        std::vector<int64_t> a_shape = b_shape;  // First b_ndim dimensions
        
        // Add extra dimensions such that their product equals b_total_size
        std::vector<int64_t> extra_dims;
        int64_t extra_prod = 1;
        
        // Simple approach: try to factorize b_total_size
        if (b_total_size == 1) {
            extra_dims.push_back(1);
        } else if (b_total_size <= 6) {
            // For small values, just use the value itself
            extra_dims.push_back(b_total_size);
        } else {
            // Try to find factors
            int64_t remaining_prod = b_total_size;
            for (int i = 0; i < a_extra_ndim && remaining_prod > 1; i++) {
                uint8_t factor_selector = 0;
                if (!consumeBytes(ptr, remaining, factor_selector)) return 0;
                
                // Find a factor of remaining_prod
                int64_t factor = 1;
                if (remaining_prod == 2) factor = 2;
                else if (remaining_prod == 3) factor = 3;
                else if (remaining_prod % 2 == 0) factor = 2;
                else if (remaining_prod % 3 == 0) factor = 3;
                else factor = remaining_prod;
                
                extra_dims.push_back(factor);
                extra_prod *= factor;
                remaining_prod /= factor;
            }
            
            // Adjust last dimension to make product exact
            if (extra_prod != b_total_size && !extra_dims.empty()) {
                extra_dims.back() = extra_dims.back() * (b_total_size / extra_prod);
                extra_prod = b_total_size;
            }
        }
        
        // Ensure product constraint is satisfied
        if (extra_prod != b_total_size) {
            // Force it to work by adjusting
            extra_dims.clear();
            extra_dims.push_back(b_total_size);
        }
        
        // Append extra dimensions to A shape
        for (auto dim : extra_dims) {
            a_shape.push_back(dim);
        }
        
        // Create tensors with remaining fuzzer data
        size_t a_elements = 1;
        for (auto dim : a_shape) a_elements *= dim;
        
        size_t b_elements = b_total_size;
        
        // Create A tensor
        torch::Tensor A;
        if (dtype == torch::kFloat || dtype == torch::kDouble) {
            std::vector<float> a_data(a_elements);
            for (size_t i = 0; i < a_elements && remaining >= sizeof(float); i++) {
                consumeBytes(ptr, remaining, a_data[i]);
                // Normalize to avoid numerical issues
                a_data[i] = std::fmod(a_data[i], 10.0f);
            }
            A = torch::from_blob(a_data.data(), a_shape, torch::kFloat).to(dtype).clone();
        } else {
            // Complex types
            std::vector<std::complex<float>> a_data(a_elements);
            for (size_t i = 0; i < a_elements && remaining >= 2*sizeof(float); i++) {
                float real = 0, imag = 0;
                consumeBytes(ptr, remaining, real);
                consumeBytes(ptr, remaining, imag);
                a_data[i] = std::complex<float>(std::fmod(real, 10.0f), std::fmod(imag, 10.0f));
            }
            A = torch::from_blob(a_data.data(), a_shape, torch::kComplexFloat).to(dtype).clone();
        }
        
        // Create B tensor
        torch::Tensor B;
        if (dtype == torch::kFloat || dtype == torch::kDouble) {
            std::vector<float> b_data(b_elements);
            for (size_t i = 0; i < b_elements && remaining >= sizeof(float); i++) {
                consumeBytes(ptr, remaining, b_data[i]);
                b_data[i] = std::fmod(b_data[i], 10.0f);
            }
            B = torch::from_blob(b_data.data(), b_shape, torch::kFloat).to(dtype).clone();
        } else {
            std::vector<std::complex<float>> b_data(b_elements);
            for (size_t i = 0; i < b_elements && remaining >= 2*sizeof(float); i++) {
                float real = 0, imag = 0;
                consumeBytes(ptr, remaining, real);
                consumeBytes(ptr, remaining, imag);
                b_data[i] = std::complex<float>(std::fmod(real, 10.0f), std::fmod(imag, 10.0f));
            }
            B = torch::from_blob(b_data.data(), b_shape, torch::kComplexFloat).to(dtype).clone();
        }
        
        // Prepare optional dims parameter
        c10::optional<at::IntArrayRef> dims_opt = c10::nullopt;
        std::vector<int64_t> dims_vec;
        if ((use_dims % 3) == 1 && a_shape.size() > b_ndim) {
            // Generate some dims to permute
            uint8_t num_dims = 0;
            if (consumeBytes(ptr, remaining, num_dims)) {
                num_dims = (num_dims % std::min(3, (int)a_shape.size())) + 1;
                for (int i = 0; i < num_dims && remaining > 0; i++) {
                    uint8_t dim_idx = 0;
                    if (consumeBytes(ptr, remaining, dim_idx)) {
                        dims_vec.push_back(dim_idx % a_shape.size());
                    }
                }
                if (!dims_vec.empty()) {
                    dims_opt = dims_vec;
                }
            }
        }
        
        // Prepare optional out tensor
        torch::Tensor out;
        bool use_out_tensor = (use_out % 4) == 1;
        if (use_out_tensor) {
            // Expected output shape is A.shape[B.ndim:]
            std::vector<int64_t> out_shape(a_shape.begin() + b_ndim, a_shape.end());
            out = torch::empty(out_shape, torch::TensorOptions().dtype(dtype));
        }
        
        // Call tensorsolve
        torch::Tensor result;
        if (use_out_tensor) {
            result = torch::linalg::tensorsolve_out(out, A, B, dims_opt);
        } else {
            result = torch::linalg::tensorsolve(A, B, dims_opt);
        }
        
        // Perform some operations to exercise the result
        if (result.numel() > 0) {
            auto sum = result.sum();
            auto mean = result.mean();
            if (result.dim() > 0) {
                auto reshaped = result.reshape({-1});
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch errors are expected for invalid inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}