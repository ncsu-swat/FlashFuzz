#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>

// Helper to consume bytes from fuzzer input
template<typename T>
T consume(const uint8_t*& data, size_t& size) {
    if (size < sizeof(T)) {
        return T{};
    }
    T value;
    std::memcpy(&value, data, sizeof(T));
    data += sizeof(T);
    size -= sizeof(T);
    return value;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 16) {
        return 0;  // Need minimum bytes for configuration
    }

    try {
        // Consume configuration bytes
        uint8_t dtype_selector = consume<uint8_t>(data, size) % 4;
        uint8_t batch_dims = consume<uint8_t>(data, size) % 3;  // 0, 1, or 2 batch dimensions
        uint8_t m = 1 + (consume<uint8_t>(data, size) % 10);  // m in [1, 10]
        uint8_t n = 1 + (consume<uint8_t>(data, size) % m);   // n in [1, m] to ensure m >= n
        uint8_t k = consume<uint8_t>(data, size) % (n + 1);   // k in [0, n] to ensure k <= n
        
        if (k == 0) k = 1;  // Ensure k is at least 1 for meaningful test
        
        // Select dtype
        torch::ScalarType dtype;
        bool is_complex = false;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat; break;
            case 1: dtype = torch::kDouble; break;
            case 2: dtype = torch::kComplexFloat; is_complex = true; break;
            case 3: dtype = torch::kComplexDouble; is_complex = true; break;
            default: dtype = torch::kFloat; break;
        }

        // Build shape for A tensor
        std::vector<int64_t> a_shape;
        std::vector<int64_t> tau_shape;
        
        // Add batch dimensions
        for (int i = 0; i < batch_dims; i++) {
            int64_t batch_size = 1 + (consume<uint8_t>(data, size) % 4);
            a_shape.push_back(batch_size);
            tau_shape.push_back(batch_size);
        }
        
        // Add matrix dimensions
        a_shape.push_back(m);
        a_shape.push_back(n);
        tau_shape.push_back(k);

        // Calculate number of elements needed
        int64_t a_numel = 1;
        for (auto dim : a_shape) a_numel *= dim;
        int64_t tau_numel = 1;
        for (auto dim : tau_shape) tau_numel *= dim;

        // Create tensors with fuzzer data
        torch::Tensor A, tau;
        
        if (is_complex) {
            // For complex types, we need twice the data (real and imaginary parts)
            std::vector<float> a_data_real, a_data_imag;
            std::vector<float> tau_data_real, tau_data_imag;
            
            for (int64_t i = 0; i < a_numel; i++) {
                float val = static_cast<float>(consume<int8_t>(data, size)) / 32.0f;
                a_data_real.push_back(val);
                val = static_cast<float>(consume<int8_t>(data, size)) / 32.0f;
                a_data_imag.push_back(val);
            }
            
            for (int64_t i = 0; i < tau_numel; i++) {
                float val = static_cast<float>(consume<int8_t>(data, size)) / 32.0f;
                tau_data_real.push_back(val);
                val = static_cast<float>(consume<int8_t>(data, size)) / 32.0f;
                tau_data_imag.push_back(val);
            }
            
            auto A_real = torch::from_blob(a_data_real.data(), a_shape, torch::kFloat).clone();
            auto A_imag = torch::from_blob(a_data_imag.data(), a_shape, torch::kFloat).clone();
            A = torch::complex(A_real, A_imag);
            
            auto tau_real = torch::from_blob(tau_data_real.data(), tau_shape, torch::kFloat).clone();
            auto tau_imag = torch::from_blob(tau_data_imag.data(), tau_shape, torch::kFloat).clone();
            tau = torch::complex(tau_real, tau_imag);
            
            if (dtype == torch::kComplexDouble) {
                A = A.to(torch::kComplexDouble);
                tau = tau.to(torch::kComplexDouble);
            }
        } else {
            // For real types
            std::vector<float> a_data, tau_data;
            
            for (int64_t i = 0; i < a_numel; i++) {
                float val = static_cast<float>(consume<int8_t>(data, size)) / 32.0f;
                a_data.push_back(val);
            }
            
            for (int64_t i = 0; i < tau_numel; i++) {
                float val = static_cast<float>(consume<int8_t>(data, size)) / 32.0f;
                tau_data.push_back(val);
            }
            
            A = torch::from_blob(a_data.data(), a_shape, torch::kFloat).clone().to(dtype);
            tau = torch::from_blob(tau_data.data(), tau_shape, torch::kFloat).clone().to(dtype);
        }

        // Test with and without out parameter
        bool use_out = consume<uint8_t>(data, size) % 2;
        
        if (use_out) {
            // Create output tensor with same shape and dtype as A
            torch::Tensor out = torch::empty_like(A);
            torch::linalg::householder_product_out(out, A, tau);
            
            // Verify output shape
            if (out.sizes() != A.sizes()) {
                std::cerr << "Output shape mismatch" << std::endl;
            }
        } else {
            // Call without out parameter
            torch::Tensor result = torch::linalg::householder_product(A, tau);
            
            // Verify output shape
            if (result.sizes() != A.sizes()) {
                std::cerr << "Result shape mismatch" << std::endl;
            }
            
            // Test edge cases
            if (size > 0) {
                uint8_t edge_case = consume<uint8_t>(data, size) % 4;
                switch (edge_case) {
                    case 0:
                        // Test with zero-sized dimension in batch
                        if (batch_dims > 0) {
                            auto zero_shape = a_shape;
                            zero_shape[0] = 0;
                            auto zero_A = torch::empty(zero_shape, A.options());
                            auto zero_tau_shape = tau_shape;
                            zero_tau_shape[0] = 0;
                            auto zero_tau = torch::empty(zero_tau_shape, tau.options());
                            try {
                                torch::linalg::householder_product(zero_A, zero_tau);
                            } catch (...) {
                                // Expected for some edge cases
                            }
                        }
                        break;
                    case 1:
                        // Test with contiguous vs non-contiguous
                        if (A.numel() > 1) {
                            auto A_t = A.transpose(-2, -1);
                            auto tau_c = tau.contiguous();
                            try {
                                torch::linalg::householder_product(A_t, tau_c);
                            } catch (...) {
                                // Expected when dimensions don't match requirements
                            }
                        }
                        break;
                    case 2:
                        // Test with different memory layouts
                        if (A.dim() >= 2) {
                            auto A_permuted = A.permute({-1, -2});
                            try {
                                torch::linalg::householder_product(A_permuted, tau);
                            } catch (...) {
                                // Expected when m < n after permutation
                            }
                        }
                        break;
                    case 3:
                        // Test with requires_grad
                        if (dtype == torch::kFloat || dtype == torch::kDouble) {
                            A.requires_grad_(true);
                            tau.requires_grad_(true);
                            auto result_grad = torch::linalg::householder_product(A, tau);
                            if (result_grad.requires_grad()) {
                                // Compute gradient
                                auto grad_out = torch::ones_like(result_grad);
                                result_grad.backward(grad_out);
                            }
                        }
                        break;
                }
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch errors are expected for invalid inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        return 0;
    }
    
    return 0;
}