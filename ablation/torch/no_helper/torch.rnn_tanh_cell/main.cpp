#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t*& data, size_t& size, T& out) {
    if (size < sizeof(T)) return false;
    std::memcpy(&out, data, sizeof(T));
    data += sizeof(T);
    size -= sizeof(T);
    return true;
}

// Helper to create tensor from fuzzer input
torch::Tensor createTensorFromBytes(const uint8_t*& data, size_t& size) {
    // Consume parameters for tensor creation
    uint8_t rank;
    if (!consumeBytes(data, size, rank)) {
        return torch::randn({1});
    }
    rank = (rank % 4) + 1; // Limit rank to 1-4 for practical fuzzing
    
    std::vector<int64_t> shape;
    for (int i = 0; i < rank; i++) {
        uint8_t dim;
        if (!consumeBytes(data, size, dim)) {
            shape.push_back(1);
        } else {
            // Allow 0-dim tensors but cap at reasonable size
            shape.push_back(dim % 32);
        }
    }
    
    // Consume dtype choice
    uint8_t dtype_choice;
    if (!consumeBytes(data, size, dtype_choice)) {
        dtype_choice = 0;
    }
    
    torch::ScalarType dtype;
    switch (dtype_choice % 4) {
        case 0: dtype = torch::kFloat32; break;
        case 1: dtype = torch::kFloat64; break;
        case 2: dtype = torch::kFloat16; break;
        case 3: dtype = torch::kBFloat16; break;
        default: dtype = torch::kFloat32;
    }
    
    // Consume requires_grad flag
    uint8_t requires_grad;
    if (!consumeBytes(data, size, requires_grad)) {
        requires_grad = 0;
    }
    
    // Create tensor with random data or from fuzzer bytes
    torch::Tensor tensor;
    try {
        tensor = torch::randn(shape, torch::dtype(dtype));
        
        // Optionally fill with fuzzer data
        if (size > 0 && (size % 2) == 0) {
            auto numel = tensor.numel();
            if (numel > 0 && size >= numel * sizeof(float)) {
                std::vector<float> values(numel);
                for (int64_t i = 0; i < numel && size >= sizeof(float); i++) {
                    consumeBytes(data, size, values[i]);
                }
                tensor = torch::from_blob(values.data(), shape, torch::kFloat32).to(dtype);
            }
        }
        
        if (requires_grad % 2 == 1 && dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            tensor.requires_grad_(true);
        }
    } catch (...) {
        tensor = torch::randn({1}, torch::dtype(dtype));
    }
    
    return tensor;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 4) {
            // Need minimum bytes to create tensors
            return 0;
        }
        
        const uint8_t* ptr = data;
        size_t remaining = size;
        
        // Create input tensor
        torch::Tensor input = createTensorFromBytes(ptr, remaining);
        
        // Create hidden state tensor
        torch::Tensor hx = createTensorFromBytes(ptr, remaining);
        
        // Create weight_ih tensor
        torch::Tensor w_ih = createTensorFromBytes(ptr, remaining);
        
        // Create weight_hh tensor
        torch::Tensor w_hh = createTensorFromBytes(ptr, remaining);
        
        // Create bias_ih tensor (optional)
        torch::Tensor b_ih;
        uint8_t use_bias_ih;
        if (consumeBytes(ptr, remaining, use_bias_ih) && (use_bias_ih % 2 == 0)) {
            b_ih = createTensorFromBytes(ptr, remaining);
        }
        
        // Create bias_hh tensor (optional)
        torch::Tensor b_hh;
        uint8_t use_bias_hh;
        if (consumeBytes(ptr, remaining, use_bias_hh) && (use_bias_hh % 2 == 0)) {
            b_hh = createTensorFromBytes(ptr, remaining);
        }
        
        // Try different tensor configurations
        try {
            // Call rnn_tanh_cell with various configurations
            torch::Tensor result;
            
            if (b_ih.defined() && b_hh.defined()) {
                result = torch::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
            } else if (b_ih.defined()) {
                result = torch::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, torch::Tensor());
            } else if (b_hh.defined()) {
                result = torch::rnn_tanh_cell(input, hx, w_ih, w_hh, torch::Tensor(), b_hh);
            } else {
                result = torch::rnn_tanh_cell(input, hx, w_ih, w_hh, torch::Tensor(), torch::Tensor());
            }
            
            // Exercise the result tensor
            if (result.defined()) {
                auto sum = result.sum();
                if (result.requires_grad()) {
                    try {
                        sum.backward();
                    } catch (...) {
                        // Ignore gradient computation errors
                    }
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
        } catch (const std::runtime_error& e) {
            // Runtime errors are expected for shape mismatches etc
        }
        
        // Try with different memory layouts
        if (remaining > 0) {
            try {
                auto input_t = input.t();
                auto hx_t = hx.t();
                auto result = torch::rnn_tanh_cell(input_t, hx_t, w_ih, w_hh, b_ih, b_hh);
            } catch (...) {
                // Ignore errors from transposed tensors
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        return -1;
    }
    
    return 0;
}