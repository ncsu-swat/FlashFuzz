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
    rank = (rank % 5) + 1; // Limit rank to 1-5 dimensions
    
    std::vector<int64_t> shape;
    for (int i = 0; i < rank; i++) {
        uint8_t dim;
        if (!consumeBytes(data, size, dim)) {
            shape.push_back(1);
        } else {
            shape.push_back((dim % 10) + 1); // Dimensions 1-10
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
    
    // Create tensor with random data
    auto options = torch::TensorOptions().dtype(dtype);
    torch::Tensor tensor = torch::randn(shape, options);
    
    // Optionally make it contiguous or not
    uint8_t contiguous;
    if (consumeBytes(data, size, contiguous) && (contiguous % 2 == 1)) {
        tensor = tensor.transpose(0, -1);
    }
    
    return tensor;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 10) return 0; // Need minimum bytes
        
        const uint8_t* ptr = data;
        size_t remaining = size;
        
        // Create input tensor
        torch::Tensor input = createTensorFromBytes(ptr, remaining);
        
        // Ensure input has at least 2 dimensions for batch norm
        if (input.dim() < 2) {
            input = input.unsqueeze(0);
        }
        
        // Get number of features (second dimension)
        int64_t num_features = input.size(1);
        
        // Create or generate running_mean and running_var
        torch::Tensor running_mean, running_var;
        uint8_t use_running_stats;
        if (consumeBytes(ptr, remaining, use_running_stats) && (use_running_stats % 2 == 0)) {
            running_mean = torch::zeros({num_features}, input.options());
            running_var = torch::ones({num_features}, input.options());
        }
        
        // Create weight and bias parameters
        torch::Tensor weight, bias;
        uint8_t use_weight_bias;
        if (consumeBytes(ptr, remaining, use_weight_bias)) {
            if (use_weight_bias % 2 == 0) {
                weight = torch::randn({num_features}, input.options());
            }
            if ((use_weight_bias >> 1) % 2 == 0) {
                bias = torch::randn({num_features}, input.options());
            }
        }
        
        // Consume training flag
        bool training = true;
        uint8_t training_byte;
        if (consumeBytes(ptr, remaining, training_byte)) {
            training = (training_byte % 2 == 0);
        }
        
        // Consume momentum
        double momentum = 0.1;
        uint8_t momentum_byte;
        if (consumeBytes(ptr, remaining, momentum_byte)) {
            momentum = static_cast<double>(momentum_byte) / 255.0;
        }
        
        // Consume epsilon
        double eps = 1e-5;
        uint8_t eps_byte;
        if (consumeBytes(ptr, remaining, eps_byte)) {
            eps = 1e-8 + (static_cast<double>(eps_byte) / 255.0) * 1e-3;
        }
        
        // Call batch_norm with various parameter combinations
        torch::Tensor output;
        
        // Try different invocation patterns
        uint8_t pattern;
        if (consumeBytes(ptr, remaining, pattern)) {
            switch (pattern % 4) {
                case 0:
                    // Full parameters
                    output = torch::batch_norm(
                        input,
                        weight.defined() ? weight : torch::Tensor(),
                        bias.defined() ? bias : torch::Tensor(),
                        running_mean.defined() ? running_mean : torch::Tensor(),
                        running_var.defined() ? running_var : torch::Tensor(),
                        training,
                        momentum,
                        eps,
                        false  // cudnn_enabled
                    );
                    break;
                case 1:
                    // Minimal parameters
                    output = torch::batch_norm(
                        input,
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        training,
                        momentum,
                        eps,
                        false
                    );
                    break;
                case 2:
                    // With running stats but no affine
                    output = torch::batch_norm(
                        input,
                        torch::Tensor(),
                        torch::Tensor(),
                        running_mean.defined() ? running_mean : torch::zeros({num_features}),
                        running_var.defined() ? running_var : torch::ones({num_features}),
                        training,
                        momentum,
                        eps,
                        false
                    );
                    break;
                case 3:
                    // Edge case: very small epsilon
                    output = torch::batch_norm(
                        input,
                        weight.defined() ? weight : torch::Tensor(),
                        bias.defined() ? bias : torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        training,
                        0.999,  // High momentum
                        1e-10,  // Very small epsilon
                        false
                    );
                    break;
            }
        } else {
            // Default call
            output = torch::batch_norm(
                input,
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                training,
                momentum,
                eps,
                false
            );
        }
        
        // Perform some operations on output to trigger potential issues
        if (output.defined()) {
            auto sum = output.sum();
            auto mean = output.mean();
            
            // Try backward pass if in training mode
            if (training && output.requires_grad()) {
                try {
                    sum.backward();
                } catch (...) {
                    // Ignore backward errors
                }
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        return -1;
    }
    
    return 0;
}