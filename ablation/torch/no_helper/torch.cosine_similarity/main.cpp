#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t*& data, size_t& size, T& value) {
    if (size < sizeof(T)) return false;
    std::memcpy(&value, data, sizeof(T));
    data += sizeof(T);
    size -= sizeof(T);
    return true;
}

// Helper to create tensor from fuzzer input
torch::Tensor createTensorFromBytes(const uint8_t*& data, size_t& size) {
    // Consume parameters for tensor creation
    uint8_t rank;
    if (!consumeBytes(data, size, rank)) {
        return torch::empty({0});
    }
    rank = (rank % 5) + 1; // Limit rank to 1-5 dimensions
    
    std::vector<int64_t> shape;
    for (int i = 0; i < rank; ++i) {
        uint8_t dim_size;
        if (!consumeBytes(data, size, dim_size)) {
            shape.push_back(1);
        } else {
            // Allow 0-sized dimensions and various sizes
            shape.push_back(dim_size % 10); // Limit individual dimension size
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
    
    // Create tensor with random values or specific patterns
    uint8_t pattern;
    if (!consumeBytes(data, size, pattern)) {
        pattern = 0;
    }
    
    torch::Tensor tensor;
    try {
        switch (pattern % 6) {
            case 0:
                tensor = torch::randn(shape, torch::dtype(dtype));
                break;
            case 1:
                tensor = torch::zeros(shape, torch::dtype(dtype));
                break;
            case 2:
                tensor = torch::ones(shape, torch::dtype(dtype));
                break;
            case 3:
                tensor = torch::full(shape, std::numeric_limits<float>::quiet_NaN(), torch::dtype(dtype));
                break;
            case 4:
                tensor = torch::full(shape, std::numeric_limits<float>::infinity(), torch::dtype(dtype));
                break;
            case 5:
                tensor = torch::full(shape, -std::numeric_limits<float>::infinity(), torch::dtype(dtype));
                break;
        }
    } catch (...) {
        tensor = torch::empty({0});
    }
    
    return tensor;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0; // Need minimum bytes for basic parameters
    
    try {
        // Create two input tensors
        torch::Tensor x1 = createTensorFromBytes(data, size);
        torch::Tensor x2 = createTensorFromBytes(data, size);
        
        // Consume dimension parameter
        int64_t dim = 1;
        uint8_t dim_byte;
        if (consumeBytes(data, size, dim_byte)) {
            // Allow negative dimensions too
            dim = static_cast<int64_t>(dim_byte % 10) - 5;
        }
        
        // Consume epsilon parameter
        double eps = 1e-8;
        uint8_t eps_choice;
        if (consumeBytes(data, size, eps_choice)) {
            switch (eps_choice % 6) {
                case 0: eps = 1e-8; break;
                case 1: eps = 0.0; break;
                case 2: eps = 1e-16; break;
                case 3: eps = 1.0; break;
                case 4: eps = -1e-8; break; // Negative epsilon
                case 5: eps = std::numeric_limits<double>::quiet_NaN(); break;
            }
        }
        
        // Try to make tensors broadcastable by adjusting shapes
        uint8_t broadcast_strategy;
        if (consumeBytes(data, size, broadcast_strategy)) {
            switch (broadcast_strategy % 4) {
                case 0:
                    // Keep as is
                    break;
                case 1:
                    // Make x2 same shape as x1
                    if (x1.defined() && x1.numel() > 0) {
                        try {
                            x2 = x2.reshape_as(x1);
                        } catch (...) {
                            // Ignore reshape failures
                        }
                    }
                    break;
                case 2:
                    // Add singleton dimensions
                    if (x1.dim() > 0 && x2.dim() > 0) {
                        x2 = x2.unsqueeze(0);
                    }
                    break;
                case 3:
                    // Expand dimensions
                    if (x1.dim() > 0 && x2.dim() > 0) {
                        try {
                            auto max_dim = std::max(x1.dim(), x2.dim());
                            if (x1.dim() < max_dim) {
                                for (int i = x1.dim(); i < max_dim; ++i) {
                                    x1 = x1.unsqueeze(0);
                                }
                            }
                            if (x2.dim() < max_dim) {
                                for (int i = x2.dim(); i < max_dim; ++i) {
                                    x2 = x2.unsqueeze(0);
                                }
                            }
                        } catch (...) {
                            // Ignore expansion failures
                        }
                    }
                    break;
            }
        }
        
        // Call cosine_similarity
        try {
            torch::Tensor result = torch::cosine_similarity(x1, x2, dim, eps);
            
            // Perform some basic operations on result to increase coverage
            if (result.defined() && result.numel() > 0) {
                auto sum = result.sum();
                auto mean = result.mean();
                auto max_val = result.max();
                auto min_val = result.min();
                
                // Check for special values
                bool has_nan = result.isnan().any().item<bool>();
                bool has_inf = result.isinf().any().item<bool>();
            }
        } catch (const c10::Error& e) {
            // Expected errors from invalid inputs - continue fuzzing
        } catch (const std::runtime_error& e) {
            // Expected errors from invalid inputs - continue fuzzing
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}