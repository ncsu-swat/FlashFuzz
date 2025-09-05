#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper function to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) {
        return false;
    }
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 16) {
        return 0;  // Need minimum bytes for basic parameters
    }

    try {
        size_t offset = 0;
        
        // Consume parameters for tensor creation
        uint8_t rank;
        if (!consumeBytes(data, size, offset, rank)) return 0;
        rank = (rank % 4) + 1;  // Limit rank to 1-4 dimensions
        
        // Build shape vector
        std::vector<int64_t> shape;
        for (uint8_t i = 0; i < rank; ++i) {
            uint8_t dim_size;
            if (!consumeBytes(data, size, offset, dim_size)) {
                // Use default if we run out of data
                shape.push_back(2);
            } else {
                // Allow 0-sized dimensions for edge cases, cap at 32 for memory
                shape.push_back(dim_size % 33);
            }
        }
        
        // Consume groups parameter
        int32_t groups = 1;
        if (!consumeBytes(data, size, offset, groups)) {
            groups = 2;  // Default value
        }
        // Ensure groups is positive and reasonable
        groups = std::abs(groups) % 100;
        if (groups == 0) groups = 1;
        
        // Consume dtype selector
        uint8_t dtype_selector;
        if (!consumeBytes(data, size, offset, dtype_selector)) {
            dtype_selector = 0;
        }
        
        // Select dtype based on fuzzer input
        torch::ScalarType dtype;
        switch (dtype_selector % 6) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            case 3: dtype = torch::kBFloat16; break;
            case 4: dtype = torch::kInt32; break;
            case 5: dtype = torch::kInt64; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Consume device selector
        uint8_t device_selector;
        if (!consumeBytes(data, size, offset, device_selector)) {
            device_selector = 0;
        }
        
        // Create device (CPU or CUDA if available)
        torch::Device device(torch::kCPU);
        #ifdef USE_CUDA
        if (device_selector % 2 == 1 && torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
        }
        #endif
        
        // Consume requires_grad flag
        uint8_t requires_grad;
        if (!consumeBytes(data, size, offset, requires_grad)) {
            requires_grad = 0;
        }
        
        // Create tensor options
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .requires_grad((requires_grad % 2) == 1 && 
                          (dtype == torch::kFloat32 || dtype == torch::kFloat64 || 
                           dtype == torch::kFloat16 || dtype == torch::kBFloat16));
        
        // Calculate total elements
        int64_t total_elements = 1;
        for (auto dim : shape) {
            if (dim > 0 && total_elements > INT64_MAX / dim) {
                // Prevent overflow
                return 0;
            }
            total_elements *= dim;
        }
        
        // Limit total elements to prevent OOM
        if (total_elements > 100000) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input;
        
        // Decide initialization method based on fuzzer input
        uint8_t init_method;
        if (!consumeBytes(data, size, offset, init_method)) {
            init_method = 0;
        }
        
        switch (init_method % 5) {
            case 0:
                input = torch::randn(shape, options);
                break;
            case 1:
                input = torch::ones(shape, options);
                break;
            case 2:
                input = torch::zeros(shape, options);
                break;
            case 3:
                input = torch::arange(total_elements, options).reshape(shape);
                break;
            case 4:
                // Create with specific values from fuzzer data
                if (total_elements > 0 && offset < size) {
                    std::vector<float> values;
                    for (int64_t i = 0; i < total_elements; ++i) {
                        if (offset < size) {
                            values.push_back(static_cast<float>(data[offset++]) / 255.0f);
                        } else {
                            values.push_back(0.0f);
                        }
                    }
                    input = torch::from_blob(values.data(), shape, torch::kFloat32).to(options);
                } else {
                    input = torch::randn(shape, options);
                }
                break;
            default:
                input = torch::randn(shape, options);
        }
        
        // Test with contiguous and non-contiguous tensors
        uint8_t make_non_contiguous;
        if (consumeBytes(data, size, offset, make_non_contiguous) && 
            (make_non_contiguous % 3 == 0) && shape.size() >= 2) {
            // Make tensor non-contiguous by transposing
            input = input.transpose(0, shape.size() - 1);
        }
        
        // Call native_channel_shuffle
        torch::Tensor output = torch::native_channel_shuffle(input, groups);
        
        // Perform some operations on output to ensure it's valid
        if (output.numel() > 0) {
            auto sum = output.sum();
            auto mean = output.mean();
            
            // Test backward pass if applicable
            if (output.requires_grad()) {
                try {
                    sum.backward();
                } catch (...) {
                    // Ignore backward errors
                }
            }
        }
        
        // Test with different memory formats if applicable
        if (shape.size() == 4 && shape[0] > 0 && shape[1] > 0 && 
            shape[2] > 0 && shape[3] > 0) {
            try {
                auto channels_last_input = input.contiguous(torch::MemoryFormat::ChannelsLast);
                auto cl_output = torch::native_channel_shuffle(channels_last_input, groups);
            } catch (...) {
                // Ignore memory format errors
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Unknown exception
        return -1;
    }
    
    return 0;
}