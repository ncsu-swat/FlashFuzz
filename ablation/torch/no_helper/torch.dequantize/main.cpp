#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

// Create quantized tensor from fuzzer input
torch::Tensor createQuantizedTensor(const uint8_t* data, size_t size, size_t& offset) {
    // Consume parameters for tensor creation
    uint8_t ndims = 0;
    if (!consumeBytes(data, size, offset, ndims)) {
        return torch::quantize_per_tensor(torch::randn({1}), 1.0, 0, torch::kQInt8);
    }
    ndims = (ndims % 5) + 1; // Limit dimensions to 1-5
    
    std::vector<int64_t> shape;
    for (uint8_t i = 0; i < ndims; ++i) {
        uint8_t dim_size = 0;
        if (!consumeBytes(data, size, offset, dim_size)) {
            dim_size = 1;
        }
        shape.push_back(dim_size % 10); // Keep dimensions small (0-9)
    }
    
    // Get quantization parameters
    float scale = 1.0f;
    int32_t zero_point = 0;
    uint8_t dtype_choice = 0;
    
    consumeBytes(data, size, offset, scale);
    consumeBytes(data, size, offset, zero_point);
    consumeBytes(data, size, offset, dtype_choice);
    
    // Sanitize scale to avoid inf/nan
    if (!std::isfinite(scale) || scale == 0.0f) {
        scale = 1.0f;
    }
    scale = std::abs(scale);
    if (scale > 1e6f) scale = 1e6f;
    if (scale < 1e-6f) scale = 1e-6f;
    
    // Choose quantized dtype
    torch::ScalarType qtype;
    switch (dtype_choice % 4) {
        case 0: qtype = torch::kQInt8; zero_point = std::max(-128, std::min(127, zero_point)); break;
        case 1: qtype = torch::kQUInt8; zero_point = std::max(0, std::min(255, zero_point)); break;
        case 2: qtype = torch::kQInt32; break;
        default: qtype = torch::kQInt8; zero_point = std::max(-128, std::min(127, zero_point)); break;
    }
    
    // Create base tensor with fuzzer data
    int64_t numel = 1;
    for (auto d : shape) numel *= d;
    
    torch::Tensor base_tensor;
    if (numel == 0) {
        base_tensor = torch::zeros(shape);
    } else {
        std::vector<float> values;
        for (int64_t i = 0; i < numel; ++i) {
            float val = 0.0f;
            if (consumeBytes(data, size, offset, val)) {
                // Sanitize value
                if (!std::isfinite(val)) val = 0.0f;
                val = std::max(-1000.0f, std::min(1000.0f, val));
            }
            values.push_back(val);
        }
        base_tensor = torch::from_blob(values.data(), shape, torch::kFloat32).clone();
    }
    
    // Quantize the tensor
    try {
        if (qtype == torch::kQInt32) {
            // QInt32 requires per-channel quantization
            int axis = shape.size() > 0 ? 0 : -1;
            int64_t channel_count = shape.size() > 0 ? shape[0] : 1;
            if (channel_count == 0) channel_count = 1;
            
            std::vector<double> scales(channel_count, static_cast<double>(scale));
            std::vector<int64_t> zero_points(channel_count, 0); // QInt32 zero points must be 0
            
            return torch::quantize_per_channel(base_tensor, 
                torch::from_blob(scales.data(), {channel_count}, torch::kFloat64).to(torch::kFloat32),
                torch::from_blob(zero_points.data(), {channel_count}, torch::kInt64).to(torch::kInt32),
                axis, qtype);
        } else {
            return torch::quantize_per_tensor(base_tensor, scale, zero_point, qtype);
        }
    } catch (...) {
        // Fallback to simple quantization
        return torch::quantize_per_tensor(torch::randn({2, 2}), 1.0, 0, torch::kQInt8);
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 1) return 0;
        
        size_t offset = 0;
        uint8_t mode = 0;
        consumeBytes(data, size, offset, mode);
        
        if (mode % 2 == 0) {
            // Test single tensor dequantization
            torch::Tensor quantized = createQuantizedTensor(data, size, offset);
            
            // Test dequantize operation
            torch::Tensor dequantized = torch::dequantize(quantized);
            
            // Perform some operations to ensure tensor is valid
            if (dequantized.numel() > 0) {
                auto sum = dequantized.sum();
                auto mean = dequantized.mean();
                
                // Test edge cases
                if (dequantized.dim() > 0) {
                    auto reshaped = dequantized.reshape({-1});
                    auto max_val = dequantized.max();
                    auto min_val = dequantized.min();
                }
            }
            
            // Test with different tensor views
            if (quantized.numel() > 1 && quantized.dim() > 0) {
                auto sliced = quantized.narrow(0, 0, std::min(int64_t(1), quantized.size(0)));
                auto dequantized_slice = torch::dequantize(sliced);
            }
            
        } else {
            // Test list of tensors dequantization
            uint8_t num_tensors = 0;
            consumeBytes(data, size, offset, num_tensors);
            num_tensors = (num_tensors % 5) + 1; // 1-5 tensors
            
            std::vector<torch::Tensor> quantized_list;
            for (uint8_t i = 0; i < num_tensors; ++i) {
                quantized_list.push_back(createQuantizedTensor(data, size, offset));
            }
            
            // Dequantize list
            auto dequantized_list = torch::dequantize(quantized_list);
            
            // Verify results
            for (size_t i = 0; i < dequantized_list.size(); ++i) {
                if (dequantized_list[i].numel() > 0) {
                    auto sum = dequantized_list[i].sum();
                    
                    // Test concatenation of dequantized tensors
                    if (i > 0 && dequantized_list[i].sizes() == dequantized_list[0].sizes()) {
                        try {
                            auto stacked = torch::stack({dequantized_list[0], dequantized_list[i]});
                        } catch (...) {
                            // Ignore stacking errors
                        }
                    }
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
        return 0;
    }
    
    return 0;
}