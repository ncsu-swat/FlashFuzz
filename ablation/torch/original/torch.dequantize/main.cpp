#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to create quantized tensor from fuzzer input
torch::Tensor createQuantizedTensor(const uint8_t* data, size_t& offset, size_t size) {
    // Need at least 4 bytes: dtype(1), rank(1), qscheme(1), reserved(1)
    if (offset + 4 > size) {
        throw std::runtime_error("Not enough data for quantized tensor metadata");
    }
    
    // Parse quantization scheme
    uint8_t qscheme_byte = data[offset++];
    torch::QScheme qscheme;
    switch (qscheme_byte % 4) {
        case 0: qscheme = torch::kPerTensorAffine; break;
        case 1: qscheme = torch::kPerChannelAffine; break;
        case 2: qscheme = torch::kPerTensorSymmetric; break;
        case 3: qscheme = torch::kPerChannelSymmetric; break;
        default: qscheme = torch::kPerTensorAffine;
    }
    
    // Parse quantized dtype
    uint8_t qdtype_byte = data[offset++];
    torch::ScalarType qdtype;
    switch (qdtype_byte % 4) {
        case 0: qdtype = torch::kQInt8; break;
        case 1: qdtype = torch::kQUInt8; break;
        case 2: qdtype = torch::kQInt32; break;
        case 3: qdtype = torch::kQUInt4x2; break;
        default: qdtype = torch::kQInt8;
    }
    
    // Parse rank
    uint8_t rank = data[offset++] % 5; // 0-4 dimensions
    offset++; // Skip reserved byte
    
    // Parse shape
    std::vector<int64_t> shape;
    for (uint8_t i = 0; i < rank; ++i) {
        if (offset >= size) {
            shape.push_back(1);
        } else {
            shape.push_back((data[offset++] % 10) + 1); // 1-10 per dimension
        }
    }
    
    // Calculate number of elements
    int64_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }
    
    // Parse scale and zero_point
    double scale = 1.0;
    int64_t zero_point = 0;
    
    if (offset + sizeof(float) <= size) {
        float scale_raw;
        std::memcpy(&scale_raw, data + offset, sizeof(float));
        offset += sizeof(float);
        scale = std::abs(scale_raw) * 0.001 + 0.001; // Keep scale positive and reasonable
    }
    
    if (offset + sizeof(int32_t) <= size) {
        int32_t zp_raw;
        std::memcpy(&zp_raw, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        zero_point = zp_raw % 256; // Keep in reasonable range
    }
    
    // Create base tensor with float data
    torch::Tensor base_tensor;
    if (num_elements == 0) {
        base_tensor = torch::empty(shape, torch::kFloat32);
    } else {
        // Fill with some data from fuzzer input
        std::vector<float> float_data(num_elements);
        for (int64_t i = 0; i < num_elements; ++i) {
            if (offset < size) {
                float_data[i] = static_cast<float>(data[offset++]) / 255.0f * 10.0f - 5.0f;
            } else {
                float_data[i] = 0.0f;
            }
        }
        base_tensor = torch::from_blob(float_data.data(), shape, torch::kFloat32).clone();
    }
    
    // Quantize the tensor
    torch::Tensor quantized;
    try {
        if (qscheme == torch::kPerChannelAffine || qscheme == torch::kPerChannelSymmetric) {
            // For per-channel, we need scales and zero_points per channel
            int64_t axis = 0;
            if (rank > 0 && offset < size) {
                axis = data[offset++] % rank;
            }
            
            int64_t num_channels = (rank > 0 && axis < rank) ? shape[axis] : 1;
            
            torch::Tensor scales = torch::ones({num_channels}, torch::kFloat64) * scale;
            torch::Tensor zero_points = torch::zeros({num_channels}, torch::kInt64);
            
            // Fill channel-wise scales and zero points
            for (int64_t i = 0; i < num_channels && offset < size; ++i) {
                scales[i] = (data[offset++] / 255.0) * 0.1 + 0.001;
                if (offset < size) {
                    zero_points[i] = static_cast<int64_t>(data[offset++]) - 128;
                }
            }
            
            quantized = torch::quantize_per_channel(base_tensor, scales, zero_points, axis, qdtype);
        } else {
            quantized = torch::quantize_per_tensor(base_tensor, scale, zero_point, qdtype);
        }
    } catch (...) {
        // Fallback to simple per-tensor quantization
        quantized = torch::quantize_per_tensor(base_tensor, 0.1, 0, torch::kQInt8);
    }
    
    return quantized;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 1) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Decide whether to test single tensor or list of tensors
        bool test_list = (size > 0) && (data[offset++] % 2 == 0);
        
        if (test_list) {
            // Test dequantize with list of tensors
            uint8_t num_tensors = (offset < size) ? (data[offset++] % 5 + 1) : 1; // 1-5 tensors
            
            std::vector<torch::Tensor> quantized_tensors;
            
            for (uint8_t i = 0; i < num_tensors; ++i) {
                try {
                    if (offset >= size) break;
                    
                    torch::Tensor qt = createQuantizedTensor(data, offset, size);
                    quantized_tensors.push_back(qt);
                } catch (const std::exception& e) {
                    // Continue with partial list if tensor creation fails
                    break;
                }
            }
            
            if (!quantized_tensors.empty()) {
                // Convert to IValue list for dequantize
                c10::List<torch::Tensor> tensor_list;
                for (const auto& t : quantized_tensors) {
                    tensor_list.push_back(t);
                }
                
                // Call dequantize on list
                try {
                    auto result = torch::dequantize(tensor_list);
                    
                    // Verify results
                    for (size_t i = 0; i < result.size(); ++i) {
                        if (!result[i].defined()) {
                            std::cerr << "Undefined tensor in result at index " << i << std::endl;
                        }
                        if (result[i].is_quantized()) {
                            std::cerr << "Result tensor still quantized at index " << i << std::endl;
                        }
                    }
                } catch (const c10::Error& e) {
                    // PyTorch errors are expected for edge cases
                    return 0;
                }
            }
        } else {
            // Test dequantize with single tensor
            torch::Tensor quantized = createQuantizedTensor(data, offset, size);
            
            // Call dequantize
            torch::Tensor dequantized = torch::dequantize(quantized);
            
            // Verify result
            if (!dequantized.defined()) {
                std::cerr << "Dequantized tensor is undefined" << std::endl;
            }
            if (dequantized.is_quantized()) {
                std::cerr << "Result is still quantized" << std::endl;
            }
            
            // Test edge cases with the dequantized tensor
            if (dequantized.defined()) {
                // Try various operations that might reveal issues
                auto sizes = dequantized.sizes();
                auto numel = dequantized.numel();
                auto dtype = dequantized.dtype();
                
                // Test with empty tensor
                if (numel == 0) {
                    auto empty_clone = dequantized.clone();
                }
                
                // Test view operations if tensor has elements
                if (numel > 0) {
                    try {
                        auto flat = dequantized.view({-1});
                    } catch (...) {
                        // View might fail for non-contiguous tensors
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (const c10::Error& e) {
        // PyTorch-specific errors
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}