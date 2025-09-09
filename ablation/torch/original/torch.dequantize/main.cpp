#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }

        uint8_t operation_selector = Data[offset++];
        bool test_list_version = (operation_selector % 2) == 1;
        
        if (test_list_version) {
            uint8_t num_tensors_byte = Data[offset++];
            uint8_t num_tensors = (num_tensors_byte % 4) + 1;
            
            std::vector<torch::Tensor> quantized_tensors;
            
            for (uint8_t i = 0; i < num_tensors; ++i) {
                if (offset >= Size) break;
                
                torch::Tensor base_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                if (offset >= Size) break;
                uint8_t qscheme_selector = Data[offset++];
                
                torch::Tensor quantized_tensor;
                
                if (qscheme_selector % 4 == 0) {
                    if (offset + 8 <= Size) {
                        double scale_raw;
                        std::memcpy(&scale_raw, Data + offset, 8);
                        offset += 8;
                        double scale = std::abs(scale_raw);
                        if (scale == 0.0) scale = 1.0;
                        if (std::isnan(scale) || std::isinf(scale)) scale = 1.0;
                        
                        int64_t zero_point = 0;
                        if (offset + 8 <= Size) {
                            std::memcpy(&zero_point, Data + offset, 8);
                            offset += 8;
                            zero_point = zero_point % 256;
                        }
                        
                        quantized_tensor = torch::quantize_per_tensor(base_tensor.to(torch::kFloat), scale, zero_point, torch::kQUInt8);
                    } else {
                        quantized_tensor = torch::quantize_per_tensor(base_tensor.to(torch::kFloat), 1.0, 0, torch::kQUInt8);
                    }
                } else if (qscheme_selector % 4 == 1) {
                    if (offset + 8 <= Size) {
                        double scale_raw;
                        std::memcpy(&scale_raw, Data + offset, 8);
                        offset += 8;
                        double scale = std::abs(scale_raw);
                        if (scale == 0.0) scale = 1.0;
                        if (std::isnan(scale) || std::isinf(scale)) scale = 1.0;
                        
                        int64_t zero_point = 0;
                        if (offset + 8 <= Size) {
                            std::memcpy(&zero_point, Data + offset, 8);
                            offset += 8;
                            zero_point = zero_point % 256 - 128;
                        }
                        
                        quantized_tensor = torch::quantize_per_tensor(base_tensor.to(torch::kFloat), scale, zero_point, torch::kQInt8);
                    } else {
                        quantized_tensor = torch::quantize_per_tensor(base_tensor.to(torch::kFloat), 1.0, 0, torch::kQInt8);
                    }
                } else if (qscheme_selector % 4 == 2) {
                    if (offset + 8 <= Size) {
                        double scale_raw;
                        std::memcpy(&scale_raw, Data + offset, 8);
                        offset += 8;
                        double scale = std::abs(scale_raw);
                        if (scale == 0.0) scale = 1.0;
                        if (std::isnan(scale) || std::isinf(scale)) scale = 1.0;
                        
                        int64_t zero_point = 0;
                        if (offset + 8 <= Size) {
                            std::memcpy(&zero_point, Data + offset, 8);
                            offset += 8;
                        }
                        
                        quantized_tensor = torch::quantize_per_tensor(base_tensor.to(torch::kFloat), scale, zero_point, torch::kQInt32);
                    } else {
                        quantized_tensor = torch::quantize_per_tensor(base_tensor.to(torch::kFloat), 1.0, 0, torch::kQInt32);
                    }
                } else {
                    auto float_tensor = base_tensor.to(torch::kFloat);
                    if (float_tensor.dim() == 0) {
                        quantized_tensor = torch::quantize_per_tensor(float_tensor, 1.0, 0, torch::kQUInt8);
                    } else {
                        int64_t axis = 0;
                        if (offset < Size) {
                            axis = Data[offset++] % float_tensor.dim();
                        }
                        
                        auto scales = torch::ones({float_tensor.size(axis)}, torch::kDouble);
                        auto zero_points = torch::zeros({float_tensor.size(axis)}, torch::kLong);
                        
                        if (offset + scales.numel() * 8 <= Size) {
                            for (int64_t j = 0; j < scales.numel(); ++j) {
                                double scale_raw;
                                std::memcpy(&scale_raw, Data + offset, 8);
                                offset += 8;
                                double scale = std::abs(scale_raw);
                                if (scale == 0.0) scale = 1.0;
                                if (std::isnan(scale) || std::isinf(scale)) scale = 1.0;
                                scales[j] = scale;
                            }
                        }
                        
                        quantized_tensor = torch::quantize_per_channel(float_tensor, scales, zero_points, axis, torch::kQUInt8);
                    }
                }
                
                quantized_tensors.push_back(quantized_tensor);
            }
            
            if (!quantized_tensors.empty()) {
                auto dequantized_tensors = torch::dequantize(quantized_tensors);
            }
            
        } else {
            torch::Tensor base_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset >= Size) {
                return 0;
            }
            
            uint8_t qscheme_selector = Data[offset++];
            torch::Tensor quantized_tensor;
            
            if (qscheme_selector % 4 == 0) {
                if (offset + 8 <= Size) {
                    double scale_raw;
                    std::memcpy(&scale_raw, Data + offset, 8);
                    offset += 8;
                    double scale = std::abs(scale_raw);
                    if (scale == 0.0) scale = 1.0;
                    if (std::isnan(scale) || std::isinf(scale)) scale = 1.0;
                    
                    int64_t zero_point = 0;
                    if (offset + 8 <= Size) {
                        std::memcpy(&zero_point, Data + offset, 8);
                        offset += 8;
                        zero_point = zero_point % 256;
                    }
                    
                    quantized_tensor = torch::quantize_per_tensor(base_tensor.to(torch::kFloat), scale, zero_point, torch::kQUInt8);
                } else {
                    quantized_tensor = torch::quantize_per_tensor(base_tensor.to(torch::kFloat), 1.0, 0, torch::kQUInt8);
                }
            } else if (qscheme_selector % 4 == 1) {
                if (offset + 8 <= Size) {
                    double scale_raw;
                    std::memcpy(&scale_raw, Data + offset, 8);
                    offset += 8;
                    double scale = std::abs(scale_raw);
                    if (scale == 0.0) scale = 1.0;
                    if (std::isnan(scale) || std::isinf(scale)) scale = 1.0;
                    
                    int64_t zero_point = 0;
                    if (offset + 8 <= Size) {
                        std::memcpy(&zero_point, Data + offset, 8);
                        offset += 8;
                        zero_point = zero_point % 256 - 128;
                    }
                    
                    quantized_tensor = torch::quantize_per_tensor(base_tensor.to(torch::kFloat), scale, zero_point, torch::kQInt8);
                } else {
                    quantized_tensor = torch::quantize_per_tensor(base_tensor.to(torch::kFloat), 1.0, 0, torch::kQInt8);
                }
            } else if (qscheme_selector % 4 == 2) {
                if (offset + 8 <= Size) {
                    double scale_raw;
                    std::memcpy(&scale_raw, Data + offset, 8);
                    offset += 8;
                    double scale = std::abs(scale_raw);
                    if (scale == 0.0) scale = 1.0;
                    if (std::isnan(scale) || std::isinf(scale)) scale = 1.0;
                    
                    int64_t zero_point = 0;
                    if (offset + 8 <= Size) {
                        std::memcpy(&zero_point, Data + offset, 8);
                        offset += 8;
                    }
                    
                    quantized_tensor = torch::quantize_per_tensor(base_tensor.to(torch::kFloat), scale, zero_point, torch::kQInt32);
                } else {
                    quantized_tensor = torch::quantize_per_tensor(base_tensor.to(torch::kFloat), 1.0, 0, torch::kQInt32);
                }
            } else {
                auto float_tensor = base_tensor.to(torch::kFloat);
                if (float_tensor.dim() == 0) {
                    quantized_tensor = torch::quantize_per_tensor(float_tensor, 1.0, 0, torch::kQUInt8);
                } else {
                    int64_t axis = 0;
                    if (offset < Size) {
                        axis = Data[offset++] % float_tensor.dim();
                    }
                    
                    auto scales = torch::ones({float_tensor.size(axis)}, torch::kDouble);
                    auto zero_points = torch::zeros({float_tensor.size(axis)}, torch::kLong);
                    
                    if (offset + scales.numel() * 8 <= Size) {
                        for (int64_t j = 0; j < scales.numel(); ++j) {
                            double scale_raw;
                            std::memcpy(&scale_raw, Data + offset, 8);
                            offset += 8;
                            double scale = std::abs(scale_raw);
                            if (scale == 0.0) scale = 1.0;
                            if (std::isnan(scale) || std::isinf(scale)) scale = 1.0;
                            scales[j] = scale;
                        }
                    }
                    
                    quantized_tensor = torch::quantize_per_channel(float_tensor, scales, zero_points, axis, torch::kQUInt8);
                }
            }
            
            auto dequantized_tensor = torch::dequantize(quantized_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}