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

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t quant_params_byte = Data[offset++];
        
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            if (scale <= 0.0 || !std::isfinite(scale)) {
                scale = 1.0;
            }
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        torch::ScalarType quant_dtype = torch::kQInt8;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            switch (dtype_selector % 4) {
                case 0: quant_dtype = torch::kQInt8; break;
                case 1: quant_dtype = torch::kQUInt8; break;
                case 2: quant_dtype = torch::kQInt32; break;
                case 3: quant_dtype = torch::kQUInt4x2; break;
            }
        }
        
        torch::Tensor quantized_tensor;
        
        try {
            if (input_tensor.numel() == 0) {
                quantized_tensor = torch::empty({0}, torch::TensorOptions().dtype(quant_dtype));
            } else {
                auto float_tensor = input_tensor.to(torch::kFloat);
                quantized_tensor = torch::quantize_per_tensor(float_tensor, scale, zero_point, quant_dtype);
            }
        } catch (...) {
            quantized_tensor = torch::empty({1}, torch::TensorOptions().dtype(quant_dtype));
            quantized_tensor = torch::quantize_per_tensor(torch::zeros({1}), scale, zero_point, quant_dtype);
        }
        
        auto dequantized_tensor = torch::dequantize(quantized_tensor);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                if (input_tensor2.numel() > 0) {
                    auto float_tensor2 = input_tensor2.to(torch::kFloat);
                    auto quantized_tensor2 = torch::quantize_per_tensor(float_tensor2, scale, zero_point, quant_dtype);
                    auto dequantized_tensor2 = torch::dequantize(quantized_tensor2);
                }
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            std::vector<torch::Tensor> tensor_list;
            size_t num_tensors = std::min(static_cast<size_t>(5), (Size - offset) / 20);
            
            for (size_t i = 0; i < num_tensors && offset < Size; ++i) {
                try {
                    auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    if (tensor.numel() > 0) {
                        auto float_tensor = tensor.to(torch::kFloat);
                        auto quant_tensor = torch::quantize_per_tensor(float_tensor, scale, zero_point, quant_dtype);
                        tensor_list.push_back(quant_tensor);
                    }
                } catch (...) {
                    break;
                }
            }
            
            if (!tensor_list.empty()) {
                auto dequantized_list = torch::dequantize(tensor_list);
            }
        }
        
        try {
            auto empty_quantized = torch::empty({0}, torch::TensorOptions().dtype(quant_dtype));
            auto dequant_empty = torch::dequantize(empty_quantized);
        } catch (...) {
        }
        
        try {
            auto scalar_quantized = torch::quantize_per_tensor(torch::tensor(42.0f), scale, zero_point, quant_dtype);
            auto dequant_scalar = torch::dequantize(scalar_quantized);
        } catch (...) {
        }
        
        if (offset < Size) {
            int64_t large_dim = static_cast<int64_t>(Data[offset++]) * 1000 + 1;
            try {
                auto large_tensor = torch::zeros({large_dim});
                auto large_quantized = torch::quantize_per_tensor(large_tensor, scale, zero_point, quant_dtype);
                auto large_dequantized = torch::dequantize(large_quantized);
            } catch (...) {
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}