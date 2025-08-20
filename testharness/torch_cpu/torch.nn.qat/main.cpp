#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t module_type = Data[offset++] % 5;
        
        torch::nn::AnyModule qat_module;
        
        switch (module_type) {
            case 0: {
                auto linear = torch::nn::Linear(input.size(1), input.size(0));
                qat_module = linear;
                break;
            }
            case 1: {
                auto relu = torch::nn::ReLU();
                qat_module = relu;
                break;
            }
            case 2: {
                if (input.dim() >= 2) {
                    int64_t in_channels = input.size(1);
                    int64_t out_channels = std::max<int64_t>(1, in_channels);
                    int64_t kernel_size = 3;
                    
                    auto conv = torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                            .stride(1)
                            .padding(1)
                            .dilation(1)
                            .groups(1)
                            .bias(true)
                    );
                    
                    qat_module = conv;
                } else {
                    auto relu = torch::nn::ReLU();
                    qat_module = relu;
                }
                break;
            }
            case 3: {
                auto embedding = torch::nn::Embedding(
                    std::max<int64_t>(1, input.size(0)), 
                    std::max<int64_t>(1, input.size(1)));
                qat_module = embedding;
                break;
            }
            case 4: {
                if (input.dim() >= 2) {
                    int64_t in_channels = input.size(1);
                    int64_t out_channels = std::max<int64_t>(1, in_channels);
                    int64_t kernel_size = 3;
                    
                    auto conv = torch::nn::Conv1d(
                        torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                            .stride(1)
                            .padding(1)
                            .dilation(1)
                            .groups(1)
                            .bias(true)
                    );
                    
                    qat_module = conv;
                } else {
                    auto relu = torch::nn::ReLU();
                    qat_module = relu;
                }
                break;
            }
            default: {
                auto relu = torch::nn::ReLU();
                qat_module = relu;
                break;
            }
        }
        
        try {
            qat_module.ptr()->train();
            
            torch::Tensor output;
            
            if (module_type == 3) {
                torch::Tensor indices = torch::randint(0, std::max<int64_t>(1, input.size(0)), {input.size(0)});
                output = qat_module.forward(indices);
            } else {
                output = qat_module.forward(input);
            }
            
            qat_module.ptr()->eval();
            
            if (module_type == 3) {
                torch::Tensor indices = torch::randint(0, std::max<int64_t>(1, input.size(0)), {input.size(0)});
                output = qat_module.forward(indices);
            } else {
                output = qat_module.forward(input);
            }
            
            // Simulate quantization by converting to int8 and back
            auto quantized_output = torch::quantize_per_tensor(output, 0.1, 10, torch::kQInt8);
            auto dequantized_output = torch::dequantize(quantized_output);
            
        } catch (...) {
            // Handle exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}