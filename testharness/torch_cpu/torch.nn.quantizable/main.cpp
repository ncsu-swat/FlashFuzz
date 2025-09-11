#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a linear module
        torch::nn::Linear linear_module = torch::nn::Linear(
            input_tensor.size(-1) > 0 ? input_tensor.size(-1) : 1, 
            10);
        
        // Test basic quantization operations
        if (offset + 2 < Size) {
            uint8_t qscheme_byte = Data[offset++];
            uint8_t dtype_byte = Data[offset++];
            
            // Select quantization scheme
            auto qscheme = qscheme_byte % 2 == 0 ? 
                torch::kPerTensorAffine : torch::kPerChannelAffine;
            
            // Select dtype for quantization
            auto dtype = dtype_byte % 2 == 0 ? 
                torch::kQInt8 : torch::kQUInt8;
            
            try {
                // Test quantization and dequantization
                double scale = 0.1;
                int64_t zero_point = 0;
                
                torch::Tensor quantized = torch::quantize_per_tensor(
                    input_tensor.to(torch::kFloat), scale, zero_point, dtype);
                torch::Tensor dequantized = quantized.dequantize();
                
                // Test linear module forward pass
                auto output = linear_module->forward(input_tensor);
                
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors during operations
                // This is expected for some invalid inputs
            }
        }
        
        // Test other modules if there's more data
        if (offset + 1 < Size) {
            uint8_t module_type = Data[offset++];
            
            switch (module_type % 4) {
                case 0: {
                    // Test Conv2d
                    int64_t in_channels = input_tensor.dim() > 1 ? input_tensor.size(1) : 1;
                    torch::nn::Conv2d conv = torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(in_channels, 16, 3).padding(1));
                    
                    // Reshape input if needed for Conv2d
                    torch::Tensor conv_input = input_tensor;
                    if (input_tensor.dim() < 4) {
                        std::vector<int64_t> new_shape = {1, in_channels, 8, 8};
                        conv_input = torch::ones(new_shape);
                    }
                    
                    try {
                        auto output = conv->forward(conv_input);
                    } catch (const c10::Error& e) {
                        // Expected for some inputs
                    }
                    break;
                }
                case 1: {
                    // Test ReLU
                    torch::nn::ReLU relu;
                    
                    try {
                        auto output = relu->forward(input_tensor);
                    } catch (const c10::Error& e) {
                        // Expected for some inputs
                    }
                    break;
                }
                case 2: {
                    // Test ELU
                    torch::nn::ELU elu;
                    
                    try {
                        auto output = elu->forward(input_tensor);
                    } catch (const c10::Error& e) {
                        // Expected for some inputs
                    }
                    break;
                }
                case 3: {
                    // Test MaxPool2d
                    torch::nn::MaxPool2d maxpool = torch::nn::MaxPool2d(
                        torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
                    
                    // Reshape input if needed for MaxPool2d
                    torch::Tensor pool_input = input_tensor;
                    if (input_tensor.dim() < 3) {
                        std::vector<int64_t> new_shape = {1, 3, 8, 8};
                        pool_input = torch::ones(new_shape);
                    }
                    
                    try {
                        auto output = maxpool->forward(pool_input);
                    } catch (const c10::Error& e) {
                        // Expected for some inputs
                    }
                    break;
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
