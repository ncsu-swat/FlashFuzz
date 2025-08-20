#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (batch_size, channels, sequence_length)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for Conv1d from the remaining data
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        int64_t kernel_size = 1;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Parse additional parameters if data available
        if (offset + 8 <= Size) {
            out_channels = (Data[offset] % 8) + 1;
            offset++;
            
            kernel_size = (Data[offset] % 5) + 1;
            offset++;
            
            stride = (Data[offset] % 3) + 1;
            offset++;
            
            padding = Data[offset] % 3;
            offset++;
            
            dilation = (Data[offset] % 2) + 1;
            offset++;
            
            // Ensure groups divides in_channels
            if (in_channels > 0) {
                groups = (Data[offset] % in_channels) + 1;
            }
            offset++;
            
            bias = Data[offset] % 2 == 0;
            offset++;
            
            // Ensure in_channels is divisible by groups
            if (groups > 0 && in_channels % groups != 0) {
                in_channels = groups; // Make it divisible
            }
        }
        
        // Create a regular Conv1d module and apply dynamic quantization
        torch::nn::Conv1dOptions options = 
            torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias);
        
        auto conv1d = torch::nn::Conv1d(options);
        
        // Apply dynamic quantization to the module
        torch::jit::Module quantized_module = torch::jit::quantized::quantize_dynamic(
            conv1d, {torch::kQInt8}, torch::kFBGEMM);
        
        // Apply the Conv1d operation
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        torch::Tensor output = quantized_module.forward(inputs).toTensor();
        
        // Use the output to prevent optimization from removing the computation
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}