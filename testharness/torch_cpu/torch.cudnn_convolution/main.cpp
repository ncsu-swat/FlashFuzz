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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight tensor
        torch::Tensor weight;
        if (offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a compatible weight tensor
            auto input_size = input.sizes();
            if (input_size.size() >= 2) {
                // For convolution, weight should be [out_channels, in_channels/groups, kernel_size...]
                std::vector<int64_t> weight_size = {
                    std::max(int64_t(1), input_size[0]), // out_channels
                    std::max(int64_t(1), input_size[1])  // in_channels
                };
                
                // Add kernel dimensions (at least 1x1)
                for (size_t i = 2; i < input_size.size(); i++) {
                    weight_size.push_back(1);
                }
                
                weight = torch::ones(weight_size, input.options());
            } else {
                // Default weight for low-dimensional input
                weight = torch::ones({1, 1, 1, 1}, input.options());
            }
        }
        
        // Parse padding
        std::vector<int64_t> padding;
        if (offset + 2 <= Size) {
            uint16_t padding_val;
            std::memcpy(&padding_val, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            padding_val = padding_val % 5; // Limit padding size
            padding = {padding_val, padding_val};
        } else {
            padding = {0, 0};
        }
        
        // Parse stride
        std::vector<int64_t> stride;
        if (offset + 2 <= Size) {
            uint16_t stride_val;
            std::memcpy(&stride_val, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            stride_val = (stride_val % 3) + 1; // Stride between 1-3
            stride = {stride_val, stride_val};
        } else {
            stride = {1, 1};
        }
        
        // Parse dilation
        std::vector<int64_t> dilation;
        if (offset + 2 <= Size) {
            uint16_t dilation_val;
            std::memcpy(&dilation_val, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            dilation_val = (dilation_val % 2) + 1; // Dilation between 1-2
            dilation = {dilation_val, dilation_val};
        } else {
            dilation = {1, 1};
        }
        
        // Parse groups
        int64_t groups = 1;
        if (offset + 1 <= Size) {
            uint8_t groups_val = Data[offset++];
            groups = (groups_val % 4) + 1; // Groups between 1-4
        }
        
        // Parse benchmark flag
        bool benchmark = false;
        if (offset < Size) {
            benchmark = Data[offset++] & 0x1;
        }
        
        // Parse deterministic flag
        bool deterministic = false;
        if (offset < Size) {
            deterministic = Data[offset++] & 0x1;
        }
        
        // Parse allow_tf32 flag
        bool allow_tf32 = false;
        if (offset < Size) {
            allow_tf32 = Data[offset++] & 0x1;
        }
        
        // Ensure input and weight have compatible dtypes for convolution
        if (input.scalar_type() == torch::kBool || 
            input.scalar_type() == torch::kInt8 || 
            input.scalar_type() == torch::kInt16 || 
            input.scalar_type() == torch::kInt32 || 
            input.scalar_type() == torch::kInt64 ||
            input.scalar_type() == torch::kUInt8) {
            input = input.to(torch::kFloat);
        }
        
        if (weight.scalar_type() == torch::kBool || 
            weight.scalar_type() == torch::kInt8 || 
            weight.scalar_type() == torch::kInt16 || 
            weight.scalar_type() == torch::kInt32 || 
            weight.scalar_type() == torch::kInt64 ||
            weight.scalar_type() == torch::kUInt8) {
            weight = weight.to(torch::kFloat);
        }
        
        // Ensure input and weight have same dtype
        if (input.scalar_type() != weight.scalar_type()) {
            weight = weight.to(input.scalar_type());
        }
        
        // Move tensors to CUDA if available
        if (torch::cuda::is_available()) {
            input = input.cuda();
            weight = weight.cuda();
        }
        
        // Apply cudnn_convolution
        torch::Tensor output;
        try {
            output = torch::cudnn_convolution(
                input, weight, 
                padding, stride, dilation, 
                groups, benchmark, deterministic, allow_tf32
            );
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
            return 0;
        }
        
        // Ensure we use the output to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        return 0; // discard the input
    }
    return 0; // keep the input
}
