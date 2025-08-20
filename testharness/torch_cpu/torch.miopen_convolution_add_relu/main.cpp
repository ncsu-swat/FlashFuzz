#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight tensor
        torch::Tensor weight;
        if (offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create bias tensor
        torch::Tensor bias;
        if (offset < Size) {
            bias = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create z tensor (for addition)
        torch::Tensor z;
        if (offset < Size) {
            z = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Parse convolution parameters
        std::vector<int64_t> stride, padding, dilation;
        int64_t groups = 1;
        
        // Parse stride
        if (offset + 1 < Size) {
            uint8_t stride_size = Data[offset++] % 3 + 1;
            for (int i = 0; i < stride_size && offset + sizeof(int64_t) <= Size; i++) {
                int64_t s;
                std::memcpy(&s, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                s = std::abs(s) % 4 + 1;
                stride.push_back(s);
            }
        }
        
        // Parse padding
        if (offset + 1 < Size) {
            uint8_t padding_size = Data[offset++] % 3 + 1;
            for (int i = 0; i < padding_size && offset + sizeof(int64_t) <= Size; i++) {
                int64_t p;
                std::memcpy(&p, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                p = std::abs(p) % 3;
                padding.push_back(p);
            }
        }
        
        // Parse dilation
        if (offset + 1 < Size) {
            uint8_t dilation_size = Data[offset++] % 3 + 1;
            for (int i = 0; i < dilation_size && offset + sizeof(int64_t) <= Size; i++) {
                int64_t d;
                std::memcpy(&d, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                d = std::abs(d) % 3 + 1;
                dilation.push_back(d);
            }
        }
        
        // Parse groups
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % 4 + 1;
        }
        
        // Parse alpha parameter
        std::optional<at::Scalar> alpha = std::nullopt;
        if (offset + sizeof(float) <= Size) {
            float alpha_val;
            std::memcpy(&alpha_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            alpha = at::Scalar(alpha_val);
        }
        
        // Ensure tensors are on CUDA if MIOpen is being used
        if (torch::cuda::is_available()) {
            input = input.cuda();
            weight = weight.cuda();
            bias = bias.cuda();
            z = z.cuda();
            
            // Ensure tensors have correct dtype for MIOpen
            if (input.dtype() != torch::kFloat && input.dtype() != torch::kHalf) {
                input = input.to(torch::kFloat);
            }
            if (weight.dtype() != input.dtype()) {
                weight = weight.to(input.dtype());
            }
            if (bias.dtype() != input.dtype()) {
                bias = bias.to(input.dtype());
            }
            if (z.dtype() != input.dtype()) {
                z = z.to(input.dtype());
            }
            
            // Apply miopen_convolution_add_relu
            try {
                torch::Tensor output = torch::miopen_convolution_add_relu(
                    input, weight, z, alpha, bias, stride, padding, dilation, groups);
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
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