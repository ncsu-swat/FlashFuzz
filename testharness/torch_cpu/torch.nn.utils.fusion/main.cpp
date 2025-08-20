#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors for fusion
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create more tensors if we have data left
        std::vector<torch::Tensor> tensors;
        tensors.push_back(tensor1);
        
        // Try to create 1-3 more tensors if we have enough data
        for (int i = 0; i < 3 && offset + 4 < Size; i++) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            tensors.push_back(tensor);
        }
        
        // Test different fusion-like operations using available PyTorch functions
        if (offset < Size) {
            uint8_t fusion_type = Data[offset++] % 3;
            
            switch (fusion_type) {
                case 0: {
                    // Test dropout (simulating fused_dropout)
                    if (!tensors.empty()) {
                        float p = 0.5;
                        if (offset < Size) {
                            // Get a probability between 0 and 1
                            p = static_cast<float>(Data[offset++]) / 255.0f;
                        }
                        
                        auto result = torch::dropout(tensors[0], p, true);
                    }
                    break;
                }
                
                case 1: {
                    // Test linear + relu (simulating fused_linear_relu)
                    if (tensors.size() >= 2) {
                        try {
                            auto linear_result = torch::linear(tensors[0], tensors[1], 
                                tensors.size() > 2 ? tensors[2] : torch::Tensor());
                            auto result = torch::relu(linear_result);
                        } catch (...) {
                            // Catch any exceptions from the operation itself
                        }
                    }
                    break;
                }
                
                case 2: {
                    // Test conv2d + batch_norm + relu (simulating fused_conv_bn_relu)
                    if (tensors.size() >= 3) {
                        try {
                            // Extract parameters for conv2d
                            std::vector<int64_t> stride = {1, 1};
                            std::vector<int64_t> padding = {0, 0};
                            std::vector<int64_t> dilation = {1, 1};
                            int64_t groups = 1;
                            
                            // Use remaining data to set parameters if available
                            if (offset + 3 < Size) {
                                stride[0] = static_cast<int64_t>(Data[offset++]) % 4 + 1;
                                stride[1] = static_cast<int64_t>(Data[offset++]) % 4 + 1;
                                padding[0] = static_cast<int64_t>(Data[offset++]) % 3;
                                padding[1] = static_cast<int64_t>(Data[offset++]) % 3;
                            }
                            
                            if (offset < Size) {
                                groups = static_cast<int64_t>(Data[offset++]) % 4 + 1;
                            }
                            
                            auto conv_result = torch::conv2d(tensors[0], tensors[1], torch::Tensor(),
                                stride, padding, dilation, groups);
                            auto result = torch::relu(conv_result);
                        } catch (...) {
                            // Catch any exceptions from the operation itself
                        }
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