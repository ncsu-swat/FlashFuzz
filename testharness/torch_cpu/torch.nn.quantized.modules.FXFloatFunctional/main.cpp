#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor if we have more data
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, create a tensor with same shape as input1
            input2 = torch::ones_like(input1);
        }
        
        // Get operation type from remaining data
        uint8_t op_type = 0;
        if (offset < Size) {
            op_type = Data[offset++] % 8; // Choose from available operations
        }
        
        // Apply different operations based on op_type
        torch::Tensor result;
        switch (op_type) {
            case 0:
                // add
                result = torch::add(input1, input2);
                break;
            case 1:
                // add_scalar
                if (offset < Size) {
                    float scalar = *reinterpret_cast<const float*>(Data + offset);
                    offset += sizeof(float);
                    result = torch::add(input1, scalar);
                } else {
                    result = torch::add(input1, 1.0f);
                }
                break;
            case 2:
                // mul
                result = torch::mul(input1, input2);
                break;
            case 3:
                // mul_scalar
                if (offset < Size) {
                    float scalar = *reinterpret_cast<const float*>(Data + offset);
                    offset += sizeof(float);
                    result = torch::mul(input1, scalar);
                } else {
                    result = torch::mul(input1, 2.0f);
                }
                break;
            case 4:
                // cat
                {
                    std::vector<torch::Tensor> tensors = {input1, input2};
                    int64_t dim = 0;
                    if (offset < Size && !input1.sizes().empty()) {
                        dim = Data[offset++] % input1.dim();
                    }
                    result = torch::cat(tensors, dim);
                }
                break;
            case 5:
                // add_relu
                result = torch::relu(torch::add(input1, input2));
                break;
            case 6:
                // mul_relu
                result = torch::relu(torch::mul(input1, input2));
                break;
            case 7:
                // hardtanh
                {
                    float min_val = -1.0f;
                    float max_val = 1.0f;
                    if (offset + sizeof(float) * 2 <= Size) {
                        min_val = *reinterpret_cast<const float*>(Data + offset);
                        offset += sizeof(float);
                        max_val = *reinterpret_cast<const float*>(Data + offset);
                        offset += sizeof(float);
                    }
                    result = torch::hardtanh(input1, min_val, max_val);
                }
                break;
        }
        
        // Force computation to ensure any potential errors are triggered
        result.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
