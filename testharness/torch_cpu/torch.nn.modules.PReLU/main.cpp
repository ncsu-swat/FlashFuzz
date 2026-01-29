#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a parameter for weight initialization
        float weight_init = 0.25f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&weight_init, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize weight_init to avoid NaN/Inf issues
            if (!std::isfinite(weight_init)) {
                weight_init = 0.25f;
            }
        }
        
        // Get a boolean to determine if we should use num_parameters > 1
        bool use_channel_wise = false;
        if (offset < Size) {
            use_channel_wise = Data[offset++] & 0x1;
        }
        
        // Test 1: Single parameter PReLU (works with any input)
        {
            torch::nn::PReLU prelu(torch::nn::PReLUOptions().num_parameters(1).init(weight_init));
            torch::Tensor output = prelu->forward(input);
            
            // Test eval mode
            prelu->eval();
            torch::Tensor output_eval = prelu->forward(input);
            
            // Test train mode
            prelu->train();
            torch::Tensor output_train = prelu->forward(input);
        }
        
        // Test 2: Channel-wise PReLU (requires input.dim() >= 2)
        if (use_channel_wise && input.dim() >= 2) {
            try {
                int64_t num_channels = input.size(1);
                if (num_channels > 0 && num_channels <= 1024) {
                    torch::nn::PReLU prelu_ch(torch::nn::PReLUOptions()
                        .num_parameters(num_channels)
                        .init(weight_init));
                    torch::Tensor output_ch = prelu_ch->forward(input);
                }
            } catch (...) {
                // Channel-wise PReLU may fail for certain input shapes
            }
        }
        
        // Test 3: PReLU with modified weights
        {
            torch::nn::PReLU prelu_mod(torch::nn::PReLUOptions().num_parameters(1).init(0.25f));
            
            // Test with zero weights
            prelu_mod->weight.data().zero_();
            torch::Tensor output_zero = prelu_mod->forward(input);
            
            // Test with negative weights (leaky ReLU with negative slope)
            prelu_mod->weight.data().fill_(-0.5f);
            torch::Tensor output_neg = prelu_mod->forward(input);
            
            // Test with weight from fuzzer data
            if (offset + sizeof(float) <= Size) {
                float fuzz_weight;
                std::memcpy(&fuzz_weight, Data + offset, sizeof(float));
                offset += sizeof(float);
                if (std::isfinite(fuzz_weight)) {
                    prelu_mod->weight.data().fill_(fuzz_weight);
                    torch::Tensor output_fuzz = prelu_mod->forward(input);
                }
            }
        }
        
        // Test 4: PReLU with different input types
        try {
            torch::nn::PReLU prelu_type(torch::nn::PReLUOptions().num_parameters(1));
            if (input.is_floating_point()) {
                torch::Tensor output = prelu_type->forward(input);
            } else {
                // Convert to float for PReLU
                torch::Tensor float_input = input.to(torch::kFloat32);
                torch::Tensor output = prelu_type->forward(float_input);
            }
        } catch (...) {
            // Type conversion may fail
        }
        
        // Test 5: PReLU with contiguous/non-contiguous tensors
        if (input.dim() >= 2) {
            try {
                torch::nn::PReLU prelu_nc(torch::nn::PReLUOptions().num_parameters(1));
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor output_nc = prelu_nc->forward(transposed);
            } catch (...) {
                // May fail for certain configurations
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}