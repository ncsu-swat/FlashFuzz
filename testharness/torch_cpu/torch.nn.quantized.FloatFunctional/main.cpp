#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        
        // Create a FloatFunctional module
        torch::nn::quantized::FloatFunctional floatFunc;
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // If we have more data, create a second tensor
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for second tensor, clone the first one
            input2 = input1.clone();
        }
        
        // Get operation type from the next byte if available
        uint8_t op_type = 0;
        if (offset < Size) {
            op_type = Data[offset++];
        }
        
        // Apply different operations based on the op_type
        switch (op_type % 6) {
            case 0: {
                // Test add operation
                torch::nn::functional::add(input1, input2);
                break;
            }
                
            case 1: {
                // Test add_scalar operation
                float scalar_value = 1.0f;
                if (offset + sizeof(float) <= Size) {
                    std::memcpy(&scalar_value, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
                torch::add(input1, scalar_value);
                break;
            }
                
            case 2: {
                // Test mul operation
                torch::mul(input1, input2);
                break;
            }
                
            case 3: {
                // Test mul_scalar operation
                float mul_scalar = 1.0f;
                if (offset + sizeof(float) <= Size) {
                    std::memcpy(&mul_scalar, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
                torch::mul(input1, mul_scalar);
                break;
            }
                
            case 4: {
                // Test cat operation
                std::vector<torch::Tensor> tensors_to_cat = {input1, input2};
                int64_t dim = 0;
                if (offset < Size) {
                    dim = static_cast<int64_t>(Data[offset++]) % (input1.dim() + 1);
                }
                torch::cat(tensors_to_cat, dim);
                break;
            }
                
            case 5: {
                // Test add_relu operation
                torch::relu(torch::add(input1, input2));
                break;
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