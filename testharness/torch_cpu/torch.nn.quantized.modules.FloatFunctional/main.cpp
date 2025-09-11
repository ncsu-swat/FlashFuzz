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
        
        // If we have more data, create a second tensor for binary operations
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, create a tensor with the same shape as input1
            input2 = torch::ones_like(input1);
        }
        
        // Get an operation selector from the data if available
        uint8_t op_selector = 0;
        if (offset < Size) {
            op_selector = Data[offset++];
        }
        
        // Apply different operations based on the selector
        switch (op_selector % 7) {
            case 0:
                // Test add operation
                torch::add(input1, input2);
                break;
                
            case 1:
                // Test add_scalar operation
                torch::add(input1, 1.0);
                break;
                
            case 2:
                // Test mul operation
                torch::mul(input1, input2);
                break;
                
            case 3:
                // Test mul_scalar operation
                torch::mul(input1, 2.0);
                break;
                
            case 4:
                // Test cat operation
                {
                    std::vector<torch::Tensor> tensors = {input1, input2};
                    int64_t dim = 0;
                    if (!input1.sizes().empty()) {
                        if (offset < Size) {
                            dim = static_cast<int64_t>(Data[offset++]) % input1.dim();
                        }
                        torch::cat(tensors, dim);
                    }
                }
                break;
                
            case 5:
                // Test add_relu operation
                torch::relu(torch::add(input1, input2));
                break;
                
            case 6:
                // Test mul_relu operation
                torch::relu(torch::mul(input1, input2));
                break;
        }
        
        // Test quantization-related functionality
        if (offset < Size) {
            double scale = 1.0 / 256.0;
            int64_t zero_point = 0;
            
            // Try to quantize tensors
            torch::quantize_per_tensor(input1, scale, zero_point, torch::kQUInt8);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
