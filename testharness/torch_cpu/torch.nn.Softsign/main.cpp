#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Softsign module
        torch::nn::Softsign softsign;
        
        // Apply Softsign to the input tensor
        torch::Tensor output = softsign->forward(input);
        
        // Alternative direct function call
        torch::Tensor output2 = torch::nn::functional::softsign(input);
        
        // Try with different tensor options
        if (offset + 1 < Size) {
            // Create a tensor with different options
            torch::Tensor input2 = input.clone();
            
            // Try different tensor properties
            if (Data[offset] % 4 == 0) {
                input2 = input2.to(torch::kFloat16);
            } else if (Data[offset] % 4 == 1) {
                input2 = input2.to(torch::kDouble);
            } else if (Data[offset] % 4 == 2) {
                input2 = input2.contiguous();
            } else {
                input2 = input2.to(torch::kBFloat16);
            }
            
            // Apply Softsign to the modified tensor
            torch::Tensor output3 = softsign->forward(input2);
            
            offset++;
        }
        
        // Try with non-contiguous tensor if possible
        if (input.dim() > 1 && input.size(0) > 1) {
            torch::Tensor non_contiguous = input.transpose(0, input.dim() - 1);
            if (!non_contiguous.is_contiguous()) {
                torch::Tensor output4 = softsign->forward(non_contiguous);
            }
        }
        
        // Try with empty tensor
        if (offset + 1 < Size && Data[offset] % 2 == 0) {
            std::vector<int64_t> empty_shape = {0};
            torch::Tensor empty_tensor = torch::empty(empty_shape, input.options());
            torch::Tensor empty_output = softsign->forward(empty_tensor);
        }
        
        // Try with scalar tensor
        if (offset + 1 < Size) {
            torch::Tensor scalar_tensor;
            if (Data[offset] % 3 == 0) {
                scalar_tensor = torch::tensor(3.14f);
            } else if (Data[offset] % 3 == 1) {
                scalar_tensor = torch::tensor(-100.0f);
            } else {
                scalar_tensor = torch::tensor(0.0f);
            }
            
            torch::Tensor scalar_output = softsign->forward(scalar_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}