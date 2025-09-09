#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            torch::diag_embed(input_tensor);
            return 0;
        }
        
        int offset_val = static_cast<int>(static_cast<int8_t>(Data[offset++]));
        
        if (offset >= Size) {
            torch::diag_embed(input_tensor, offset_val);
            return 0;
        }
        
        int dim1_val = static_cast<int>(static_cast<int8_t>(Data[offset++]));
        
        if (offset >= Size) {
            torch::diag_embed(input_tensor, offset_val, dim1_val);
            return 0;
        }
        
        int dim2_val = static_cast<int>(static_cast<int8_t>(Data[offset++]));
        
        torch::diag_embed(input_tensor, offset_val, dim1_val, dim2_val);
        
        if (input_tensor.numel() == 0) {
            torch::diag_embed(input_tensor, 0);
        }
        
        if (input_tensor.dim() > 0) {
            torch::diag_embed(input_tensor, -input_tensor.size(-1));
            torch::diag_embed(input_tensor, input_tensor.size(-1));
        }
        
        torch::diag_embed(input_tensor, 0, -1, -2);
        torch::diag_embed(input_tensor, 0, -2, -1);
        
        if (input_tensor.dim() >= 2) {
            torch::diag_embed(input_tensor, 0, 0, 1);
            torch::diag_embed(input_tensor, 0, 1, 0);
        }
        
        torch::diag_embed(input_tensor, 1000);
        torch::diag_embed(input_tensor, -1000);
        
        torch::diag_embed(input_tensor, 0, 1000, 1001);
        torch::diag_embed(input_tensor, 0, -1000, -1001);
        
        auto squeezed = input_tensor.squeeze();
        if (squeezed.dim() > 0) {
            torch::diag_embed(squeezed);
        }
        
        auto unsqueezed = input_tensor.unsqueeze(0);
        torch::diag_embed(unsqueezed);
        
        if (input_tensor.dim() > 1) {
            auto flattened = input_tensor.flatten();
            torch::diag_embed(flattened);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}