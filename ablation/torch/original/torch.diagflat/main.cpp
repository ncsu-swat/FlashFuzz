#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t offset_byte = Data[offset++];
        int64_t diagonal_offset = static_cast<int64_t>(static_cast<int8_t>(offset_byte));
        
        torch::diagflat(input_tensor, diagonal_offset);
        
        if (offset < Size) {
            uint8_t extreme_offset_bytes[8] = {0};
            size_t bytes_to_copy = std::min(static_cast<size_t>(8), Size - offset);
            std::memcpy(extreme_offset_bytes, Data + offset, bytes_to_copy);
            
            int64_t extreme_offset;
            std::memcpy(&extreme_offset, extreme_offset_bytes, sizeof(int64_t));
            
            torch::diagflat(input_tensor, extreme_offset);
            
            torch::diagflat(input_tensor, std::numeric_limits<int64_t>::max());
            torch::diagflat(input_tensor, std::numeric_limits<int64_t>::min());
        }
        
        torch::diagflat(input_tensor, 0);
        torch::diagflat(input_tensor, 1);
        torch::diagflat(input_tensor, -1);
        torch::diagflat(input_tensor, 10);
        torch::diagflat(input_tensor, -10);
        
        if (input_tensor.numel() > 0) {
            auto reshaped = input_tensor.reshape({-1});
            torch::diagflat(reshaped, diagonal_offset);
            
            if (input_tensor.dim() > 1) {
                auto flattened = input_tensor.flatten();
                torch::diagflat(flattened, diagonal_offset);
            }
        }
        
        if (input_tensor.dim() == 0) {
            torch::diagflat(input_tensor, diagonal_offset);
        }
        
        if (input_tensor.numel() == 1) {
            torch::diagflat(input_tensor, diagonal_offset);
        }
        
        auto empty_tensor = torch::empty({0});
        torch::diagflat(empty_tensor, diagonal_offset);
        
        auto large_1d = torch::ones({1000});
        torch::diagflat(large_1d, 0);
        
        if (input_tensor.dtype() != torch::kBool) {
            auto bool_tensor = input_tensor.to(torch::kBool);
            torch::diagflat(bool_tensor, diagonal_offset);
        }
        
        if (input_tensor.dtype() != torch::kComplexFloat && input_tensor.dtype() != torch::kComplexDouble) {
            if (input_tensor.dtype().isFloatingPoint()) {
                auto complex_tensor = torch::complex(input_tensor, torch::zeros_like(input_tensor));
                torch::diagflat(complex_tensor, diagonal_offset);
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}