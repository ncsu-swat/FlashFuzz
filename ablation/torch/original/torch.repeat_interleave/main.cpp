#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t operation_mode = Data[offset++] % 4;
        
        if (operation_mode == 0) {
            if (offset >= Size) {
                return 0;
            }
            int32_t repeats_scalar;
            if (offset + sizeof(int32_t) <= Size) {
                std::memcpy(&repeats_scalar, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
            } else {
                repeats_scalar = 1;
                offset = Size;
            }
            
            repeats_scalar = std::abs(repeats_scalar) % 100;
            if (repeats_scalar == 0) repeats_scalar = 1;
            
            auto result = torch::repeat_interleave(input_tensor, repeats_scalar);
        }
        else if (operation_mode == 1) {
            if (offset >= Size) {
                return 0;
            }
            
            auto repeats_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (repeats_tensor.numel() == 0) {
                return 0;
            }
            
            repeats_tensor = torch::abs(repeats_tensor).to(torch::kInt64);
            repeats_tensor = torch::clamp(repeats_tensor, 0, 50);
            
            auto result = torch::repeat_interleave(input_tensor, repeats_tensor);
        }
        else if (operation_mode == 2) {
            if (offset >= Size) {
                return 0;
            }
            
            int32_t repeats_scalar;
            if (offset + sizeof(int32_t) <= Size) {
                std::memcpy(&repeats_scalar, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
            } else {
                repeats_scalar = 1;
                offset = Size;
            }
            
            repeats_scalar = std::abs(repeats_scalar) % 100;
            if (repeats_scalar == 0) repeats_scalar = 1;
            
            if (offset >= Size) {
                return 0;
            }
            
            int32_t dim_raw;
            if (offset + sizeof(int32_t) <= Size) {
                std::memcpy(&dim_raw, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
            } else {
                dim_raw = 0;
                offset = Size;
            }
            
            int64_t dim = dim_raw;
            if (input_tensor.dim() > 0) {
                dim = dim % input_tensor.dim();
                if (dim < 0) dim += input_tensor.dim();
            } else {
                dim = 0;
            }
            
            auto result = torch::repeat_interleave(input_tensor, repeats_scalar, dim);
        }
        else {
            if (offset >= Size) {
                return 0;
            }
            
            auto repeats_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (repeats_tensor.numel() == 0) {
                return 0;
            }
            
            repeats_tensor = torch::abs(repeats_tensor).to(torch::kInt64);
            repeats_tensor = torch::clamp(repeats_tensor, 0, 50);
            
            if (offset >= Size) {
                return 0;
            }
            
            int32_t dim_raw;
            if (offset + sizeof(int32_t) <= Size) {
                std::memcpy(&dim_raw, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
            } else {
                dim_raw = 0;
                offset = Size;
            }
            
            int64_t dim = dim_raw;
            if (input_tensor.dim() > 0) {
                dim = dim % input_tensor.dim();
                if (dim < 0) dim += input_tensor.dim();
            } else {
                dim = 0;
            }
            
            if (offset < Size) {
                int64_t output_size_raw;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&output_size_raw, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else {
                    output_size_raw = -1;
                    offset = Size;
                }
                
                if (output_size_raw >= 0) {
                    int64_t output_size = std::abs(output_size_raw) % 10000;
                    auto result = torch::repeat_interleave(input_tensor, repeats_tensor, dim, output_size);
                } else {
                    auto result = torch::repeat_interleave(input_tensor, repeats_tensor, dim);
                }
            } else {
                auto result = torch::repeat_interleave(input_tensor, repeats_tensor, dim);
            }
        }
        
        if (offset < Size) {
            auto standalone_repeats = fuzzer_utils::createTensor(Data, Size, offset);
            if (standalone_repeats.numel() > 0) {
                standalone_repeats = torch::abs(standalone_repeats).to(torch::kInt64);
                standalone_repeats = torch::clamp(standalone_repeats, 0, 20);
                auto standalone_result = torch::repeat_interleave(standalone_repeats);
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