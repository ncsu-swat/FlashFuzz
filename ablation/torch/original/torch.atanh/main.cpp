#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        torch::Tensor result = torch::atanh(input_tensor);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor out_tensor = torch::empty_like(input_tensor2);
            torch::atanh_out(out_tensor, input_tensor2);
        }
        
        if (offset < Size) {
            auto input_tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
            input_tensor3.atanh_();
        }
        
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            auto special_values = torch::tensor({-1.0, -0.9999, -0.5, 0.0, 0.5, 0.9999, 1.0}, torch::TensorOptions().dtype(torch::kFloat));
            if (dtype != torch::kBool && dtype != torch::kInt8 && dtype != torch::kUInt8 && 
                dtype != torch::kInt16 && dtype != torch::kInt32 && dtype != torch::kInt64) {
                special_values = special_values.to(dtype);
                torch::Tensor special_result = torch::atanh(special_values);
            }
        }
        
        if (offset + 8 < Size) {
            int64_t shape_val;
            std::memcpy(&shape_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            int64_t dim_size = std::abs(shape_val) % 100 + 1;
            auto large_tensor = torch::randn({dim_size, dim_size}, torch::TensorOptions().dtype(torch::kFloat));
            large_tensor = large_tensor * 0.99;
            torch::Tensor large_result = torch::atanh(large_tensor);
        }
        
        if (offset < Size) {
            auto empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat));
            torch::Tensor empty_result = torch::atanh(empty_tensor);
        }
        
        if (offset < Size) {
            auto scalar_tensor = torch::scalar_tensor(0.5, torch::TensorOptions().dtype(torch::kFloat));
            torch::Tensor scalar_result = torch::atanh(scalar_tensor);
        }
        
        if (offset + 4 < Size) {
            float val;
            std::memcpy(&val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            auto boundary_tensor = torch::tensor({val}, torch::TensorOptions().dtype(torch::kFloat));
            torch::Tensor boundary_result = torch::atanh(boundary_tensor);
        }
        
        if (offset < Size) {
            auto inf_nan_tensor = torch::tensor({std::numeric_limits<float>::infinity(), 
                                               -std::numeric_limits<float>::infinity(),
                                               std::numeric_limits<float>::quiet_NaN()}, 
                                               torch::TensorOptions().dtype(torch::kFloat));
            torch::Tensor inf_nan_result = torch::atanh(inf_nan_tensor);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}