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
        
        uint8_t num_tensors_byte = Data[offset++];
        uint8_t num_tensors = (num_tensors_byte % 10) + 1;
        
        std::vector<torch::Tensor> tensors;
        tensors.reserve(num_tensors);
        
        for (uint8_t i = 0; i < num_tensors; ++i) {
            if (offset >= Size) {
                break;
            }
            
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (const std::exception &e) {
                if (tensors.empty()) {
                    return 0;
                }
                break;
            }
        }
        
        if (tensors.empty()) {
            return 0;
        }
        
        torch::Tensor result = torch::dstack(tensors);
        
        if (offset < Size) {
            uint8_t use_out_tensor = Data[offset++];
            if (use_out_tensor % 2 == 1 && offset < Size) {
                try {
                    torch::Tensor out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    torch::dstack_out(out_tensor, tensors);
                } catch (const std::exception &e) {
                }
            }
        }
        
        torch::Tensor single_tensor_case = torch::randn({2, 3});
        std::vector<torch::Tensor> single_vec = {single_tensor_case};
        torch::Tensor single_result = torch::dstack(single_vec);
        
        torch::Tensor empty_tensor = torch::empty({0});
        std::vector<torch::Tensor> with_empty = {empty_tensor};
        try {
            torch::Tensor empty_result = torch::dstack(with_empty);
        } catch (const std::exception &e) {
        }
        
        if (tensors.size() >= 2) {
            std::vector<torch::Tensor> mixed_shapes;
            mixed_shapes.push_back(torch::randn({1}));
            mixed_shapes.push_back(torch::randn({2, 3}));
            try {
                torch::Tensor mixed_result = torch::dstack(mixed_shapes);
            } catch (const std::exception &e) {
            }
        }
        
        std::vector<torch::Tensor> large_tensors;
        try {
            large_tensors.push_back(torch::randn({1000, 1000}));
            large_tensors.push_back(torch::randn({1000, 1000}));
            torch::Tensor large_result = torch::dstack(large_tensors);
        } catch (const std::exception &e) {
        }
        
        std::vector<torch::Tensor> zero_dim_tensors;
        zero_dim_tensors.push_back(torch::tensor(1.0));
        zero_dim_tensors.push_back(torch::tensor(2.0));
        try {
            torch::Tensor zero_dim_result = torch::dstack(zero_dim_tensors);
        } catch (const std::exception &e) {
        }
        
        if (tensors.size() > 0) {
            std::vector<torch::Tensor> different_dtypes;
            different_dtypes.push_back(tensors[0].to(torch::kFloat));
            different_dtypes.push_back(torch::randn_like(tensors[0]).to(torch::kDouble));
            try {
                torch::Tensor dtype_result = torch::dstack(different_dtypes);
            } catch (const std::exception &e) {
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