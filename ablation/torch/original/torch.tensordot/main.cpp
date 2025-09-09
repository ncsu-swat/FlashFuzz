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

        auto tensor_a = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }
        
        auto tensor_b = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }

        uint8_t dims_type = Data[offset++] % 3;
        
        if (dims_type == 0) {
            if (offset >= Size) {
                return 0;
            }
            int64_t dims_int = static_cast<int64_t>(static_cast<int8_t>(Data[offset++]));
            torch::tensordot(tensor_a, tensor_b, dims_int);
        }
        else if (dims_type == 1) {
            if (offset + 2 >= Size) {
                return 0;
            }
            
            uint8_t num_dims_a = Data[offset++] % 5;
            uint8_t num_dims_b = Data[offset++] % 5;
            
            std::vector<int64_t> dims_a;
            std::vector<int64_t> dims_b;
            
            for (uint8_t i = 0; i < num_dims_a && offset < Size; ++i) {
                int64_t dim = static_cast<int64_t>(static_cast<int8_t>(Data[offset++]));
                dims_a.push_back(dim);
            }
            
            for (uint8_t i = 0; i < num_dims_b && offset < Size; ++i) {
                int64_t dim = static_cast<int64_t>(static_cast<int8_t>(Data[offset++]));
                dims_b.push_back(dim);
            }
            
            std::vector<std::vector<int64_t>> dims_list = {dims_a, dims_b};
            torch::tensordot(tensor_a, tensor_b, dims_list);
        }
        else {
            if (offset + 1 >= Size) {
                return 0;
            }
            
            uint8_t tensor_dims_size = Data[offset++] % 8;
            
            if (tensor_dims_size == 0) {
                auto dims_tensor = torch::empty({0}, torch::kLong);
                torch::tensordot(tensor_a, tensor_b, dims_tensor);
            } else {
                std::vector<int64_t> dims_data;
                for (uint8_t i = 0; i < tensor_dims_size && offset < Size; ++i) {
                    int64_t dim = static_cast<int64_t>(static_cast<int8_t>(Data[offset++]));
                    dims_data.push_back(dim);
                }
                
                auto dims_tensor = torch::tensor(dims_data, torch::kLong);
                torch::tensordot(tensor_a, tensor_b, dims_tensor);
            }
        }

        if (offset < Size) {
            auto tensor_c = fuzzer_utils::createTensor(Data, Size, offset);
            if (offset < Size) {
                int64_t negative_dims = -static_cast<int64_t>(Data[offset++] % 10 + 1);
                torch::tensordot(tensor_a, tensor_c, negative_dims);
            }
        }

        if (offset < Size) {
            auto empty_a = torch::empty({0}, tensor_a.dtype());
            auto empty_b = torch::empty({0}, tensor_b.dtype());
            torch::tensordot(empty_a, empty_b, 0);
        }

        if (offset < Size) {
            auto large_tensor_a = torch::ones({1000, 1000}, tensor_a.dtype());
            auto large_tensor_b = torch::ones({1000, 1000}, tensor_b.dtype());
            torch::tensordot(large_tensor_a, large_tensor_b, 1);
        }

        if (offset < Size) {
            std::vector<std::vector<int64_t>> invalid_dims = {{-1, -2}, {-3, -4}};
            torch::tensordot(tensor_a, tensor_b, invalid_dims);
        }

        if (offset < Size) {
            std::vector<std::vector<int64_t>> mismatched_dims = {{0}, {0, 1}};
            torch::tensordot(tensor_a, tensor_b, mismatched_dims);
        }

        if (offset < Size) {
            auto scalar_a = torch::scalar_tensor(1.0, tensor_a.dtype());
            auto scalar_b = torch::scalar_tensor(2.0, tensor_b.dtype());
            torch::tensordot(scalar_a, scalar_b, 0);
        }

        if (offset < Size) {
            std::vector<std::vector<int64_t>> out_of_bounds = {{100}, {200}};
            torch::tensordot(tensor_a, tensor_b, out_of_bounds);
        }

        if (offset < Size) {
            int64_t large_dims = 1000000;
            torch::tensordot(tensor_a, tensor_b, large_dims);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}