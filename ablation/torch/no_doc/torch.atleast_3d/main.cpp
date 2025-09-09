#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 1) {
            return 0;
        }
        
        uint8_t num_tensors_byte = Data[offset++];
        uint8_t num_tensors = (num_tensors_byte % 5) + 1;
        
        std::vector<torch::Tensor> tensors;
        
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (const std::exception&) {
                break;
            }
        }
        
        if (tensors.empty()) {
            return 0;
        }
        
        for (const auto& tensor : tensors) {
            torch::atleast_3d(tensor);
        }
        
        if (tensors.size() > 1) {
            torch::atleast_3d(tensors);
        }
        
        torch::Tensor scalar_tensor = torch::tensor(42.0);
        torch::atleast_3d(scalar_tensor);
        
        torch::Tensor empty_tensor = torch::empty({0});
        torch::atleast_3d(empty_tensor);
        
        torch::Tensor one_d = torch::ones({5});
        torch::atleast_3d(one_d);
        
        torch::Tensor two_d = torch::ones({3, 4});
        torch::atleast_3d(two_d);
        
        torch::Tensor three_d = torch::ones({2, 3, 4});
        torch::atleast_3d(three_d);
        
        torch::Tensor four_d = torch::ones({2, 3, 4, 5});
        torch::atleast_3d(four_d);
        
        torch::Tensor zero_dim_tensor = torch::ones({1, 0, 1});
        torch::atleast_3d(zero_dim_tensor);
        
        torch::Tensor large_tensor = torch::ones({1, 1, 100});
        torch::atleast_3d(large_tensor);
        
        std::vector<torch::Tensor> mixed_tensors = {
            torch::tensor(1.0),
            torch::ones({5}),
            torch::ones({3, 4}),
            torch::ones({2, 3, 4})
        };
        torch::atleast_3d(mixed_tensors);
        
        std::vector<torch::Tensor> empty_vector;
        torch::atleast_3d(empty_vector);
        
        torch::Tensor bool_tensor = torch::ones({2, 3}, torch::kBool);
        torch::atleast_3d(bool_tensor);
        
        torch::Tensor int_tensor = torch::ones({4}, torch::kInt32);
        torch::atleast_3d(int_tensor);
        
        torch::Tensor complex_tensor = torch::ones({2, 2}, torch::kComplexFloat);
        torch::atleast_3d(complex_tensor);
        
        torch::Tensor half_tensor = torch::ones({3, 3}, torch::kHalf);
        torch::atleast_3d(half_tensor);
        
        torch::Tensor double_tensor = torch::ones({2}, torch::kDouble);
        torch::atleast_3d(double_tensor);
        
        torch::Tensor int64_tensor = torch::ones({1, 1}, torch::kInt64);
        torch::atleast_3d(int64_tensor);
        
        torch::Tensor uint8_tensor = torch::ones({5, 1}, torch::kUInt8);
        torch::atleast_3d(uint8_tensor);
        
        torch::Tensor int8_tensor = torch::ones({1, 5}, torch::kInt8);
        torch::atleast_3d(int8_tensor);
        
        torch::Tensor bfloat16_tensor = torch::ones({2, 2}, torch::kBFloat16);
        torch::atleast_3d(bfloat16_tensor);
        
        torch::Tensor complex_double_tensor = torch::ones({1}, torch::kComplexDouble);
        torch::atleast_3d(complex_double_tensor);
        
        std::vector<torch::Tensor> different_dtypes = {
            torch::tensor(1.0f),
            torch::tensor(2, torch::kInt32),
            torch::tensor(true, torch::kBool)
        };
        torch::atleast_3d(different_dtypes);
        
        torch::Tensor very_large_1d = torch::ones({1000});
        torch::atleast_3d(very_large_1d);
        
        torch::Tensor very_large_2d = torch::ones({100, 100});
        torch::atleast_3d(very_large_2d);
        
        torch::Tensor single_element = torch::ones({1});
        torch::atleast_3d(single_element);
        
        torch::Tensor single_element_2d = torch::ones({1, 1});
        torch::atleast_3d(single_element_2d);
        
        torch::Tensor asymmetric_2d = torch::ones({1, 10});
        torch::atleast_3d(asymmetric_2d);
        
        torch::Tensor asymmetric_2d_alt = torch::ones({10, 1});
        torch::atleast_3d(asymmetric_2d_alt);
        
        std::vector<torch::Tensor> many_tensors;
        for (int i = 0; i < 50; ++i) {
            many_tensors.push_back(torch::ones({i % 3 + 1}));
        }
        torch::atleast_3d(many_tensors);
        
        torch::Tensor zero_size_dim = torch::empty({0, 5});
        torch::atleast_3d(zero_size_dim);
        
        torch::Tensor zero_size_dim_alt = torch::empty({5, 0});
        torch::atleast_3d(zero_size_dim_alt);
        
        torch::Tensor multiple_zero_dims = torch::empty({0, 0, 5});
        torch::atleast_3d(multiple_zero_dims);
        
        std::vector<torch::Tensor> zero_dim_vector = {
            torch::empty({0}),
            torch::empty({0, 1}),
            torch::empty({1, 0})
        };
        torch::atleast_3d(zero_dim_vector);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}