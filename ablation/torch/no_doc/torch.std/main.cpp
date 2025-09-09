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

        uint8_t operation_selector = Data[offset++];
        uint8_t operation_type = operation_selector % 8;

        switch (operation_type) {
            case 0: {
                auto result = torch::std(input_tensor);
                break;
            }
            case 1: {
                if (offset >= Size) break;
                uint8_t unbiased_selector = Data[offset++];
                bool unbiased = (unbiased_selector % 2) == 1;
                auto result = torch::std(input_tensor, unbiased);
                break;
            }
            case 2: {
                if (input_tensor.dim() == 0) break;
                if (offset >= Size) break;
                uint8_t dim_selector = Data[offset++];
                int64_t dim = static_cast<int64_t>(dim_selector) % input_tensor.dim();
                if (dim_selector & 0x80) {
                    dim = -dim - 1;
                }
                auto result = torch::std(input_tensor, dim);
                break;
            }
            case 3: {
                if (input_tensor.dim() == 0) break;
                if (offset >= Size) break;
                uint8_t dim_selector = Data[offset++];
                int64_t dim = static_cast<int64_t>(dim_selector) % input_tensor.dim();
                if (dim_selector & 0x80) {
                    dim = -dim - 1;
                }
                if (offset >= Size) break;
                uint8_t unbiased_selector = Data[offset++];
                bool unbiased = (unbiased_selector % 2) == 1;
                auto result = torch::std(input_tensor, dim, unbiased);
                break;
            }
            case 4: {
                if (input_tensor.dim() == 0) break;
                if (offset >= Size) break;
                uint8_t dim_selector = Data[offset++];
                int64_t dim = static_cast<int64_t>(dim_selector) % input_tensor.dim();
                if (dim_selector & 0x80) {
                    dim = -dim - 1;
                }
                if (offset >= Size) break;
                uint8_t keepdim_selector = Data[offset++];
                bool keepdim = (keepdim_selector % 2) == 1;
                auto result = torch::std(input_tensor, dim, keepdim);
                break;
            }
            case 5: {
                if (input_tensor.dim() == 0) break;
                if (offset >= Size) break;
                uint8_t dim_selector = Data[offset++];
                int64_t dim = static_cast<int64_t>(dim_selector) % input_tensor.dim();
                if (dim_selector & 0x80) {
                    dim = -dim - 1;
                }
                if (offset + 1 >= Size) break;
                uint8_t unbiased_selector = Data[offset++];
                bool unbiased = (unbiased_selector % 2) == 1;
                uint8_t keepdim_selector = Data[offset++];
                bool keepdim = (keepdim_selector % 2) == 1;
                auto result = torch::std(input_tensor, dim, unbiased, keepdim);
                break;
            }
            case 6: {
                if (input_tensor.dim() == 0) break;
                if (offset >= Size) break;
                uint8_t num_dims_selector = Data[offset++];
                uint8_t num_dims = (num_dims_selector % input_tensor.dim()) + 1;
                
                std::vector<int64_t> dims;
                for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                    uint8_t dim_selector = Data[offset++];
                    int64_t dim = static_cast<int64_t>(dim_selector) % input_tensor.dim();
                    if (dim_selector & 0x80) {
                        dim = -dim - 1;
                    }
                    dims.push_back(dim);
                }
                
                if (!dims.empty()) {
                    auto result = torch::std(input_tensor, dims);
                }
                break;
            }
            case 7: {
                if (input_tensor.dim() == 0) break;
                if (offset >= Size) break;
                uint8_t num_dims_selector = Data[offset++];
                uint8_t num_dims = (num_dims_selector % input_tensor.dim()) + 1;
                
                std::vector<int64_t> dims;
                for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                    uint8_t dim_selector = Data[offset++];
                    int64_t dim = static_cast<int64_t>(dim_selector) % input_tensor.dim();
                    if (dim_selector & 0x80) {
                        dim = -dim - 1;
                    }
                    dims.push_back(dim);
                }
                
                if (!dims.empty() && offset + 1 < Size) {
                    uint8_t unbiased_selector = Data[offset++];
                    bool unbiased = (unbiased_selector % 2) == 1;
                    uint8_t keepdim_selector = Data[offset++];
                    bool keepdim = (keepdim_selector % 2) == 1;
                    auto result = torch::std(input_tensor, dims, unbiased, keepdim);
                }
                break;
            }
        }

        if (offset < Size) {
            uint8_t edge_case_selector = Data[offset++];
            uint8_t edge_case = edge_case_selector % 6;
            
            switch (edge_case) {
                case 0: {
                    auto empty_tensor = torch::empty({0});
                    auto result = torch::std(empty_tensor);
                    break;
                }
                case 1: {
                    auto single_element = torch::tensor({42.0});
                    auto result = torch::std(single_element);
                    break;
                }
                case 2: {
                    auto inf_tensor = torch::full({3, 3}, std::numeric_limits<float>::infinity());
                    auto result = torch::std(inf_tensor);
                    break;
                }
                case 3: {
                    auto nan_tensor = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN());
                    auto result = torch::std(nan_tensor);
                    break;
                }
                case 4: {
                    if (input_tensor.dim() > 0) {
                        int64_t invalid_dim = input_tensor.dim() + 10;
                        auto result = torch::std(input_tensor, invalid_dim);
                    }
                    break;
                }
                case 5: {
                    if (input_tensor.dim() > 0) {
                        int64_t negative_invalid_dim = -(input_tensor.dim() + 10);
                        auto result = torch::std(input_tensor, negative_invalid_dim);
                    }
                    break;
                }
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