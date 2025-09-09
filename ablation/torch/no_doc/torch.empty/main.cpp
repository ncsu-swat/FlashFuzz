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

        uint8_t dtype_selector = Data[offset++];
        auto dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        uint8_t rank_byte = Data[offset++];
        uint8_t rank = fuzzer_utils::parseRank(rank_byte);
        
        auto shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        
        auto options = torch::TensorOptions().dtype(dtype);
        
        torch::Tensor result = torch::empty(shape, options);
        
        if (offset < Size) {
            uint8_t device_selector = Data[offset++];
            if (device_selector % 2 == 0) {
                auto cpu_options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
                torch::Tensor cpu_result = torch::empty(shape, cpu_options);
            }
        }
        
        if (offset < Size) {
            uint8_t layout_selector = Data[offset++];
            if (layout_selector % 2 == 0) {
                auto strided_options = torch::TensorOptions().dtype(dtype).layout(torch::kStrided);
                torch::Tensor strided_result = torch::empty(shape, strided_options);
            }
        }
        
        if (offset < Size) {
            uint8_t requires_grad_selector = Data[offset++];
            if (requires_grad_selector % 2 == 0) {
                auto grad_options = torch::TensorOptions().dtype(dtype).requires_grad(true);
                torch::Tensor grad_result = torch::empty(shape, grad_options);
            }
        }
        
        if (offset < Size) {
            uint8_t pinned_selector = Data[offset++];
            if (pinned_selector % 2 == 0) {
                auto pinned_options = torch::TensorOptions().dtype(dtype).pinned_memory(true);
                torch::Tensor pinned_result = torch::empty(shape, pinned_options);
            }
        }
        
        if (offset + 8 <= Size) {
            int64_t raw_size;
            std::memcpy(&raw_size, Data + offset, 8);
            offset += 8;
            
            int64_t single_size = std::abs(raw_size) % 1000;
            torch::Tensor single_dim_result = torch::empty({single_size}, options);
        }
        
        if (offset + 16 <= Size) {
            int64_t raw_size1, raw_size2;
            std::memcpy(&raw_size1, Data + offset, 8);
            std::memcpy(&raw_size2, Data + offset + 8, 8);
            offset += 16;
            
            int64_t size1 = std::abs(raw_size1) % 100;
            int64_t size2 = std::abs(raw_size2) % 100;
            torch::Tensor two_dim_result = torch::empty({size1, size2}, options);
        }
        
        torch::Tensor zero_dim_result = torch::empty({}, options);
        
        if (offset < Size) {
            uint8_t large_tensor_selector = Data[offset++];
            if (large_tensor_selector % 10 == 0) {
                torch::Tensor large_result = torch::empty({1000, 1000}, options);
            }
        }
        
        if (offset < Size) {
            uint8_t empty_tensor_selector = Data[offset++];
            if (empty_tensor_selector % 3 == 0) {
                torch::Tensor empty_result = torch::empty({0}, options);
            }
            if (empty_tensor_selector % 3 == 1) {
                torch::Tensor empty_2d_result = torch::empty({0, 5}, options);
            }
            if (empty_tensor_selector % 3 == 2) {
                torch::Tensor empty_3d_result = torch::empty({2, 0, 3}, options);
            }
        }
        
        std::vector<torch::ScalarType> all_types = {
            torch::kFloat, torch::kDouble, torch::kHalf, torch::kBFloat16,
            torch::kComplexFloat, torch::kComplexDouble,
            torch::kInt8, torch::kUInt8, torch::kInt16, torch::kInt32, torch::kInt64,
            torch::kBool
        };
        
        if (offset < Size) {
            uint8_t type_test_selector = Data[offset++];
            auto test_type = all_types[type_test_selector % all_types.size()];
            auto type_options = torch::TensorOptions().dtype(test_type);
            torch::Tensor type_result = torch::empty({10, 10}, type_options);
        }
        
        if (offset + 4 <= Size) {
            uint32_t combined_selector;
            std::memcpy(&combined_selector, Data + offset, 4);
            offset += 4;
            
            auto combined_dtype = all_types[combined_selector % all_types.size()];
            bool combined_requires_grad = (combined_selector >> 8) % 2 == 1;
            bool combined_pinned = (combined_selector >> 16) % 2 == 1;
            
            auto combined_options = torch::TensorOptions()
                .dtype(combined_dtype)
                .requires_grad(combined_requires_grad)
                .pinned_memory(combined_pinned);
            
            torch::Tensor combined_result = torch::empty({5, 5}, combined_options);
        }
        
        if (rank >= 1 && !shape.empty()) {
            std::vector<int64_t> modified_shape = shape;
            if (offset < Size) {
                uint8_t modifier = Data[offset++];
                if (modifier % 4 == 0 && modified_shape[0] > 0) {
                    modified_shape[0] = 0;
                }
                if (modifier % 4 == 1 && modified_shape.size() > 1) {
                    modified_shape[1] = modified_shape[1] * 2;
                }
                if (modifier % 4 == 2) {
                    modified_shape.push_back(1);
                }
                if (modifier % 4 == 3 && modified_shape.size() > 1) {
                    modified_shape.pop_back();
                }
            }
            torch::Tensor modified_result = torch::empty(modified_shape, options);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}