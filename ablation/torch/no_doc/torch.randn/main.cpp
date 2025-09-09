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
        
        uint8_t operation_selector = Data[offset++];
        uint8_t operation_type = operation_selector % 8;
        
        switch (operation_type) {
            case 0: {
                if (offset >= Size) break;
                uint8_t rank_byte = Data[offset++];
                uint8_t rank = fuzzer_utils::parseRank(rank_byte);
                auto shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
                torch::randn(shape);
                break;
            }
            case 1: {
                if (offset >= Size) break;
                uint8_t dtype_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                if (offset >= Size) break;
                uint8_t rank_byte = Data[offset++];
                uint8_t rank = fuzzer_utils::parseRank(rank_byte);
                auto shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
                auto options = torch::TensorOptions().dtype(dtype);
                torch::randn(shape, options);
                break;
            }
            case 2: {
                if (offset >= Size) break;
                uint8_t dtype_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                if (offset >= Size) break;
                uint8_t device_selector = Data[offset++];
                auto device = (device_selector % 2 == 0) ? torch::kCPU : torch::kCUDA;
                if (offset >= Size) break;
                uint8_t rank_byte = Data[offset++];
                uint8_t rank = fuzzer_utils::parseRank(rank_byte);
                auto shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
                auto options = torch::TensorOptions().dtype(dtype).device(device);
                torch::randn(shape, options);
                break;
            }
            case 3: {
                if (offset >= Size) break;
                uint8_t layout_selector = Data[offset++];
                auto layout = (layout_selector % 2 == 0) ? torch::kStrided : torch::kSparse;
                if (offset >= Size) break;
                uint8_t rank_byte = Data[offset++];
                uint8_t rank = fuzzer_utils::parseRank(rank_byte);
                auto shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
                auto options = torch::TensorOptions().layout(layout);
                torch::randn(shape, options);
                break;
            }
            case 4: {
                if (offset >= Size) break;
                uint8_t requires_grad_selector = Data[offset++];
                bool requires_grad = (requires_grad_selector % 2 == 1);
                if (offset >= Size) break;
                uint8_t rank_byte = Data[offset++];
                uint8_t rank = fuzzer_utils::parseRank(rank_byte);
                auto shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
                auto options = torch::TensorOptions().requires_grad(requires_grad);
                torch::randn(shape, options);
                break;
            }
            case 5: {
                if (offset + 8 > Size) break;
                int64_t single_dim;
                std::memcpy(&single_dim, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                uint64_t dim_abs = static_cast<uint64_t>(std::abs(single_dim));
                int64_t dim = static_cast<int64_t>(dim_abs % 1000);
                torch::randn({dim});
                break;
            }
            case 6: {
                if (offset + 16 > Size) break;
                int64_t dim1, dim2;
                std::memcpy(&dim1, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                std::memcpy(&dim2, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                uint64_t d1_abs = static_cast<uint64_t>(std::abs(dim1));
                uint64_t d2_abs = static_cast<uint64_t>(std::abs(dim2));
                int64_t d1 = static_cast<int64_t>(d1_abs % 100);
                int64_t d2 = static_cast<int64_t>(d2_abs % 100);
                torch::randn({d1, d2});
                break;
            }
            case 7: {
                if (offset >= Size) break;
                uint8_t dtype_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                if (offset >= Size) break;
                uint8_t device_selector = Data[offset++];
                auto device = (device_selector % 2 == 0) ? torch::kCPU : torch::kCUDA;
                if (offset >= Size) break;
                uint8_t requires_grad_selector = Data[offset++];
                bool requires_grad = (requires_grad_selector % 2 == 1);
                if (offset >= Size) break;
                uint8_t rank_byte = Data[offset++];
                uint8_t rank = fuzzer_utils::parseRank(rank_byte);
                auto shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
                auto options = torch::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad);
                torch::randn(shape, options);
                break;
            }
        }
        
        if (offset < Size) {
            std::vector<int64_t> extreme_shapes = {0, 1, -1, 1000000, -1000000};
            uint8_t shape_selector = Data[offset] % extreme_shapes.size();
            int64_t extreme_dim = extreme_shapes[shape_selector];
            torch::randn({extreme_dim});
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}