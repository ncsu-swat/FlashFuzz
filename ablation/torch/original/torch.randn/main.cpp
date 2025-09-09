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
        uint8_t variant = operation_selector % 8;

        switch (variant) {
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
                torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
                
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
                torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                if (offset >= Size) break;
                uint8_t requires_grad_byte = Data[offset++];
                bool requires_grad = (requires_grad_byte % 2) == 1;
                
                if (offset >= Size) break;
                uint8_t rank_byte = Data[offset++];
                uint8_t rank = fuzzer_utils::parseRank(rank_byte);
                
                auto shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
                auto options = torch::TensorOptions().dtype(dtype).requires_grad(requires_grad);
                torch::randn(shape, options);
                break;
            }
            case 3: {
                if (offset >= Size) break;
                uint8_t rank_byte = Data[offset++];
                uint8_t rank = fuzzer_utils::parseRank(rank_byte);
                
                auto shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
                
                torch::Tensor out_tensor = torch::empty(shape);
                torch::randn_out(out_tensor, shape);
                break;
            }
            case 4: {
                if (offset + 8 > Size) break;
                
                int64_t dim1, dim2;
                std::memcpy(&dim1, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                std::memcpy(&dim2, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                dim1 = std::abs(dim1) % 100 + 1;
                dim2 = std::abs(dim2) % 100 + 1;
                
                torch::randn({dim1, dim2});
                break;
            }
            case 5: {
                if (offset + 4 > Size) break;
                
                int32_t single_dim;
                std::memcpy(&single_dim, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                
                single_dim = std::abs(single_dim) % 1000;
                
                torch::randn(single_dim);
                break;
            }
            case 6: {
                if (offset >= Size) break;
                uint8_t dtype_selector = Data[offset++];
                torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                if (offset >= Size) break;
                uint8_t pin_memory_byte = Data[offset++];
                bool pin_memory = (pin_memory_byte % 2) == 1;
                
                if (offset >= Size) break;
                uint8_t rank_byte = Data[offset++];
                uint8_t rank = fuzzer_utils::parseRank(rank_byte);
                
                auto shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
                auto options = torch::TensorOptions().dtype(dtype).pinned_memory(pin_memory);
                torch::randn(shape, options);
                break;
            }
            case 7: {
                torch::randn({});
                
                if (offset + 12 > Size) break;
                
                int32_t d1, d2, d3;
                std::memcpy(&d1, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                std::memcpy(&d2, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                std::memcpy(&d3, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                
                d1 = std::abs(d1) % 50 + 1;
                d2 = std::abs(d2) % 50 + 1;
                d3 = std::abs(d3) % 50 + 1;
                
                torch::randn({d1, d2, d3});
                break;
            }
        }

        if (offset < Size) {
            uint8_t extra_selector = Data[offset++];
            uint8_t extra_variant = extra_selector % 4;
            
            switch (extra_variant) {
                case 0: {
                    std::vector<int64_t> large_shape;
                    for (int i = 0; i < 6 && offset < Size; i++) {
                        if (offset >= Size) break;
                        uint8_t dim_byte = Data[offset++];
                        large_shape.push_back(dim_byte % 10 + 1);
                    }
                    if (!large_shape.empty()) {
                        torch::randn(large_shape);
                    }
                    break;
                }
                case 1: {
                    if (offset >= Size) break;
                    uint8_t dtype_selector = Data[offset++];
                    torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
                    
                    auto options = torch::TensorOptions().dtype(dtype);
                    torch::randn({0}, options);
                    torch::randn({1, 0}, options);
                    torch::randn({0, 1}, options);
                    break;
                }
                case 2: {
                    if (offset + 4 > Size) break;
                    
                    uint32_t seed;
                    std::memcpy(&seed, Data + offset, sizeof(uint32_t));
                    offset += sizeof(uint32_t);
                    
                    auto gen = torch::make_generator<torch::CPUGeneratorImpl>(seed);
                    auto options = torch::TensorOptions().generator(gen);
                    torch::randn({5, 5}, options);
                    break;
                }
                case 3: {
                    if (offset >= Size) break;
                    uint8_t layout_selector = Data[offset++];
                    
                    if (layout_selector % 2 == 0) {
                        auto options = torch::TensorOptions().layout(torch::kStrided);
                        torch::randn({3, 3}, options);
                    } else {
                        torch::randn({10, 10});
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