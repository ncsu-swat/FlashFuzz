#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t operation_mode = Data[offset++];
        
        if (operation_mode % 2 == 0) {
            if (offset >= Size) {
                return 0;
            }
            
            auto other_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            torch::ne(input_tensor, other_tensor);
            
            torch::Tensor out_tensor;
            if (offset < Size && Data[offset] % 3 == 0) {
                offset++;
                try {
                    out_tensor = torch::empty_like(input_tensor, torch::TensorOptions().dtype(torch::kBool));
                    torch::ne_out(out_tensor, input_tensor, other_tensor);
                } catch (...) {
                }
            }
            
        } else {
            if (offset + sizeof(double) <= Size) {
                double scalar_value;
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                torch::ne(input_tensor, scalar_value);
                
                if (offset < Size && Data[offset] % 4 == 0) {
                    try {
                        auto out_tensor = torch::empty_like(input_tensor, torch::TensorOptions().dtype(torch::kBool));
                        torch::ne_out(out_tensor, input_tensor, scalar_value);
                    } catch (...) {
                    }
                }
            } else {
                torch::ne(input_tensor, 0.0);
            }
        }
        
        if (offset < Size) {
            uint8_t special_case = Data[offset++];
            
            if (special_case % 8 == 0) {
                auto zero_tensor = torch::zeros_like(input_tensor);
                torch::ne(input_tensor, zero_tensor);
            } else if (special_case % 8 == 1) {
                auto ones_tensor = torch::ones_like(input_tensor);
                torch::ne(input_tensor, ones_tensor);
            } else if (special_case % 8 == 2) {
                auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
                torch::ne(input_tensor, inf_tensor);
            } else if (special_case % 8 == 3) {
                auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
                torch::ne(input_tensor, nan_tensor);
            } else if (special_case % 8 == 4) {
                torch::ne(input_tensor, std::numeric_limits<double>::infinity());
            } else if (special_case % 8 == 5) {
                torch::ne(input_tensor, std::numeric_limits<double>::quiet_NaN());
            } else if (special_case % 8 == 6) {
                torch::ne(input_tensor, std::numeric_limits<double>::lowest());
            } else {
                torch::ne(input_tensor, std::numeric_limits<double>::max());
            }
        }
        
        if (offset < Size) {
            uint8_t broadcast_test = Data[offset++];
            
            if (broadcast_test % 4 == 0 && input_tensor.dim() > 0) {
                try {
                    auto shape = input_tensor.sizes().vec();
                    if (!shape.empty()) {
                        shape[0] = 1;
                        auto broadcast_tensor = torch::ones(shape, input_tensor.options());
                        torch::ne(input_tensor, broadcast_tensor);
                    }
                } catch (...) {
                }
            } else if (broadcast_test % 4 == 1 && input_tensor.dim() > 1) {
                try {
                    auto shape = input_tensor.sizes().vec();
                    shape.back() = 1;
                    auto broadcast_tensor = torch::zeros(shape, input_tensor.options());
                    torch::ne(input_tensor, broadcast_tensor);
                } catch (...) {
                }
            } else if (broadcast_test % 4 == 2) {
                try {
                    auto scalar_tensor = torch::tensor(42.0, input_tensor.options());
                    torch::ne(input_tensor, scalar_tensor);
                } catch (...) {
                }
            } else {
                try {
                    std::vector<int64_t> expanded_shape = {1};
                    for (auto dim : input_tensor.sizes()) {
                        expanded_shape.push_back(dim);
                    }
                    auto expanded_tensor = torch::randn(expanded_shape, input_tensor.options());
                    torch::ne(input_tensor, expanded_tensor);
                } catch (...) {
                }
            }
        }
        
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            if (edge_case % 6 == 0) {
                try {
                    auto empty_tensor = torch::empty({0}, input_tensor.options());
                    torch::ne(empty_tensor, empty_tensor);
                } catch (...) {
                }
            } else if (edge_case % 6 == 1) {
                try {
                    auto large_tensor = torch::ones({1000000}, input_tensor.options());
                    torch::ne(input_tensor.flatten(), large_tensor);
                } catch (...) {
                }
            } else if (edge_case % 6 == 2) {
                try {
                    auto reshaped = input_tensor.view(-1);
                    torch::ne(reshaped, reshaped);
                } catch (...) {
                }
            } else if (edge_case % 6 == 3) {
                try {
                    auto transposed = input_tensor.dim() >= 2 ? input_tensor.transpose(0, 1) : input_tensor;
                    torch::ne(input_tensor, transposed);
                } catch (...) {
                }
            } else if (edge_case % 6 == 4) {
                try {
                    auto sliced = input_tensor.dim() > 0 ? input_tensor.slice(0, 0, std::min(input_tensor.size(0), static_cast<int64_t>(2))) : input_tensor;
                    torch::ne(sliced, sliced);
                } catch (...) {
                }
            } else {
                try {
                    auto cloned = input_tensor.clone();
                    torch::ne(input_tensor, cloned);
                } catch (...) {
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