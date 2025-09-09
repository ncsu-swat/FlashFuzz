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
        
        uint8_t config_byte = Data[offset++];
        
        if (input_tensor.numel() == 0) {
            torch::gradient(input_tensor);
            return 0;
        }
        
        if (input_tensor.dim() == 0) {
            torch::gradient(input_tensor);
            return 0;
        }
        
        bool use_spacing = (config_byte & 0x01) != 0;
        bool use_edge_order = (config_byte & 0x02) != 0;
        bool use_dim = (config_byte & 0x04) != 0;
        
        if (!use_spacing && !use_edge_order && !use_dim) {
            torch::gradient(input_tensor);
            return 0;
        }
        
        c10::optional<torch::Scalar> spacing;
        c10::optional<int64_t> edge_order;
        c10::optional<int64_t> dim;
        
        if (use_spacing && offset < Size) {
            float spacing_val;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&spacing_val, Data + offset, sizeof(float));
                offset += sizeof(float);
            } else {
                spacing_val = 1.0f;
            }
            spacing = torch::Scalar(spacing_val);
        }
        
        if (use_edge_order && offset < Size) {
            int8_t edge_order_raw = static_cast<int8_t>(Data[offset++]);
            edge_order = static_cast<int64_t>(edge_order_raw % 3 + 1);
        }
        
        if (use_dim && offset < Size) {
            int8_t dim_raw = static_cast<int8_t>(Data[offset++]);
            int64_t tensor_dims = input_tensor.dim();
            if (tensor_dims > 0) {
                dim = static_cast<int64_t>(std::abs(dim_raw) % tensor_dims);
            }
        }
        
        if (use_spacing && use_edge_order && use_dim) {
            torch::gradient(input_tensor, spacing, dim, edge_order);
        } else if (use_spacing && use_edge_order) {
            torch::gradient(input_tensor, spacing, c10::nullopt, edge_order);
        } else if (use_spacing && use_dim) {
            torch::gradient(input_tensor, spacing, dim);
        } else if (use_edge_order && use_dim) {
            torch::gradient(input_tensor, c10::nullopt, dim, edge_order);
        } else if (use_spacing) {
            torch::gradient(input_tensor, spacing);
        } else if (use_edge_order) {
            torch::gradient(input_tensor, c10::nullopt, c10::nullopt, edge_order);
        } else if (use_dim) {
            torch::gradient(input_tensor, c10::nullopt, dim);
        }
        
        if (offset < Size && input_tensor.dim() > 1) {
            uint8_t multi_dim_byte = Data[offset++];
            if ((multi_dim_byte & 0x01) != 0) {
                std::vector<torch::Scalar> spacings;
                for (int64_t i = 0; i < input_tensor.dim() && offset < Size; ++i) {
                    float spacing_val = 1.0f;
                    if (offset + sizeof(float) <= Size) {
                        std::memcpy(&spacing_val, Data + offset, sizeof(float));
                        offset += sizeof(float);
                    }
                    spacings.push_back(torch::Scalar(spacing_val));
                }
                if (!spacings.empty()) {
                    torch::gradient(input_tensor, spacings);
                }
            }
        }
        
        if (offset < Size) {
            uint8_t negative_test = Data[offset++];
            if ((negative_test & 0x01) != 0 && input_tensor.dim() > 0) {
                int64_t invalid_dim = input_tensor.dim() + (negative_test % 10);
                torch::gradient(input_tensor, c10::nullopt, invalid_dim);
            }
            if ((negative_test & 0x02) != 0 && input_tensor.dim() > 0) {
                int64_t negative_dim = -(negative_test % 10 + 1);
                torch::gradient(input_tensor, c10::nullopt, negative_dim);
            }
            if ((negative_test & 0x04) != 0) {
                int64_t invalid_edge_order = (negative_test % 5) + 3;
                torch::gradient(input_tensor, c10::nullopt, c10::nullopt, invalid_edge_order);
            }
            if ((negative_test & 0x08) != 0) {
                int64_t negative_edge_order = -(negative_test % 3 + 1);
                torch::gradient(input_tensor, c10::nullopt, c10::nullopt, negative_edge_order);
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