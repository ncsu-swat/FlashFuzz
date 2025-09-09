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

        auto input = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (input.numel() == 0) {
            return 0;
        }

        if (offset >= Size) {
            torch::gradient(input);
            return 0;
        }

        uint8_t config_byte = Data[offset++];
        bool use_spacing = (config_byte & 0x01) != 0;
        bool use_dim = (config_byte & 0x02) != 0;
        bool use_edge_order = (config_byte & 0x04) != 0;
        uint8_t spacing_type = (config_byte >> 3) & 0x03;

        c10::optional<c10::Scalar> spacing_scalar;
        c10::optional<std::vector<c10::Scalar>> spacing_list;
        c10::optional<std::vector<torch::Tensor>> spacing_tensors;
        
        if (use_spacing && offset < Size) {
            if (spacing_type == 0) {
                if (offset + sizeof(float) <= Size) {
                    float spacing_val;
                    std::memcpy(&spacing_val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    spacing_scalar = c10::Scalar(spacing_val);
                }
            } else if (spacing_type == 1) {
                uint8_t num_spacings = std::min(static_cast<uint8_t>(input.dim()), static_cast<uint8_t>(4));
                if (num_spacings > 0 && offset + num_spacings * sizeof(float) <= Size) {
                    std::vector<c10::Scalar> spacings;
                    for (uint8_t i = 0; i < num_spacings; ++i) {
                        float spacing_val;
                        std::memcpy(&spacing_val, Data + offset, sizeof(float));
                        offset += sizeof(float);
                        spacings.push_back(c10::Scalar(spacing_val));
                    }
                    spacing_list = spacings;
                }
            } else if (spacing_type == 2) {
                uint8_t num_tensors = std::min(static_cast<uint8_t>(input.dim()), static_cast<uint8_t>(4));
                std::vector<torch::Tensor> tensors;
                for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
                    try {
                        int64_t tensor_size = input.size(i);
                        if (tensor_size > 0 && tensor_size <= 1000) {
                            auto spacing_tensor = torch::arange(tensor_size, torch::kFloat);
                            if (offset + sizeof(float) <= Size) {
                                float scale;
                                std::memcpy(&scale, Data + offset, sizeof(float));
                                offset += sizeof(float);
                                spacing_tensor = spacing_tensor * scale;
                            }
                            tensors.push_back(spacing_tensor);
                        }
                    } catch (...) {
                        break;
                    }
                }
                if (!tensors.empty()) {
                    spacing_tensors = tensors;
                }
            }
        }

        c10::optional<c10::IntArrayRef> dim_opt;
        std::vector<int64_t> dims;
        if (use_dim && offset < Size) {
            uint8_t num_dims = Data[offset++] % 5;
            if (num_dims > 0 && input.dim() > 0) {
                for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                    int8_t dim_val = static_cast<int8_t>(Data[offset++]);
                    int64_t actual_dim = dim_val % input.dim();
                    if (actual_dim < 0) {
                        actual_dim += input.dim();
                    }
                    dims.push_back(actual_dim);
                }
                if (!dims.empty()) {
                    dim_opt = c10::IntArrayRef(dims);
                }
            }
        }

        int64_t edge_order = 1;
        if (use_edge_order && offset < Size) {
            edge_order = (Data[offset++] % 2) + 1;
        }

        torch::TensorOptions options;
        if (spacing_scalar.has_value()) {
            options = options.dtype(torch::kFloat);
        }
        if (spacing_list.has_value()) {
            options = options.dtype(torch::kFloat);
        }

        std::vector<torch::Tensor> result;
        
        if (spacing_scalar.has_value()) {
            if (dim_opt.has_value()) {
                result = torch::gradient(input, spacing_scalar.value(), dim_opt.value(), edge_order);
            } else {
                result = torch::gradient(input, spacing_scalar.value(), c10::nullopt, edge_order);
            }
        } else if (spacing_list.has_value()) {
            if (dim_opt.has_value()) {
                result = torch::gradient(input, spacing_list.value(), dim_opt.value(), edge_order);
            } else {
                result = torch::gradient(input, spacing_list.value(), c10::nullopt, edge_order);
            }
        } else if (spacing_tensors.has_value()) {
            if (dim_opt.has_value()) {
                result = torch::gradient(input, spacing_tensors.value(), dim_opt.value(), edge_order);
            } else {
                result = torch::gradient(input, spacing_tensors.value(), c10::nullopt, edge_order);
            }
        } else {
            if (dim_opt.has_value()) {
                result = torch::gradient(input, c10::nullopt, dim_opt.value(), edge_order);
            } else {
                result = torch::gradient(input, c10::nullopt, c10::nullopt, edge_order);
            }
        }

        for (const auto& grad_tensor : result) {
            if (grad_tensor.numel() > 0) {
                auto sum = torch::sum(grad_tensor);
                auto mean = torch::mean(grad_tensor);
            }
        }

        if (offset < Size) {
            auto input2 = input.clone();
            if (Data[offset] % 2 == 0) {
                input2 = input2.to(torch::kDouble);
            }
            torch::gradient(input2);
        }

        auto zero_tensor = torch::zeros({1}, input.options());
        torch::gradient(zero_tensor);

        auto large_tensor = torch::ones({100}, input.options());
        torch::gradient(large_tensor);

        if (input.dim() >= 2) {
            auto slice = input.select(0, 0);
            torch::gradient(slice);
        }

        if (input.is_floating_point()) {
            auto nan_tensor = input.clone();
            if (nan_tensor.numel() > 0) {
                nan_tensor.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
                torch::gradient(nan_tensor);
            }
            
            auto inf_tensor = input.clone();
            if (inf_tensor.numel() > 0) {
                inf_tensor.flatten()[0] = std::numeric_limits<float>::infinity();
                torch::gradient(inf_tensor);
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