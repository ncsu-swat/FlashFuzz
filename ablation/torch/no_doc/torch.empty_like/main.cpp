#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        if (Size < 2) {
            return 0;
        }

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        if (offset < Size) {
            uint8_t options_byte = Data[offset++];
            
            if (options_byte & 0x01) {
                auto result = torch::empty_like(input_tensor);
                if (result.sizes() != input_tensor.sizes()) {
                    throw std::runtime_error("Shape mismatch in empty_like result");
                }
                if (result.dtype() != input_tensor.dtype()) {
                    throw std::runtime_error("Dtype mismatch in empty_like result");
                }
            }

            if (options_byte & 0x02 && offset < Size) {
                uint8_t dtype_selector = Data[offset++];
                auto new_dtype = fuzzer_utils::parseDataType(dtype_selector);
                auto options = torch::TensorOptions().dtype(new_dtype);
                auto result = torch::empty_like(input_tensor, options);
                if (result.sizes() != input_tensor.sizes()) {
                    throw std::runtime_error("Shape mismatch in empty_like with dtype result");
                }
                if (result.dtype() != new_dtype) {
                    throw std::runtime_error("Dtype override failed in empty_like");
                }
            }

            if (options_byte & 0x04 && offset < Size) {
                uint8_t layout_byte = Data[offset++];
                torch::Layout layout = (layout_byte & 0x01) ? torch::kSparse : torch::kStrided;
                try {
                    auto options = torch::TensorOptions().layout(layout);
                    auto result = torch::empty_like(input_tensor, options);
                } catch (const std::exception&) {
                }
            }

            if (options_byte & 0x08 && offset < Size) {
                uint8_t device_byte = Data[offset++];
                torch::Device device = (device_byte & 0x01) ? torch::kCUDA : torch::kCPU;
                try {
                    auto options = torch::TensorOptions().device(device);
                    auto result = torch::empty_like(input_tensor, options);
                } catch (const std::exception&) {
                }
            }

            if (options_byte & 0x10 && offset < Size) {
                uint8_t requires_grad_byte = Data[offset++];
                bool requires_grad = requires_grad_byte & 0x01;
                auto options = torch::TensorOptions().requires_grad(requires_grad);
                auto result = torch::empty_like(input_tensor, options);
                if (result.requires_grad() != requires_grad) {
                    throw std::runtime_error("requires_grad setting failed in empty_like");
                }
            }

            if (options_byte & 0x20 && offset < Size) {
                uint8_t pinned_byte = Data[offset++];
                bool pinned_memory = pinned_byte & 0x01;
                try {
                    auto options = torch::TensorOptions().pinned_memory(pinned_memory);
                    auto result = torch::empty_like(input_tensor, options);
                } catch (const std::exception&) {
                }
            }

            if (options_byte & 0x40 && offset + 1 < Size) {
                uint8_t dtype_selector = Data[offset++];
                uint8_t device_byte = Data[offset++];
                auto new_dtype = fuzzer_utils::parseDataType(dtype_selector);
                torch::Device device = (device_byte & 0x01) ? torch::kCUDA : torch::kCPU;
                try {
                    auto options = torch::TensorOptions().dtype(new_dtype).device(device);
                    auto result = torch::empty_like(input_tensor, options);
                } catch (const std::exception&) {
                }
            }

            if (options_byte & 0x80 && offset + 2 < Size) {
                uint8_t dtype_selector = Data[offset++];
                uint8_t requires_grad_byte = Data[offset++];
                uint8_t layout_byte = Data[offset++];
                auto new_dtype = fuzzer_utils::parseDataType(dtype_selector);
                bool requires_grad = requires_grad_byte & 0x01;
                torch::Layout layout = (layout_byte & 0x01) ? torch::kSparse : torch::kStrided;
                try {
                    auto options = torch::TensorOptions()
                        .dtype(new_dtype)
                        .requires_grad(requires_grad)
                        .layout(layout);
                    auto result = torch::empty_like(input_tensor, options);
                } catch (const std::exception&) {
                }
            }
        } else {
            auto result = torch::empty_like(input_tensor);
        }

        if (input_tensor.numel() == 0) {
            auto result = torch::empty_like(input_tensor);
            if (result.numel() != 0) {
                throw std::runtime_error("Empty tensor should produce empty result");
            }
        }

        if (input_tensor.dim() == 0) {
            auto result = torch::empty_like(input_tensor);
            if (result.dim() != 0) {
                throw std::runtime_error("Scalar tensor should produce scalar result");
            }
        }

        auto detached_input = input_tensor.detach();
        auto result_detached = torch::empty_like(detached_input);

        if (input_tensor.is_contiguous()) {
            auto result = torch::empty_like(input_tensor);
        }

        if (!input_tensor.is_contiguous()) {
            auto non_contiguous = input_tensor.transpose(0, -1);
            auto result = torch::empty_like(non_contiguous);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}