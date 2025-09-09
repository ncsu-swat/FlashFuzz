#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }
        
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t upscale_byte = Data[offset++];
        int64_t upscale_factor = static_cast<int64_t>(upscale_byte % 8) + 1;
        
        if (input.dim() < 3) {
            std::vector<int64_t> new_shape;
            int64_t target_dims = 4;
            
            for (int64_t i = 0; i < input.dim(); ++i) {
                new_shape.push_back(input.size(i));
            }
            
            while (new_shape.size() < target_dims) {
                new_shape.insert(new_shape.begin(), 1);
            }
            
            input = input.reshape(new_shape);
        }
        
        if (input.dim() >= 3) {
            int64_t channels_dim = input.dim() - 3;
            int64_t current_channels = input.size(channels_dim);
            int64_t required_channels = upscale_factor * upscale_factor;
            
            if (current_channels % required_channels != 0) {
                int64_t new_channels = ((current_channels / required_channels) + 1) * required_channels;
                
                std::vector<int64_t> new_shape;
                for (int64_t i = 0; i < input.dim(); ++i) {
                    if (i == channels_dim) {
                        new_shape.push_back(new_channels);
                    } else {
                        new_shape.push_back(input.size(i));
                    }
                }
                
                torch::Tensor padded = torch::zeros(new_shape, input.options());
                
                std::vector<torch::indexing::TensorIndex> indices;
                for (int64_t i = 0; i < input.dim(); ++i) {
                    if (i == channels_dim) {
                        indices.push_back(torch::indexing::Slice(0, current_channels));
                    } else {
                        indices.push_back(torch::indexing::Slice());
                    }
                }
                
                padded.index_put_(indices, input);
                input = padded;
            }
        }
        
        torch::Tensor result = torch::pixel_shuffle(input, upscale_factor);
        
        if (offset < Size) {
            uint8_t negative_factor_byte = Data[offset++];
            if (negative_factor_byte % 4 == 0) {
                int64_t negative_factor = -static_cast<int64_t>(upscale_byte % 10 + 1);
                torch::Tensor negative_result = torch::pixel_shuffle(input, negative_factor);
            }
        }
        
        if (offset < Size) {
            uint8_t zero_factor_byte = Data[offset++];
            if (zero_factor_byte % 8 == 0) {
                torch::Tensor zero_result = torch::pixel_shuffle(input, 0);
            }
        }
        
        if (offset < Size) {
            uint8_t large_factor_byte = Data[offset++];
            if (large_factor_byte % 16 == 0) {
                int64_t large_factor = static_cast<int64_t>(large_factor_byte) + 100;
                torch::Tensor large_result = torch::pixel_shuffle(input, large_factor);
            }
        }
        
        torch::Tensor empty_tensor = torch::empty({0, 0, 0, 0}, input.options());
        torch::Tensor empty_result = torch::pixel_shuffle(empty_tensor, upscale_factor);
        
        torch::Tensor single_element = torch::ones({1, upscale_factor * upscale_factor, 1, 1}, input.options());
        torch::Tensor single_result = torch::pixel_shuffle(single_element, upscale_factor);
        
        if (input.dim() >= 4) {
            std::vector<int64_t> large_shape = {1, upscale_factor * upscale_factor, 1000, 1000};
            torch::Tensor large_tensor = torch::zeros(large_shape, input.options());
            torch::Tensor large_result = torch::pixel_shuffle(large_tensor, upscale_factor);
        }
        
        torch::Tensor scalar_tensor = torch::scalar_tensor(1.0, input.options());
        torch::Tensor scalar_result = torch::pixel_shuffle(scalar_tensor, upscale_factor);
        
        torch::Tensor one_d = torch::ones({upscale_factor * upscale_factor}, input.options());
        torch::Tensor one_d_result = torch::pixel_shuffle(one_d, upscale_factor);
        
        torch::Tensor two_d = torch::ones({1, upscale_factor * upscale_factor}, input.options());
        torch::Tensor two_d_result = torch::pixel_shuffle(two_d, upscale_factor);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}