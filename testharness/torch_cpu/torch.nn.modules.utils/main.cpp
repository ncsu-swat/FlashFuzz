#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/nn/modules/utils.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a second tensor if there's enough data left
        torch::Tensor tensor2;
        if (offset + 2 < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Get a value for padding if there's data left
        std::vector<int64_t> padding;
        if (offset < Size) {
            uint8_t num_dims = Data[offset++] % 4 + 1;
            for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                padding.push_back(static_cast<int64_t>(Data[offset++]));
            }
        } else {
            padding = {1, 2};
        }
        
        // Get a value for stride if there's data left
        std::vector<int64_t> stride;
        if (offset < Size) {
            uint8_t num_dims = Data[offset++] % 4 + 1;
            for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                stride.push_back(static_cast<int64_t>(Data[offset++]) % 8 + 1);
            }
        } else {
            stride = {1, 1};
        }
        
        // Get a value for kernel_size if there's data left
        std::vector<int64_t> kernel_size;
        if (offset < Size) {
            uint8_t num_dims = Data[offset++] % 4 + 1;
            for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                kernel_size.push_back(static_cast<int64_t>(Data[offset++]) % 8 + 1);
            }
        } else {
            kernel_size = {3, 3};
        }
        
        // Get a value for dilation if there's data left
        std::vector<int64_t> dilation;
        if (offset < Size) {
            uint8_t num_dims = Data[offset++] % 4 + 1;
            for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                dilation.push_back(static_cast<int64_t>(Data[offset++]) % 4 + 1);
            }
        } else {
            dilation = {1, 1};
        }
        
        // Get a value for output_padding if there's data left
        std::vector<int64_t> output_padding;
        if (offset < Size) {
            uint8_t num_dims = Data[offset++] % 4 + 1;
            for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                output_padding.push_back(static_cast<int64_t>(Data[offset++]) % 4);
            }
        } else {
            output_padding = {0, 0};
        }
        
        // Get a value for input_size if there's data left
        std::vector<int64_t> input_size;
        if (offset < Size) {
            uint8_t num_dims = Data[offset++] % 4 + 1;
            for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                input_size.push_back(static_cast<int64_t>(Data[offset++]) % 32 + 1);
            }
        } else {
            input_size = {28, 28};
        }
        
        // Get a value for output_size if there's data left
        std::vector<int64_t> output_size;
        if (offset < Size) {
            uint8_t num_dims = Data[offset++] % 4 + 1;
            for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                output_size.push_back(static_cast<int64_t>(Data[offset++]) % 32 + 1);
            }
        } else {
            output_size = {24, 24};
        }
        
        // Test available functions from torch::nn::modules::utils
        
        // Test _reverse_repeat_vector
        if (!kernel_size.empty()) {
            auto reverse_repeat_result = torch::nn::modules::utils::_reverse_repeat_vector(kernel_size, 2);
        }
        
        // Test _single
        auto single_result = torch::nn::modules::utils::_single(padding[0]);
        
        // Test _pair
        auto pair_result = torch::nn::modules::utils::_pair(padding.size() > 1 ? 
            std::vector<int64_t>{padding[0], padding[1]} : 
            std::vector<int64_t>{padding[0], padding[0]});
        
        // Test _triple
        auto triple_result = torch::nn::modules::utils::_triple(padding.size() > 2 ? 
            std::vector<int64_t>{padding[0], padding[1], padding[2]} : 
            std::vector<int64_t>{padding[0], padding[0], padding[0]});
        
        // Test _quadruple
        auto quadruple_result = torch::nn::modules::utils::_quadruple(padding.size() > 3 ? 
            std::vector<int64_t>{padding[0], padding[1], padding[2], padding[3]} : 
            std::vector<int64_t>{padding[0], padding[0], padding[0], padding[0]});
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
