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
        
        uint8_t n_byte = Data[offset++];
        int64_t n = static_cast<int64_t>(n_byte) + 1;
        
        uint8_t d_byte = Data[offset++];
        int64_t d = static_cast<int64_t>(d_byte % 100) + 1;
        
        uint8_t dtype_selector = Data[offset++];
        auto dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        uint8_t device_selector = Data[offset++];
        torch::Device device = (device_selector % 2 == 0) ? torch::kCPU : torch::kCUDA;
        
        uint8_t layout_selector = Data[offset++];
        torch::Layout layout = (layout_selector % 2 == 0) ? torch::kStrided : torch::kSparse;
        
        uint8_t requires_grad_selector = Data[offset++];
        bool requires_grad = (requires_grad_selector % 2 == 1);
        
        uint8_t pin_memory_selector = Data[offset++];
        bool pin_memory = (pin_memory_selector % 2 == 1);
        
        uint8_t engine_selector = Data[offset++];
        int64_t engine = static_cast<int64_t>(engine_selector % 3);
        
        uint8_t scramble_selector = Data[offset++];
        bool scramble = (scramble_selector % 2 == 1);
        
        uint8_t seed_selector = Data[offset++];
        int64_t seed = static_cast<int64_t>(seed_selector);
        
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .layout(layout)
            .requires_grad(requires_grad)
            .pinned_memory(pin_memory);
        
        torch::Tensor result1 = torch::quasirandom(n, d, options);
        
        torch::Tensor result2 = torch::quasirandom(n, d, engine, options);
        
        torch::Tensor result3 = torch::quasirandom(n, d, engine, scramble, options);
        
        torch::Tensor result4 = torch::quasirandom(n, d, engine, scramble, seed, options);
        
        if (offset < Size) {
            int64_t large_n = static_cast<int64_t>(Data[offset] % 200) * 1000;
            torch::Tensor result5 = torch::quasirandom(large_n, d, options);
        }
        
        if (offset + 1 < Size) {
            int64_t large_d = static_cast<int64_t>(Data[offset + 1] % 50) + 1000;
            torch::Tensor result6 = torch::quasirandom(n, large_d, options);
        }
        
        torch::Tensor result7 = torch::quasirandom(0, d, options);
        torch::Tensor result8 = torch::quasirandom(n, 1, options);
        
        int64_t negative_n = -static_cast<int64_t>(n_byte + 1);
        torch::Tensor result9 = torch::quasirandom(negative_n, d, options);
        
        int64_t negative_d = -static_cast<int64_t>(d_byte + 1);
        torch::Tensor result10 = torch::quasirandom(n, negative_d, options);
        
        int64_t negative_engine = -static_cast<int64_t>(engine_selector + 1);
        torch::Tensor result11 = torch::quasirandom(n, d, negative_engine, options);
        
        int64_t negative_seed = -static_cast<int64_t>(seed_selector + 1);
        torch::Tensor result12 = torch::quasirandom(n, d, engine, scramble, negative_seed, options);
        
        int64_t max_n = std::numeric_limits<int64_t>::max();
        torch::Tensor result13 = torch::quasirandom(max_n, d, options);
        
        int64_t max_d = std::numeric_limits<int64_t>::max();
        torch::Tensor result14 = torch::quasirandom(n, max_d, options);
        
        int64_t min_n = std::numeric_limits<int64_t>::min();
        torch::Tensor result15 = torch::quasirandom(min_n, d, options);
        
        int64_t min_d = std::numeric_limits<int64_t>::min();
        torch::Tensor result16 = torch::quasirandom(n, min_d, options);
        
        int64_t large_engine = std::numeric_limits<int64_t>::max();
        torch::Tensor result17 = torch::quasirandom(n, d, large_engine, options);
        
        int64_t large_seed = std::numeric_limits<int64_t>::max();
        torch::Tensor result18 = torch::quasirandom(n, d, engine, scramble, large_seed, options);
        
        torch::Tensor result19 = torch::quasirandom(1, 1, options);
        
        if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
            torch::Tensor result20 = torch::quasirandom(n, d, options);
        }
        
        if (dtype == torch::kBool) {
            torch::Tensor result21 = torch::quasirandom(n, d, options);
        }
        
        if (dtype == torch::kInt8 || dtype == torch::kUInt8) {
            torch::Tensor result22 = torch::quasirandom(n, d, options);
        }
        
        auto sparse_options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .layout(torch::kSparse)
            .requires_grad(requires_grad);
        torch::Tensor result23 = torch::quasirandom(n, d, sparse_options);
        
        if (device == torch::kCUDA) {
            auto cuda_options = torch::TensorOptions()
                .dtype(dtype)
                .device(torch::kCUDA)
                .requires_grad(requires_grad);
            torch::Tensor result24 = torch::quasirandom(n, d, cuda_options);
        }
        
        auto pinned_options = torch::TensorOptions()
            .dtype(dtype)
            .device(torch::kCPU)
            .pinned_memory(true);
        torch::Tensor result25 = torch::quasirandom(n, d, pinned_options);
        
        auto grad_options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .requires_grad(true);
        torch::Tensor result26 = torch::quasirandom(n, d, grad_options);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}