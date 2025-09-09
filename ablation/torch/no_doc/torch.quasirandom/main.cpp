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
        int64_t d = static_cast<int64_t>(d_byte % 10) + 1;
        
        uint8_t dtype_selector = Data[offset++];
        auto dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        uint8_t device_selector = Data[offset++];
        torch::Device device = (device_selector % 2 == 0) ? torch::kCPU : torch::kCUDA;
        
        uint8_t layout_selector = Data[offset++];
        torch::Layout layout = (layout_selector % 2 == 0) ? torch::kStrided : torch::kSparse;
        
        uint8_t requires_grad_selector = Data[offset++];
        bool requires_grad = (requires_grad_selector % 2 == 0);
        
        uint8_t pin_memory_selector = Data[offset++];
        bool pin_memory = (pin_memory_selector % 2 == 0);
        
        uint8_t engine_selector = Data[offset++];
        int64_t engine = static_cast<int64_t>(engine_selector % 3);
        
        uint8_t scramble_selector = Data[offset++];
        bool scramble = (scramble_selector % 2 == 0);
        
        uint8_t seed_selector = Data[offset++];
        int64_t seed = static_cast<int64_t>(seed_selector);

        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .layout(layout)
            .requires_grad(requires_grad)
            .pinned_memory(pin_memory);

        torch::Tensor result1 = torch::quasirandom(n, d, options);
        
        torch::Tensor result2 = torch::quasirandom(n, d, dtype);
        
        torch::Tensor result3 = torch::quasirandom(n, d, torch::TensorOptions().dtype(dtype).device(device));
        
        torch::Tensor result4 = torch::quasirandom(n, d, torch::TensorOptions().dtype(dtype).requires_grad(requires_grad));
        
        torch::Tensor result5 = torch::quasirandom(n, d, torch::TensorOptions().dtype(dtype).layout(layout));
        
        if (offset + 8 <= Size) {
            int64_t large_n;
            std::memcpy(&large_n, Data + offset, 8);
            offset += 8;
            large_n = std::abs(large_n) % 10000;
            torch::Tensor result6 = torch::quasirandom(large_n, d, dtype);
        }
        
        if (offset + 8 <= Size) {
            int64_t large_d;
            std::memcpy(&large_d, Data + offset, 8);
            offset += 8;
            large_d = std::abs(large_d) % 1000 + 1;
            torch::Tensor result7 = torch::quasirandom(n, large_d, dtype);
        }
        
        torch::Tensor result8 = torch::quasirandom(0, d, dtype);
        
        torch::Tensor result9 = torch::quasirandom(n, 1, dtype);
        
        torch::Tensor result10 = torch::quasirandom(1, 1, dtype);
        
        if (offset < Size) {
            int64_t neg_n = -static_cast<int64_t>(Data[offset++]);
            torch::Tensor result11 = torch::quasirandom(neg_n, d, dtype);
        }
        
        if (offset < Size) {
            int64_t neg_d = -static_cast<int64_t>(Data[offset++]);
            torch::Tensor result12 = torch::quasirandom(n, neg_d, dtype);
        }
        
        torch::Tensor result13 = torch::quasirandom(std::numeric_limits<int64_t>::max(), 1, dtype);
        
        torch::Tensor result14 = torch::quasirandom(1, std::numeric_limits<int64_t>::max(), dtype);
        
        torch::Tensor result15 = torch::quasirandom(std::numeric_limits<int64_t>::min(), 1, dtype);
        
        torch::Tensor result16 = torch::quasirandom(1, std::numeric_limits<int64_t>::min(), dtype);

        auto complex_options = torch::TensorOptions()
            .dtype(torch::kComplexFloat)
            .device(device)
            .requires_grad(requires_grad);
        torch::Tensor result17 = torch::quasirandom(n, d, complex_options);
        
        auto bool_options = torch::TensorOptions()
            .dtype(torch::kBool)
            .device(device);
        torch::Tensor result18 = torch::quasirandom(n, d, bool_options);
        
        auto int8_options = torch::TensorOptions()
            .dtype(torch::kInt8)
            .device(device);
        torch::Tensor result19 = torch::quasirandom(n, d, int8_options);
        
        auto uint8_options = torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(device);
        torch::Tensor result20 = torch::quasirandom(n, d, uint8_options);

        if (offset + 2 <= Size) {
            uint8_t zero_n = Data[offset++] % 2;
            uint8_t zero_d = Data[offset++] % 2;
            torch::Tensor result21 = torch::quasirandom(zero_n, zero_d, dtype);
        }

        torch::Tensor result22 = torch::quasirandom(n, d, torch::TensorOptions().dtype(torch::kHalf));
        torch::Tensor result23 = torch::quasirandom(n, d, torch::TensorOptions().dtype(torch::kBFloat16));
        torch::Tensor result24 = torch::quasirandom(n, d, torch::TensorOptions().dtype(torch::kComplexDouble));

        if (offset + 16 <= Size) {
            int64_t huge_n, huge_d;
            std::memcpy(&huge_n, Data + offset, 8);
            std::memcpy(&huge_d, Data + offset + 8, 8);
            offset += 16;
            torch::Tensor result25 = torch::quasirandom(huge_n, huge_d, dtype);
        }

        auto sparse_options = torch::TensorOptions()
            .dtype(dtype)
            .layout(torch::kSparse);
        torch::Tensor result26 = torch::quasirandom(n, d, sparse_options);

        if (device == torch::kCUDA) {
            auto cuda_options = torch::TensorOptions()
                .dtype(dtype)
                .device(torch::kCUDA)
                .pinned_memory(pin_memory);
            torch::Tensor result27 = torch::quasirandom(n, d, cuda_options);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}