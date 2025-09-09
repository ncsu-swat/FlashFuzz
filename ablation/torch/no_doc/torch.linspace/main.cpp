#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 20) {
            return 0;
        }

        double start_val = 0.0;
        double end_val = 1.0;
        int64_t steps = 50;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&start_val, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&end_val, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(int32_t) <= Size) {
            int32_t steps_raw;
            std::memcpy(&steps_raw, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            steps = static_cast<int64_t>(std::abs(steps_raw)) % 10000;
        }
        
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset]);
            offset++;
        }
        
        torch::Device device = torch::kCPU;
        bool requires_grad = false;
        
        if (offset < Size) {
            requires_grad = (Data[offset] % 2) == 1;
            offset++;
        }
        
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .requires_grad(requires_grad);
        
        torch::Tensor result = torch::linspace(start_val, end_val, steps, options);
        
        if (offset < Size) {
            int64_t steps_alt = static_cast<int64_t>(Data[offset] % 100);
            torch::Tensor result2 = torch::linspace(start_val, end_val, steps_alt);
        }
        
        if (offset + 1 < Size) {
            double start_alt = static_cast<double>(static_cast<int8_t>(Data[offset]));
            double end_alt = static_cast<double>(static_cast<int8_t>(Data[offset + 1]));
            offset += 2;
            torch::Tensor result3 = torch::linspace(start_alt, end_alt, 10);
        }
        
        torch::Tensor edge_case1 = torch::linspace(0.0, 0.0, 1);
        torch::Tensor edge_case2 = torch::linspace(-1e10, 1e10, 2);
        torch::Tensor edge_case3 = torch::linspace(std::numeric_limits<double>::lowest(), 
                                                   std::numeric_limits<double>::max(), 3);
        
        if (steps > 0) {
            torch::Tensor large_steps = torch::linspace(0.0, 1.0, steps);
        }
        
        torch::Tensor zero_steps = torch::linspace(start_val, end_val, 0);
        torch::Tensor one_step = torch::linspace(start_val, end_val, 1);
        
        if (offset < Size) {
            bool use_inf = (Data[offset] % 4) == 0;
            if (use_inf) {
                torch::Tensor inf_test = torch::linspace(std::numeric_limits<double>::infinity(), 
                                                        -std::numeric_limits<double>::infinity(), 5);
            }
            offset++;
        }
        
        if (offset < Size) {
            bool use_nan = (Data[offset] % 4) == 1;
            if (use_nan) {
                torch::Tensor nan_test = torch::linspace(std::numeric_limits<double>::quiet_NaN(), 
                                                        1.0, 3);
            }
            offset++;
        }
        
        for (auto test_dtype : {torch::kFloat, torch::kDouble, torch::kHalf}) {
            auto test_options = torch::TensorOptions().dtype(test_dtype);
            torch::Tensor dtype_test = torch::linspace(-1.0, 1.0, 10, test_options);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            int64_t large_steps_raw;
            std::memcpy(&large_steps_raw, Data + offset, sizeof(int64_t));
            int64_t large_steps_val = std::abs(large_steps_raw) % 1000000;
            if (large_steps_val > 0) {
                torch::Tensor large_result = torch::linspace(0.0, 1.0, large_steps_val);
            }
            offset += sizeof(int64_t);
        }
        
        torch::Tensor negative_steps_test = torch::linspace(1.0, 0.0, 10);
        
        if (offset + 2 * sizeof(float) <= Size) {
            float start_f, end_f;
            std::memcpy(&start_f, Data + offset, sizeof(float));
            std::memcpy(&end_f, Data + offset + sizeof(float), sizeof(float));
            torch::Tensor float_test = torch::linspace(static_cast<double>(start_f), 
                                                      static_cast<double>(end_f), 7);
            offset += 2 * sizeof(float);
        }
        
        auto complex_options = torch::TensorOptions().dtype(torch::kComplexFloat);
        torch::Tensor complex_test = torch::linspace(0.0, 1.0, 5, complex_options);
        
        if (result.numel() > 0) {
            auto sum_result = torch::sum(result);
            auto mean_result = torch::mean(result);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}