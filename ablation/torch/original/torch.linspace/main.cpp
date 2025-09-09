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
        double end_val = 0.0;
        int64_t steps = 1;
        
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
            steps = std::abs(steps_raw) % 1000 + 1;
        }

        torch::Tensor result1 = torch::linspace(start_val, end_val, steps);

        if (offset + 2 <= Size) {
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset]);
            offset++;
            
            auto options = torch::TensorOptions().dtype(dtype);
            torch::Tensor result2 = torch::linspace(start_val, end_val, steps, options);
        }

        if (offset + 1 <= Size) {
            bool requires_grad = Data[offset] % 2 == 1;
            offset++;
            
            auto options = torch::TensorOptions().requires_grad(requires_grad);
            torch::Tensor result3 = torch::linspace(start_val, end_val, steps, options);
        }

        if (offset + 2 <= Size) {
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset]);
            offset++;
            bool requires_grad = Data[offset] % 2 == 1;
            offset++;
            
            auto options = torch::TensorOptions().dtype(dtype).requires_grad(requires_grad);
            torch::Tensor result4 = torch::linspace(start_val, end_val, steps, options);
        }

        if (Size - offset >= 10) {
            try {
                torch::Tensor start_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                if (start_tensor.numel() == 1) {
                    torch::Tensor result5 = torch::linspace(start_tensor, end_val, steps);
                }
            } catch (...) {
            }
        }

        if (Size - offset >= 10) {
            try {
                torch::Tensor end_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                if (end_tensor.numel() == 1) {
                    torch::Tensor result6 = torch::linspace(start_val, end_tensor, steps);
                }
            } catch (...) {
            }
        }

        if (Size - offset >= 20) {
            try {
                size_t temp_offset = offset;
                torch::Tensor start_tensor = fuzzer_utils::createTensor(Data, Size, temp_offset);
                torch::Tensor end_tensor = fuzzer_utils::createTensor(Data, Size, temp_offset);
                if (start_tensor.numel() == 1 && end_tensor.numel() == 1) {
                    torch::Tensor result7 = torch::linspace(start_tensor, end_tensor, steps);
                }
            } catch (...) {
            }
        }

        if (offset < Size) {
            int64_t edge_steps = Data[offset] % 10;
            if (edge_steps == 0) edge_steps = 1;
            torch::Tensor result8 = torch::linspace(start_val, end_val, edge_steps);
        }

        torch::Tensor result_large = torch::linspace(-1e10, 1e10, steps);
        torch::Tensor result_small = torch::linspace(-1e-10, 1e-10, steps);
        torch::Tensor result_inf = torch::linspace(std::numeric_limits<double>::lowest(), 
                                                  std::numeric_limits<double>::max(), steps);

        if (steps > 1) {
            torch::Tensor result_reverse = torch::linspace(end_val, start_val, steps);
        }

        torch::Tensor result_same = torch::linspace(start_val, start_val, steps);

        if (offset < Size) {
            torch::ScalarType complex_dtype = (Data[offset] % 2 == 0) ? torch::kComplexFloat : torch::kComplexDouble;
            auto complex_options = torch::TensorOptions().dtype(complex_dtype);
            torch::Tensor result_complex = torch::linspace(start_val, end_val, steps, complex_options);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}