#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 12) {
            return 0;
        }

        double start = 0.0;
        double end = 1.0;
        double step = 1.0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&start, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&end, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&step, Data + offset, sizeof(double));
            offset += sizeof(double);
        }

        uint8_t dtype_selector = 0;
        if (offset < Size) {
            dtype_selector = Data[offset++];
        }
        
        auto dtype = fuzzer_utils::parseDataType(dtype_selector);

        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 8;
        }

        torch::Tensor result;
        
        switch (variant) {
            case 0:
                result = torch::arange(end);
                break;
            case 1:
                result = torch::arange(start, end);
                break;
            case 2:
                result = torch::arange(start, end, step);
                break;
            case 3:
                result = torch::arange(end, torch::TensorOptions().dtype(dtype));
                break;
            case 4:
                result = torch::arange(start, end, torch::TensorOptions().dtype(dtype));
                break;
            case 5:
                result = torch::arange(start, end, step, torch::TensorOptions().dtype(dtype));
                break;
            case 6: {
                auto options = torch::TensorOptions().dtype(dtype).requires_grad(true);
                result = torch::arange(start, end, step, options);
                break;
            }
            case 7: {
                if (step == 0.0) {
                    step = 1.0;
                }
                result = torch::arange(start, end, step);
                break;
            }
        }

        if (result.numel() > 0) {
            auto sum = torch::sum(result);
            auto mean = torch::mean(result);
            auto min_val = torch::min(result);
            auto max_val = torch::max(result);
        }

        if (result.numel() > 1) {
            auto sliced = result.slice(0, 0, std::min(static_cast<int64_t>(5), result.size(0)));
        }

        auto cloned = result.clone();
        auto detached = result.detach();

        if (result.dtype().isFloatingPoint()) {
            auto rounded = torch::round(result);
        }

        if (result.dtype().isIntegral()) {
            auto float_result = result.to(torch::kFloat);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}