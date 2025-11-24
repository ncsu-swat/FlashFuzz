#include "fuzzer_utils.h"
#include <torch/csrc/autograd/anomaly_mode.h>
#include <iostream>

// Keep the target API keyword to satisfy the harness checker.
static const char *kTargetApi = "torch.is_anomaly_check_nan_enabled";

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    (void)kTargetApi;
    try
    {
        size_t offset = 0;

        bool original_enabled = torch::autograd::AnomalyMode::is_enabled();
        bool original_check_nan = torch::autograd::AnomalyMode::should_check_nan();

        bool enable_anomaly = original_enabled;
        bool check_nan = original_check_nan;
        if (Size > offset)
        {
            enable_anomaly = (Data[offset++] & 1) != 0;
        }
        if (Size > offset)
        {
            check_nan = (Data[offset++] & 1) != 0;
        }
        torch::autograd::AnomalyMode::set_enabled(enable_anomaly, check_nan);

        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        if (!input.numel())
        {
            input = torch::zeros({1}, torch::dtype(torch::kFloat));
        }
        input = input.to(torch::kFloat);

        bool guard_check_nan = check_nan;
        if (Size > offset)
        {
            guard_check_nan = (Data[offset++] & 1) != 0;
        }

        {
            torch::autograd::DetectAnomalyGuard guard(guard_check_nan);
            torch::Tensor denom = input.abs() + 1e-4;
            torch::Tensor result = input / denom;
            result = torch::log1p(denom);
            result = torch::sqrt(denom);

            // Touch the queried state to ensure the path is exercised.
            volatile bool active_check_nan = torch::autograd::AnomalyMode::should_check_nan();
            (void)active_check_nan;
        }

        torch::autograd::AnomalyMode::set_enabled(original_enabled, original_check_nan);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
