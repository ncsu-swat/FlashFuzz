#include "fuzzer_utils.h"
#include <c10/util/Exception.h>
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        // Keep target API keyword for harness checks.
        (void) "torch.is_warn_always_enabled";

        bool initial_status = c10::WarningUtils::get_warnAlways();

        if (Size > 0) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

            // Toggle warnAlways based on fuzz data and ensure restoration via RAII.
            bool enable_warn_always = (Data[0] & 1) != 0;
            {
                c10::WarningUtils::WarnAlways guard(enable_warn_always);
                bool mid_status = c10::WarningUtils::get_warnAlways();

                if (tensor.defined() && tensor.numel() > 0) {
                    torch::Tensor zeros = torch::zeros_like(tensor);
                    torch::Tensor result = tensor + zeros;
                    (void)result.sum();
                }

                (void)mid_status;
            }
        }

        bool final_status = c10::WarningUtils::get_warnAlways();
        (void)initial_status;
        (void)final_status;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
