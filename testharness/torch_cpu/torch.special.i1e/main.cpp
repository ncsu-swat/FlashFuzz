#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for torch.special.i1e
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.i1e operation
        torch::Tensor result = torch::special::i1e(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto item = result.item<float>();
            volatile float unused = item; // Prevent optimization
            (void)unused;
        }
        
        // Try with different input types if we have more data
        if (offset + 2 < Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            torch::Tensor result2 = torch::special::i1e(input2);
            
            // Try to access the result
            if (result2.defined() && result2.numel() > 0) {
                auto item = result2.item<float>();
                volatile float unused = item;
                (void)unused;
            }
        }
        
        // Try with a scalar input if we have more data
        if (offset + 1 < Size) {
            double scalar_value = static_cast<double>(Data[offset]) / 255.0;
            torch::Tensor scalar_input = torch::tensor(scalar_value);
            torch::Tensor scalar_result = torch::special::i1e(scalar_input);
            
            if (scalar_result.defined() && scalar_result.numel() > 0) {
                auto item = scalar_result.item<double>();
                volatile double unused = item;
                (void)unused;
            }
        }
        
        // Try with extreme values
        std::vector<torch::Tensor> extreme_inputs = {
            torch::tensor(std::numeric_limits<float>::max()),
            torch::tensor(std::numeric_limits<float>::lowest()),
            torch::tensor(std::numeric_limits<float>::infinity()),
            torch::tensor(-std::numeric_limits<float>::infinity()),
            torch::tensor(std::numeric_limits<float>::quiet_NaN()),
            torch::tensor(0.0f),
            torch::tensor(-0.0f),
            torch::tensor(1.0f),
            torch::tensor(-1.0f)
        };
        
        for (const auto& extreme_input : extreme_inputs) {
            try {
                torch::Tensor extreme_result = torch::special::i1e(extreme_input);
                if (extreme_result.defined() && extreme_result.numel() > 0) {
                    auto item = extreme_result.item<float>();
                    volatile float unused = item;
                    (void)unused;
                }
            } catch (const std::exception& e) {
                // Just catch and continue with the next input
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
