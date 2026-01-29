#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>
#include <limits>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }

        size_t offset = 0;
        
        // Use first byte to select test variant
        uint8_t variant = Data[0] % 8;
        offset = 1;

        switch (variant) {
            case 0: {
                // Basic torch::fix on fuzzer-generated tensor
                torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
                torch::Tensor result = torch::fix(input);
                break;
            }
            case 1: {
                // torch::fix with float tensor conversion
                torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
                torch::Tensor float_input = input.to(torch::kFloat);
                torch::Tensor result = torch::fix(float_input);
                break;
            }
            case 2: {
                // torch::fix with double tensor conversion
                torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
                torch::Tensor double_input = input.to(torch::kDouble);
                torch::Tensor result = torch::fix(double_input);
                break;
            }
            case 3: {
                // torch::fix with out tensor
                torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
                torch::Tensor float_input = input.to(torch::kFloat);
                torch::Tensor output = torch::empty_like(float_input);
                torch::fix_outf(float_input, output);
                break;
            }
            case 4: {
                // torch::fix on empty tensor
                torch::Tensor empty_tensor = torch::empty({0}, torch::kFloat);
                torch::Tensor result = torch::fix(empty_tensor);
                break;
            }
            case 5: {
                // torch::fix on scalar tensor
                if (Size > offset + 4) {
                    float val;
                    memcpy(&val, Data + offset, sizeof(float));
                    torch::Tensor scalar_tensor = torch::tensor(val);
                    torch::Tensor result = torch::fix(scalar_tensor);
                }
                break;
            }
            case 6: {
                // torch::fix with negative values
                torch::Tensor neg_tensor = torch::tensor({-3.7f, -2.1f, -0.9f, 0.0f, 0.9f, 2.1f, 3.7f});
                torch::Tensor result = torch::fix(neg_tensor);
                break;
            }
            case 7: {
                // torch::fix with NaN and Inf values
                torch::Tensor special_tensor = torch::tensor({
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    0.0f
                });
                torch::Tensor result = torch::fix(special_tensor);
                break;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}