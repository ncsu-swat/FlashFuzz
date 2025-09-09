#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some bytes for tensor size, r value, and with_replacement flag
        if (Size < 6) {
            return 0;
        }

        // Extract tensor size (1-20 elements to keep combinations manageable)
        uint8_t tensor_size_raw = Data[offset++];
        int tensor_size = (tensor_size_raw % 20) + 1;

        // Extract r value (0 to tensor_size + 2 to test edge cases)
        uint8_t r_raw = Data[offset++];
        int r = r_raw % (tensor_size + 3);

        // Extract with_replacement flag
        bool with_replacement = (Data[offset++] % 2) == 1;

        // Extract data type choice
        uint8_t dtype_choice = Data[offset++] % 6;

        // Create input tensor based on available data
        torch::Tensor input;
        
        switch (dtype_choice) {
            case 0: { // int32
                std::vector<int32_t> values;
                for (int i = 0; i < tensor_size && offset < Size; ++i) {
                    int32_t val = static_cast<int32_t>(Data[offset++]);
                    if (offset < Size) {
                        val |= (static_cast<int32_t>(Data[offset++]) << 8);
                    }
                    values.push_back(val);
                }
                // Fill remaining with zeros if not enough data
                while (values.size() < tensor_size) {
                    values.push_back(0);
                }
                input = torch::tensor(values, torch::kInt32);
                break;
            }
            case 1: { // int64
                std::vector<int64_t> values;
                for (int i = 0; i < tensor_size && offset < Size; ++i) {
                    int64_t val = static_cast<int64_t>(Data[offset++]);
                    values.push_back(val);
                }
                while (values.size() < tensor_size) {
                    values.push_back(0);
                }
                input = torch::tensor(values, torch::kInt64);
                break;
            }
            case 2: { // float32
                std::vector<float> values;
                for (int i = 0; i < tensor_size && offset < Size; ++i) {
                    float val = static_cast<float>(Data[offset++]) / 255.0f;
                    values.push_back(val);
                }
                while (values.size() < tensor_size) {
                    values.push_back(0.0f);
                }
                input = torch::tensor(values, torch::kFloat32);
                break;
            }
            case 3: { // double
                std::vector<double> values;
                for (int i = 0; i < tensor_size && offset < Size; ++i) {
                    double val = static_cast<double>(Data[offset++]) / 255.0;
                    values.push_back(val);
                }
                while (values.size() < tensor_size) {
                    values.push_back(0.0);
                }
                input = torch::tensor(values, torch::kFloat64);
                break;
            }
            case 4: { // bool
                std::vector<bool> values;
                for (int i = 0; i < tensor_size && offset < Size; ++i) {
                    bool val = (Data[offset++] % 2) == 1;
                    values.push_back(val);
                }
                while (values.size() < tensor_size) {
                    values.push_back(false);
                }
                input = torch::tensor(values, torch::kBool);
                break;
            }
            default: { // int8
                std::vector<int8_t> values;
                for (int i = 0; i < tensor_size && offset < Size; ++i) {
                    int8_t val = static_cast<int8_t>(Data[offset++]);
                    values.push_back(val);
                }
                while (values.size() < tensor_size) {
                    values.push_back(0);
                }
                input = torch::tensor(values, torch::kInt8);
                break;
            }
        }

        // Test different combinations calls
        
        // Test 1: Default parameters (r=2, with_replacement=False)
        auto result1 = torch::combinations(input);
        
        // Test 2: With specified r value
        auto result2 = torch::combinations(input, r);
        
        // Test 3: With specified r and with_replacement
        auto result3 = torch::combinations(input, r, with_replacement);
        
        // Test 4: Edge cases
        if (r == 0) {
            auto result4 = torch::combinations(input, 0, with_replacement);
        }
        
        if (r == 1) {
            auto result5 = torch::combinations(input, 1, with_replacement);
        }
        
        // Test 5: r larger than input size (should return empty tensor)
        if (tensor_size > 0) {
            auto result6 = torch::combinations(input, tensor_size + 1, false);
        }
        
        // Test 6: Empty tensor edge case
        if (offset < Size && (Data[offset++] % 10) == 0) {
            auto empty_input = torch::tensor({}, input.dtype());
            auto empty_result = torch::combinations(empty_input, r, with_replacement);
        }
        
        // Test 7: Single element tensor
        if (offset < Size && (Data[offset++] % 10) == 1) {
            auto single_input = input.slice(0, 0, 1);
            auto single_result = torch::combinations(single_input, r, with_replacement);
        }
        
        // Test 8: Large r with replacement (test memory limits)
        if (with_replacement && tensor_size <= 5 && r <= 10) {
            auto large_r_result = torch::combinations(input, r, true);
        }
        
        // Test 9: Verify output properties
        if (result1.numel() > 0) {
            // Check that result has correct number of dimensions
            if (result1.dim() != 2) {
                throw std::runtime_error("Result should be 2D");
            }
            
            // Check that second dimension equals r (default 2)
            if (result1.size(1) != 2) {
                throw std::runtime_error("Second dimension should equal r");
            }
        }
        
        if (result2.numel() > 0 && r > 0) {
            if (result2.dim() != 2) {
                throw std::runtime_error("Result2 should be 2D");
            }
            
            if (result2.size(1) != r) {
                throw std::runtime_error("Second dimension should equal r");
            }
        }
        
        // Test 10: Different tensor layouts/strides
        if (offset < Size && (Data[offset++] % 5) == 0) {
            auto contiguous_input = input.contiguous();
            auto contiguous_result = torch::combinations(contiguous_input, r, with_replacement);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}