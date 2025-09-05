#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>
#include <cmath>

// Helper to consume a value from fuzzer data
template<typename T>
T consumeValue(const uint8_t* data, size_t& offset, size_t size, T default_val = T{}) {
    if (offset + sizeof(T) > size) {
        return default_val;
    }
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 4) {
            // Need at least some bytes for basic parameters
            return 0;
        }

        size_t offset = 0;

        // Parse dtype for the range tensor
        uint8_t dtype_selector = consumeValue<uint8_t>(Data, offset, Size, 0);
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // torch.range only supports floating point types
        // Filter to valid types for range operation
        if (dtype != torch::kFloat && dtype != torch::kDouble && 
            dtype != torch::kHalf && dtype != torch::kBFloat16) {
            // Default to float if invalid type selected
            dtype = torch::kFloat;
        }

        // Parse start, end, and step values
        // Use different strategies based on remaining data
        double start = 0.0, end = 10.0, step = 1.0;
        
        if (offset < Size) {
            // Strategy 1: Parse raw doubles
            if (Size - offset >= sizeof(double) * 3) {
                start = consumeValue<double>(Data, offset, Size, 0.0);
                end = consumeValue<double>(Data, offset, Size, 10.0);
                step = consumeValue<double>(Data, offset, Size, 1.0);
            }
            // Strategy 2: Parse floats for smaller ranges
            else if (Size - offset >= sizeof(float) * 3) {
                start = static_cast<double>(consumeValue<float>(Data, offset, Size, 0.0f));
                end = static_cast<double>(consumeValue<float>(Data, offset, Size, 10.0f));
                step = static_cast<double>(consumeValue<float>(Data, offset, Size, 1.0f));
            }
            // Strategy 3: Parse int8 for compact representation
            else if (Size - offset >= 3) {
                int8_t start_i8 = consumeValue<int8_t>(Data, offset, Size, 0);
                int8_t end_i8 = consumeValue<int8_t>(Data, offset, Size, 10);
                int8_t step_i8 = consumeValue<int8_t>(Data, offset, Size, 1);
                start = static_cast<double>(start_i8);
                end = static_cast<double>(end_i8);
                step = static_cast<double>(step_i8) / 10.0; // Allow fractional steps
            }
        }

        // Handle special cases from fuzzer
        if (offset < Size) {
            uint8_t special_case = consumeValue<uint8_t>(Data, offset, Size, 0);
            switch (special_case % 10) {
                case 0: // Normal case, use parsed values
                    break;
                case 1: // Infinity values
                    if (special_case & 0x10) start = std::numeric_limits<double>::infinity();
                    if (special_case & 0x20) end = -std::numeric_limits<double>::infinity();
                    break;
                case 2: // NaN values
                    if (special_case & 0x10) start = std::numeric_limits<double>::quiet_NaN();
                    if (special_case & 0x20) end = std::numeric_limits<double>::quiet_NaN();
                    if (special_case & 0x40) step = std::numeric_limits<double>::quiet_NaN();
                    break;
                case 3: // Zero step (edge case)
                    step = 0.0;
                    break;
                case 4: // Negative step with start < end
                    step = -std::abs(step);
                    if (start < end) {
                        std::swap(start, end);
                    }
                    break;
                case 5: // Very large range
                    start = -1e10;
                    end = 1e10;
                    step = 1e8;
                    break;
                case 6: // Very small step
                    step = 1e-10;
                    end = start + 100 * step; // Limit range to avoid huge tensors
                    break;
                case 7: // Reverse range
                    if (start < end && step > 0) {
                        step = -step;
                    }
                    break;
                case 8: // Equal start and end
                    end = start;
                    break;
                case 9: // Very large values
                    double scale = 1e100;
                    start *= scale;
                    end *= scale;
                    step *= scale;
                    break;
            }
        }

        // Set tensor options
        auto options = torch::TensorOptions().dtype(dtype);
        
        // Parse device selection if enough data
        if (offset < Size) {
            uint8_t device_selector = consumeValue<uint8_t>(Data, offset, Size, 0);
            if ((device_selector % 4) == 1 && torch::cuda::is_available()) {
                options = options.device(torch::kCUDA);
            }
        }

        // Create tensor using torch.range
        // Note: torch.range is deprecated but still available
        torch::Tensor result;
        
        // Try different API variations based on fuzzer input
        if (offset < Size) {
            uint8_t api_variant = consumeValue<uint8_t>(Data, offset, Size, 0);
            switch (api_variant % 4) {
                case 0:
                    // Standard call with all parameters
                    result = torch::range(start, end, step, options);
                    break;
                case 1:
                    // Call with default step
                    result = torch::range(start, end, options);
                    break;
                case 2:
                    // Create on CPU then move to device
                    result = torch::range(start, end, step, torch::TensorOptions().dtype(dtype));
                    if (options.device() == torch::kCUDA && torch::cuda::is_available()) {
                        result = result.to(torch::kCUDA);
                    }
                    break;
                case 3:
                    // Try with explicit scalar type conversion
                    if (dtype == torch::kHalf || dtype == torch::kBFloat16) {
                        // For half types, create in float then convert
                        result = torch::range(start, end, step, torch::TensorOptions().dtype(torch::kFloat));
                        result = result.to(dtype);
                    } else {
                        result = torch::range(start, end, step, options);
                    }
                    break;
            }
        } else {
            // Default case
            result = torch::range(start, end, step, options);
        }

        // Perform operations on the result to exercise more code paths
        if (result.defined() && result.numel() > 0) {
            // Test various properties
            auto size = result.size(0);
            auto numel = result.numel();
            auto is_contiguous = result.is_contiguous();
            auto device = result.device();
            
            // Try some operations based on fuzzer input
            if (offset < Size) {
                uint8_t op_selector = consumeValue<uint8_t>(Data, offset, Size, 0);
                switch (op_selector % 8) {
                    case 0:
                        // Sum reduction
                        if (result.numel() < 1000000) { // Avoid huge computations
                            auto sum_result = result.sum();
                        }
                        break;
                    case 1:
                        // Mean
                        if (result.numel() > 0 && result.numel() < 1000000) {
                            auto mean_result = result.mean();
                        }
                        break;
                    case 2:
                        // Min/Max
                        if (result.numel() > 0 && result.numel() < 1000000) {
                            auto min_result = result.min();
                            auto max_result = result.max();
                        }
                        break;
                    case 3:
                        // Clone
                        auto cloned = result.clone();
                        break;
                    case 4:
                        // Reshape to 2D if possible
                        if (result.numel() > 1 && result.numel() % 2 == 0) {
                            auto reshaped = result.reshape({2, -1});
                        }
                        break;
                    case 5:
                        // Type conversion
                        if (dtype != torch::kInt64 && result.numel() < 10000) {
                            auto converted = result.to(torch::kInt64);
                        }
                        break;
                    case 6:
                        // Slice
                        if (result.numel() > 2) {
                            auto sliced = result.slice(0, 1, std::min(static_cast<int64_t>(3), result.size(0)));
                        }
                        break;
                    case 7:
                        // Negative indexing
                        if (result.numel() > 0) {
                            auto last_elem = result[-1];
                        }
                        break;
                }
            }
        }

        // Test edge cases with another range call using modified parameters
        if (offset < Size) {
            uint8_t edge_case = consumeValue<uint8_t>(Data, offset, Size, 0);
            if (edge_case % 4 == 0) {
                // Try creating empty range (start > end with positive step)
                auto empty_range = torch::range(10.0, 1.0, 1.0, options);
            } else if (edge_case % 4 == 1) {
                // Single element range
                auto single = torch::range(5.0, 5.0, 1.0, options);
            }
        }

        return 0;
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected for invalid inputs
        return 0; // Continue fuzzing
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    catch (...)
    {
        // Catch any other exceptions
        return -1;
    }
}