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

// Parse device type from fuzzer input
torch::Device parseDevice(uint8_t selector) {
    // Only use CPU for fuzzing to avoid GPU-specific issues
    return torch::kCPU;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) {
        return 0;  // Need minimum bytes for basic parameters
    }

    try {
        size_t offset = 0;
        
        // 1. Parse dtype selector
        uint8_t dtype_selector = consumeValue<uint8_t>(data, offset, size);
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // 2. Parse device selector
        uint8_t device_selector = consumeValue<uint8_t>(data, offset, size);
        torch::Device device = parseDevice(device_selector);
        
        // 3. Parse requires_grad flag
        bool requires_grad = consumeValue<uint8_t>(data, offset, size) & 1;
        
        // 4. Parse mode selector (different arange overloads)
        uint8_t mode = consumeValue<uint8_t>(data, offset, size) % 4;
        
        // 5. Parse numerical values for start, end, step
        // Use double for maximum flexibility, will be cast as needed
        double start_raw = consumeValue<double>(data, offset, size);
        double end_raw = consumeValue<double>(data, offset, size);
        double step_raw = consumeValue<double>(data, offset, size);
        
        // 6. Constrain values to reasonable ranges to avoid hangs
        const double MAX_RANGE = 10000.0;
        const double MIN_STEP = 0.0001;
        
        // Normalize start and end
        start_raw = std::fmod(start_raw, MAX_RANGE);
        end_raw = std::fmod(end_raw, MAX_RANGE);
        
        // Ensure step is non-zero
        if (std::abs(step_raw) < MIN_STEP) {
            step_raw = (step_raw >= 0) ? MIN_STEP : -MIN_STEP;
        }
        step_raw = std::fmod(step_raw, MAX_RANGE);
        if (std::abs(step_raw) < MIN_STEP) {
            step_raw = MIN_STEP;
        }
        
        // 7. Additional constraint: limit the number of elements
        const int64_t MAX_ELEMENTS = 100000;
        double num_elements_est = std::abs((end_raw - start_raw) / step_raw);
        if (num_elements_est > MAX_ELEMENTS) {
            // Scale down the range
            double scale = MAX_ELEMENTS / num_elements_est;
            end_raw = start_raw + (end_raw - start_raw) * scale;
        }
        
        // 8. Test different arange overloads based on mode
        torch::Tensor result;
        
        switch (mode) {
            case 0: {
                // arange(end) - single argument version
                if (dtype == torch::kFloat || dtype == torch::kDouble || 
                    dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
                    // Floating point version
                    result = torch::arange(end_raw, torch::TensorOptions()
                        .dtype(dtype)
                        .device(device)
                        .requires_grad(requires_grad && (dtype == torch::kFloat || dtype == torch::kDouble)));
                } else {
                    // Integer version
                    int64_t end_int = static_cast<int64_t>(end_raw);
                    result = torch::arange(end_int, torch::TensorOptions()
                        .dtype(dtype)
                        .device(device)
                        .requires_grad(false));  // Integer tensors don't support gradients
                }
                break;
            }
            case 1: {
                // arange(start, end) - two argument version
                if (dtype == torch::kFloat || dtype == torch::kDouble ||
                    dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
                    result = torch::arange(start_raw, end_raw, torch::TensorOptions()
                        .dtype(dtype)
                        .device(device)
                        .requires_grad(requires_grad && (dtype == torch::kFloat || dtype == torch::kDouble)));
                } else {
                    int64_t start_int = static_cast<int64_t>(start_raw);
                    int64_t end_int = static_cast<int64_t>(end_raw);
                    result = torch::arange(start_int, end_int, torch::TensorOptions()
                        .dtype(dtype)
                        .device(device)
                        .requires_grad(false));
                }
                break;
            }
            case 2: {
                // arange(start, end, step) - three argument version
                if (dtype == torch::kFloat || dtype == torch::kDouble ||
                    dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
                    result = torch::arange(start_raw, end_raw, step_raw, torch::TensorOptions()
                        .dtype(dtype)
                        .device(device)
                        .requires_grad(requires_grad && (dtype == torch::kFloat || dtype == torch::kDouble)));
                } else {
                    int64_t start_int = static_cast<int64_t>(start_raw);
                    int64_t end_int = static_cast<int64_t>(end_raw);
                    int64_t step_int = static_cast<int64_t>(step_raw);
                    if (step_int == 0) step_int = 1;  // Avoid zero step for integers
                    result = torch::arange(start_int, end_int, step_int, torch::TensorOptions()
                        .dtype(dtype)
                        .device(device)
                        .requires_grad(false));
                }
                break;
            }
            case 3: {
                // Test edge cases with special values
                uint8_t edge_case = consumeValue<uint8_t>(data, offset, size) % 6;
                switch (edge_case) {
                    case 0:
                        // Empty range (start == end)
                        result = torch::arange(5.0, 5.0, 1.0, torch::TensorOptions().dtype(dtype).device(device));
                        break;
                    case 1:
                        // Negative step with start < end (should be empty)
                        result = torch::arange(0.0, 10.0, -1.0, torch::TensorOptions().dtype(dtype).device(device));
                        break;
                    case 2:
                        // Positive step with start > end (should be empty)
                        result = torch::arange(10.0, 0.0, 1.0, torch::TensorOptions().dtype(dtype).device(device));
                        break;
                    case 3:
                        // Very small step (test precision)
                        result = torch::arange(0.0, 0.1, 0.01, torch::TensorOptions().dtype(dtype).device(device));
                        break;
                    case 4:
                        // Large negative range
                        result = torch::arange(0.0, -100.0, -1.0, torch::TensorOptions().dtype(dtype).device(device));
                        break;
                    case 5:
                        // Test with infinity/NaN if floating point
                        if (dtype == torch::kFloat || dtype == torch::kDouble) {
                            uint8_t special = consumeValue<uint8_t>(data, offset, size) % 4;
                            switch (special) {
                                case 0:
                                    result = torch::arange(0.0, std::numeric_limits<double>::infinity(), 1.0, 
                                        torch::TensorOptions().dtype(dtype).device(device));
                                    break;
                                case 1:
                                    result = torch::arange(-std::numeric_limits<double>::infinity(), 0.0, 1.0,
                                        torch::TensorOptions().dtype(dtype).device(device));
                                    break;
                                case 2:
                                    result = torch::arange(0.0, 10.0, std::numeric_limits<double>::infinity(),
                                        torch::TensorOptions().dtype(dtype).device(device));
                                    break;
                                default:
                                    result = torch::arange(0.0, 10.0, 1.0, torch::TensorOptions().dtype(dtype).device(device));
                                    break;
                            }
                        } else {
                            result = torch::arange(0, 10, torch::TensorOptions().dtype(dtype).device(device));
                        }
                        break;
                }
                break;
            }
        }
        
        // 9. Perform operations on the result to trigger more code paths
        if (result.numel() > 0) {
            // Test basic properties
            auto shape = result.sizes();
            auto stride = result.strides();
            bool is_contiguous = result.is_contiguous();
            
            // Test with out parameter
            if (result.numel() < 10000) {  // Limit size for out parameter test
                torch::Tensor out_tensor = torch::empty_like(result);
                if (mode == 0) {
                    torch::arange_out(out_tensor, end_raw);
                } else if (mode == 1) {
                    torch::arange_out(out_tensor, start_raw, end_raw);
                } else {
                    torch::arange_out(out_tensor, start_raw, end_raw, step_raw);
                }
            }
            
            // Test gradient computation if applicable
            if (requires_grad && result.requires_grad()) {
                auto sum_result = result.sum();
                sum_result.backward();
            }
            
            // Test type conversions
            if (dtype != torch::kInt64) {
                auto converted = result.to(torch::kInt64);
            }
            if (dtype != torch::kFloat) {
                auto converted = result.to(torch::kFloat);
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}