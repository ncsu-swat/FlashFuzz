#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>

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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 4) {
            // Need at least some bytes for basic parameters
            return 0;
        }

        size_t offset = 0;
        
        // Consume control bytes for different arange variants
        uint8_t variant = consumeValue<uint8_t>(Data, offset, Size) % 8;
        uint8_t dtype_selector = consumeValue<uint8_t>(Data, offset, Size);
        uint8_t device_selector = consumeValue<uint8_t>(Data, offset, Size) % 2;
        uint8_t requires_grad = consumeValue<uint8_t>(Data, offset, Size) % 2;
        
        // Parse dtype
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Setup device
        torch::Device device = device_selector ? torch::kCUDA : torch::kCPU;
        if (device == torch::kCUDA && !torch::cuda::is_available()) {
            device = torch::kCPU;
        }
        
        // Setup tensor options
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .requires_grad(requires_grad && torch::isFloatingType(dtype));
        
        torch::Tensor result;
        
        // Test different arange variants based on variant selector
        switch (variant) {
            case 0: {
                // arange(end)
                double end = consumeValue<double>(Data, offset, Size);
                // Limit range to prevent excessive memory usage
                end = std::fmod(end, 10000.0);
                if (std::isnan(end) || std::isinf(end)) {
                    end = 10.0;
                }
                result = torch::arange(end, options);
                break;
            }
            case 1: {
                // arange(start, end)
                double start = consumeValue<double>(Data, offset, Size);
                double end = consumeValue<double>(Data, offset, Size);
                
                // Limit ranges
                start = std::fmod(start, 10000.0);
                end = std::fmod(end, 10000.0);
                
                if (std::isnan(start)) start = 0.0;
                if (std::isnan(end)) end = 10.0;
                if (std::isinf(start)) start = -100.0;
                if (std::isinf(end)) end = 100.0;
                
                result = torch::arange(start, end, options);
                break;
            }
            case 2: {
                // arange(start, end, step)
                double start = consumeValue<double>(Data, offset, Size);
                double end = consumeValue<double>(Data, offset, Size);
                double step = consumeValue<double>(Data, offset, Size);
                
                // Limit ranges
                start = std::fmod(start, 10000.0);
                end = std::fmod(end, 10000.0);
                step = std::fmod(step, 100.0);
                
                if (std::isnan(start)) start = 0.0;
                if (std::isnan(end)) end = 10.0;
                if (std::isnan(step) || step == 0.0) step = 1.0;
                if (std::isinf(start)) start = -100.0;
                if (std::isinf(end)) end = 100.0;
                if (std::isinf(step)) step = 1.0;
                
                // Ensure step has correct sign
                if ((end > start && step < 0) || (end < start && step > 0)) {
                    step = -step;
                }
                
                // Prevent too many elements
                double num_elements = std::abs((end - start) / step);
                if (num_elements > 100000) {
                    step = (end - start) / 1000.0;
                    if (step == 0.0) step = 1.0;
                }
                
                result = torch::arange(start, end, step, options);
                break;
            }
            case 3: {
                // Integer arange with int64_t
                int64_t start = consumeValue<int64_t>(Data, offset, Size) % 10000;
                int64_t end = consumeValue<int64_t>(Data, offset, Size) % 10000;
                int64_t step = consumeValue<int64_t>(Data, offset, Size) % 100;
                
                if (step == 0) step = 1;
                
                // Ensure step has correct sign
                if ((end > start && step < 0) || (end < start && step > 0)) {
                    step = -step;
                }
                
                result = torch::arange(start, end, step, options);
                break;
            }
            case 4: {
                // Test with negative ranges
                double start = -consumeValue<double>(Data, offset, Size);
                double end = consumeValue<double>(Data, offset, Size);
                
                start = std::fmod(start, 1000.0);
                end = std::fmod(end, 1000.0);
                
                if (std::isnan(start)) start = -10.0;
                if (std::isnan(end)) end = 10.0;
                
                result = torch::arange(start, end, options);
                break;
            }
            case 5: {
                // Test edge case: start == end
                double value = consumeValue<double>(Data, offset, Size);
                value = std::fmod(value, 1000.0);
                if (std::isnan(value)) value = 5.0;
                
                result = torch::arange(value, value, options);
                break;
            }
            case 6: {
                // Test with very small step
                double start = consumeValue<double>(Data, offset, Size);
                double end = consumeValue<double>(Data, offset, Size);
                double step_raw = consumeValue<double>(Data, offset, Size);
                
                start = std::fmod(start, 10.0);
                end = std::fmod(end, 10.0);
                
                if (std::isnan(start)) start = 0.0;
                if (std::isnan(end)) end = 1.0;
                
                // Create small step but not too small to avoid excessive elements
                double step = 0.001 + std::abs(std::fmod(step_raw, 0.1));
                
                if (end < start) step = -step;
                
                result = torch::arange(start, end, step, options);
                break;
            }
            case 7: {
                // Test scalar output (single element)
                result = torch::arange(0, 1, 1, options);
                
                // Additional operations on the result
                if (result.numel() > 0) {
                    // Test various operations that might trigger edge cases
                    uint8_t op = consumeValue<uint8_t>(Data, offset, Size) % 6;
                    switch(op) {
                        case 0:
                            result = result.sum();
                            break;
                        case 1:
                            result = result.mean();
                            break;
                        case 2:
                            if (result.numel() > 1) {
                                result = result.reshape({-1});
                            }
                            break;
                        case 3:
                            result = result.abs();
                            break;
                        case 4:
                            if (torch::isFloatingType(result.dtype())) {
                                result = result.sin();
                            }
                            break;
                        case 5:
                            result = result.clone();
                            break;
                    }
                }
                break;
            }
        }
        
        // Perform additional checks and operations on result
        if (result.defined()) {
            // Check basic properties
            auto numel = result.numel();
            auto dims = result.dim();
            auto dtype_result = result.dtype();
            
            // Try to access some elements if tensor is not empty
            if (numel > 0 && numel < 1000000) {
                // Test indexing
                if (numel > 1) {
                    auto first = result[0];
                    auto last = result[-1];
                }
                
                // Test conversion to different types if reasonable size
                if (numel < 10000) {
                    if (dtype_result != torch::kFloat32) {
                        auto float_result = result.to(torch::kFloat32);
                    }
                    if (dtype_result != torch::kInt64) {
                        auto int_result = result.to(torch::kInt64);
                    }
                }
                
                // Test basic arithmetic if floating point
                if (torch::isFloatingType(dtype_result) && numel < 10000) {
                    auto doubled = result * 2.0;
                    auto added = result + 1.0;
                }
            }
            
            // Test edge cases with empty or single-element results
            if (numel == 0) {
                // Empty tensor operations
                auto reshaped = result.reshape({0});
                auto cloned = result.clone();
            } else if (numel == 1) {
                // Single element operations
                auto item = result.item();
                auto squeezed = result.squeeze();
            }
        }
        
        return 0;
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    }
    catch (const std::bad_alloc &e)
    {
        // Memory allocation failures are expected with large ranges
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}