#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::abs, std::isfinite

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least space for 3 doubles + 1 byte for dtype
        if (Size < 3 * sizeof(double) + 1) {
            return 0;
        }
        
        // Extract start, end, and step values from the input data
        double start = 0.0, end = 0.0, step = 1.0;
        
        std::memcpy(&start, Data + offset, sizeof(double));
        offset += sizeof(double);
        
        std::memcpy(&end, Data + offset, sizeof(double));
        offset += sizeof(double);
        
        std::memcpy(&step, Data + offset, sizeof(double));
        offset += sizeof(double);
        
        // Sanitize inputs to avoid problematic values
        if (!std::isfinite(start) || !std::isfinite(end) || !std::isfinite(step)) {
            return 0;
        }
        
        // Avoid zero step which would cause infinite loop
        if (step == 0.0) {
            step = 1.0;
        }
        
        // Constrain values to reasonable range to avoid OOM
        const double MAX_VAL = 1e6;
        start = std::max(-MAX_VAL, std::min(MAX_VAL, start));
        end = std::max(-MAX_VAL, std::min(MAX_VAL, end));
        
        // Ensure step magnitude is reasonable to avoid huge tensors
        const double MIN_STEP = 0.001;
        if (std::abs(step) < MIN_STEP) {
            step = (step > 0) ? MIN_STEP : -MIN_STEP;
        }
        
        // Limit maximum number of elements to prevent OOM
        const int64_t MAX_ELEMENTS = 100000;
        double num_elements = std::abs((end - start) / step) + 1;
        if (num_elements > MAX_ELEMENTS) {
            // Adjust step to limit elements
            step = (end - start) / (MAX_ELEMENTS - 1);
            if (step == 0.0) step = 1.0;
        }
        
        // Get dtype for the range operation
        torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
        
        // Get device for the range operation
        torch::Device device = torch::kCPU;
        
        // Variant 1: Basic range with default step=1
        try {
            auto result1 = torch::range(start, end, torch::TensorOptions().dtype(dtype).device(device));
            (void)result1;
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
        
        // Variant 2: Range with custom step
        try {
            auto result2 = torch::range(start, end, step, torch::TensorOptions().dtype(dtype).device(device));
            (void)result2;
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
        
        // Variant 3: Range with different dtypes
        try {
            auto result3 = torch::range(start, end, step, torch::TensorOptions().dtype(torch::kInt64).device(device));
            (void)result3;
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
        
        try {
            auto result4 = torch::range(start, end, step, torch::TensorOptions().dtype(torch::kDouble).device(device));
            (void)result4;
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
        
        try {
            auto result5 = torch::range(start, end, step, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            (void)result5;
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
        
        // Variant 4: Range with negative step (reversed direction)
        try {
            if (end > start && step > 0) {
                auto result6 = torch::range(end, start, -step, torch::TensorOptions().dtype(dtype).device(device));
                (void)result6;
            } else if (start > end && step < 0) {
                auto result6 = torch::range(start, end, step, torch::TensorOptions().dtype(dtype).device(device));
                (void)result6;
            }
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
        
        // Variant 5: Small range edge case
        try {
            auto result7 = torch::range(start, start, 1.0, torch::TensorOptions().dtype(dtype).device(device));
            (void)result7;
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
        
        // Variant 6: Use parsed values with different options
        if (offset < Size) {
            bool requires_grad = (Data[offset] % 2 == 0) && (dtype == torch::kFloat32 || dtype == torch::kFloat64);
            try {
                auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device).requires_grad(requires_grad);
                auto result8 = torch::range(start, end, step, opts);
                (void)result8;
            } catch (const std::exception&) {
                // Allow exceptions from the API
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}