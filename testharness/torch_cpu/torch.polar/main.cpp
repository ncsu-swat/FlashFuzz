#include "fuzzer_utils.h"
#include <iostream>

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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create two tensors for abs and angle inputs to polar
        torch::Tensor abs_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            abs_tensor = torch::abs(abs_tensor.to(torch::kFloat));
            torch::Tensor angle_tensor = torch::zeros_like(abs_tensor);
            torch::Tensor result = torch::polar(abs_tensor, angle_tensor);
            return 0;
        }
        
        torch::Tensor angle_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // polar requires float or double inputs, convert to float
        abs_tensor = abs_tensor.to(torch::kFloat);
        angle_tensor = angle_tensor.to(torch::kFloat);
        
        // Make sure abs is non-negative as required by polar
        abs_tensor = torch::abs(abs_tensor);
        
        // 1. Basic polar call with matching shapes
        try {
            // Ensure same shape for basic call
            torch::Tensor flat_abs = abs_tensor.reshape({-1});
            torch::Tensor flat_angle = angle_tensor.reshape({-1});
            
            if (flat_abs.numel() > 0 && flat_angle.numel() > 0) {
                int64_t min_size = std::min(flat_abs.size(0), flat_angle.size(0));
                torch::Tensor result1 = torch::polar(
                    flat_abs.slice(0, 0, min_size),
                    flat_angle.slice(0, 0, min_size)
                );
            }
        } catch (const std::exception &) {
            // Shape mismatch or other expected errors
        }
        
        // 2. Try with scalar tensors for broadcasting
        if (Size > offset) {
            uint8_t selector = Data[offset++] % 3;
            try {
                if (selector == 0 && abs_tensor.numel() > 0) {
                    // Use scalar for abs
                    torch::Tensor scalar_abs = torch::tensor(std::abs(abs_tensor.flatten()[0].item<float>()));
                    torch::Tensor flat_angle = angle_tensor.reshape({-1});
                    if (flat_angle.numel() > 0) {
                        torch::Tensor result2 = torch::polar(scalar_abs, flat_angle);
                    }
                } else if (selector == 1 && angle_tensor.numel() > 0) {
                    // Use scalar for angle
                    torch::Tensor scalar_angle = torch::tensor(angle_tensor.flatten()[0].item<float>());
                    torch::Tensor flat_abs = torch::abs(abs_tensor.reshape({-1}));
                    if (flat_abs.numel() > 0) {
                        torch::Tensor result2 = torch::polar(flat_abs, scalar_angle);
                    }
                } else {
                    // Try with reshaped tensors of matching size
                    torch::Tensor reshaped_abs = torch::abs(abs_tensor.reshape({-1}));
                    torch::Tensor reshaped_angle = angle_tensor.reshape({-1});
                    if (reshaped_abs.numel() > 0 && reshaped_angle.numel() > 0) {
                        int64_t min_size = std::min(reshaped_abs.size(0), reshaped_angle.size(0));
                        torch::Tensor result2 = torch::polar(
                            reshaped_abs.slice(0, 0, min_size),
                            reshaped_angle.slice(0, 0, min_size)
                        );
                    }
                }
            } catch (const std::exception &) {
                // Expected failures
            }
        }
        
        // 3. Try with out variant
        if (Size > offset && abs_tensor.numel() > 0 && angle_tensor.numel() > 0) {
            try {
                torch::Tensor flat_abs = torch::abs(abs_tensor.reshape({-1}));
                torch::Tensor flat_angle = angle_tensor.reshape({-1});
                int64_t min_size = std::min(flat_abs.size(0), flat_angle.size(0));
                
                torch::Tensor abs_slice = flat_abs.slice(0, 0, min_size);
                torch::Tensor angle_slice = flat_angle.slice(0, 0, min_size);
                
                // Output must be complex type
                torch::Tensor out_tensor = torch::empty({min_size}, torch::kComplexFloat);
                torch::polar_out(out_tensor, abs_slice, angle_slice);
            } catch (const std::exception &) {
                // Expected failures
            }
        }
        
        // 4. Try with different dtypes (float vs double)
        if (Size > offset) {
            uint8_t dtype_selector = Data[offset++] % 2;
            try {
                torch::ScalarType dtype = (dtype_selector == 0) ? torch::kFloat : torch::kDouble;
                
                torch::Tensor abs_converted = torch::abs(abs_tensor.to(dtype).reshape({-1}));
                torch::Tensor angle_converted = angle_tensor.to(dtype).reshape({-1});
                
                if (abs_converted.numel() > 0 && angle_converted.numel() > 0) {
                    int64_t min_size = std::min(abs_converted.size(0), angle_converted.size(0));
                    torch::Tensor result3 = torch::polar(
                        abs_converted.slice(0, 0, min_size),
                        angle_converted.slice(0, 0, min_size)
                    );
                }
            } catch (const std::exception &) {
                // Expected failures
            }
        }
        
        // 5. Try with empty tensors
        if (Size > offset) {
            try {
                torch::Tensor empty_abs = torch::empty({0}, torch::kFloat);
                torch::Tensor empty_angle = torch::empty({0}, torch::kFloat);
                torch::Tensor result4 = torch::polar(empty_abs, empty_angle);
            } catch (const std::exception &) {
                // Expected failures
            }
        }
        
        // 6. Try with special values (zeros, large values)
        if (Size > offset) {
            try {
                torch::Tensor zero_abs = torch::zeros({5}, torch::kFloat);
                torch::Tensor pi_angle = torch::full({5}, 3.14159f, torch::kFloat);
                torch::Tensor result5 = torch::polar(zero_abs, pi_angle);
                
                torch::Tensor large_abs = torch::full({5}, 1e10f, torch::kFloat);
                torch::Tensor result6 = torch::polar(large_abs, pi_angle);
            } catch (const std::exception &) {
                // Expected failures
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