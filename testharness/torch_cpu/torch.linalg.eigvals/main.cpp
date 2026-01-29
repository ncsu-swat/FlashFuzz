#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;
        
        // Use first byte to determine test variant
        uint8_t test_mode = Data[0] % 4;
        offset = 1;
        
        // Create a tensor from fuzz data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // eigvals requires a square matrix (..., n, n)
        // Reshape to square matrix
        int64_t total_elements = input.numel();
        if (total_elements < 1) {
            return 0;
        }
        
        // Determine the size of the square matrix
        int64_t square_size = static_cast<int64_t>(std::sqrt(static_cast<double>(total_elements)));
        if (square_size < 1) {
            square_size = 1;
        }
        
        // Ensure we have enough elements
        int64_t needed_elements = square_size * square_size;
        if (total_elements < needed_elements) {
            square_size = static_cast<int64_t>(std::sqrt(static_cast<double>(total_elements)));
            if (square_size < 1) {
                square_size = 1;
            }
            needed_elements = square_size * square_size;
        }
        
        // Flatten and take only what we need
        torch::Tensor flat = input.reshape({-1});
        if (flat.size(0) < needed_elements) {
            // Pad with zeros if needed
            flat = torch::nn::functional::pad(flat, 
                torch::nn::functional::PadFuncOptions({0, needed_elements - flat.size(0)}));
        } else {
            flat = flat.slice(0, 0, needed_elements);
        }
        
        // Reshape to square matrix
        input = flat.reshape({square_size, square_size});
        
        // Convert to supported dtype (float, double, complex float, complex double)
        if (input.scalar_type() == torch::kBool || 
            input.scalar_type() == torch::kUInt8 || 
            input.scalar_type() == torch::kInt8 || 
            input.scalar_type() == torch::kInt16 || 
            input.scalar_type() == torch::kInt32 || 
            input.scalar_type() == torch::kInt64 ||
            input.scalar_type() == torch::kHalf ||
            input.scalar_type() == torch::kBFloat16) {
            input = input.to(torch::kFloat);
        }
        
        // Clamp values to avoid numerical instability
        if (!input.is_complex()) {
            input = torch::clamp(input, -1e6, 1e6);
        }
        
        torch::Tensor eigenvalues;
        
        switch (test_mode) {
            case 0: {
                // Test basic eigvals on square matrix
                // Use torch::linalg_eigvals (underscore notation for C++ API)
                eigenvalues = torch::linalg_eigvals(input);
                break;
            }
            case 1: {
                // Test with double precision
                torch::Tensor input_double = input.to(torch::kDouble);
                eigenvalues = torch::linalg_eigvals(input_double);
                break;
            }
            case 2: {
                // Test with complex float input
                torch::Tensor input_complex = input.to(torch::kComplexFloat);
                eigenvalues = torch::linalg_eigvals(input_complex);
                break;
            }
            case 3: {
                // Test with batch of matrices if we have enough elements
                if (square_size >= 2 && total_elements >= 2 * square_size * square_size) {
                    int64_t batch_size = 2;
                    int64_t mat_size = square_size / 2;
                    if (mat_size < 1) mat_size = 1;
                    
                    int64_t batch_needed = batch_size * mat_size * mat_size;
                    torch::Tensor batch_flat = input.reshape({-1});
                    
                    if (batch_flat.size(0) >= batch_needed) {
                        batch_flat = batch_flat.slice(0, 0, batch_needed);
                        torch::Tensor batch_input = batch_flat.reshape({batch_size, mat_size, mat_size});
                        batch_input = batch_input.to(torch::kFloat);
                        batch_input = torch::clamp(batch_input, -1e6, 1e6);
                        eigenvalues = torch::linalg_eigvals(batch_input);
                    } else {
                        eigenvalues = torch::linalg_eigvals(input);
                    }
                } else {
                    eigenvalues = torch::linalg_eigvals(input);
                }
                break;
            }
        }
        
        // Verify output is valid (complex tensor with eigenvalues)
        if (eigenvalues.defined()) {
            // Access data to ensure computation completed
            (void)eigenvalues.numel();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}