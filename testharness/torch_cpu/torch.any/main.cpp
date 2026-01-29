#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a dimension value from the data if available
        int64_t dim = 0;
        bool has_dim = false;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            has_dim = true;
        }
        
        // Extract a keepdim boolean from the data if available
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Extract test case selector
        uint8_t test_case = 0;
        if (offset < Size) {
            test_case = Data[offset++] % 4;
        }
        
        torch::Tensor result;
        
        switch (test_case) {
            case 0: {
                // Test case 1: torch.any without arguments - returns scalar bool
                result = torch::any(input_tensor);
                break;
            }
            
            case 1: {
                // Test case 2: torch.any with dimension and keepdim
                if (input_tensor.dim() > 0) {
                    // Normalize dim to valid range [-ndim, ndim-1]
                    int64_t ndim = input_tensor.dim();
                    int64_t normalized_dim = ((dim % ndim) + ndim) % ndim;
                    
                    try {
                        result = torch::any(input_tensor, normalized_dim, keepdim);
                    } catch (...) {
                        // Shape-related errors are expected for some inputs
                    }
                } else {
                    // For 0-dim tensors, call without dim
                    result = torch::any(input_tensor);
                }
                break;
            }
            
            case 2: {
                // Test case 3: torch.any with dimension only (keepdim defaults to false)
                if (input_tensor.dim() > 0) {
                    int64_t ndim = input_tensor.dim();
                    int64_t normalized_dim = ((dim % ndim) + ndim) % ndim;
                    
                    try {
                        result = torch::any(input_tensor, normalized_dim);
                    } catch (...) {
                        // Shape-related errors are expected
                    }
                } else {
                    result = torch::any(input_tensor);
                }
                break;
            }
            
            case 3: {
                // Test case 4: torch.any_out with out parameter
                if (input_tensor.dim() > 0) {
                    int64_t ndim = input_tensor.dim();
                    int64_t normalized_dim = ((dim % ndim) + ndim) % ndim;
                    
                    try {
                        // Calculate output shape
                        std::vector<int64_t> out_shape;
                        for (int64_t i = 0; i < ndim; i++) {
                            if (i == normalized_dim) {
                                if (keepdim) {
                                    out_shape.push_back(1);
                                }
                            } else {
                                out_shape.push_back(input_tensor.size(i));
                            }
                        }
                        
                        torch::Tensor out = torch::empty(out_shape, torch::kBool);
                        result = torch::any_out(out, input_tensor, normalized_dim, keepdim);
                    } catch (...) {
                        // Shape mismatch or other errors are expected
                    }
                } else {
                    // For scalar output
                    try {
                        torch::Tensor out = torch::empty({}, torch::kBool);
                        // For 0-dim tensor, we need to use different approach
                        result = torch::any(input_tensor);
                    } catch (...) {
                        // Expected for some edge cases
                    }
                }
                break;
            }
        }
        
        // Additional coverage: test with different tensor types
        // Convert to bool tensor and test (any works on all types but bool is canonical)
        try {
            torch::Tensor bool_tensor = input_tensor.to(torch::kBool);
            torch::Tensor bool_result = torch::any(bool_tensor);
            (void)bool_result;
        } catch (...) {
            // Conversion may fail for some dtypes
        }
        
        // Test with integer tensor
        try {
            torch::Tensor int_tensor = input_tensor.to(torch::kInt);
            torch::Tensor int_result = torch::any(int_tensor);
            (void)int_result;
        } catch (...) {
            // Conversion may fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;  // Keep the input
}