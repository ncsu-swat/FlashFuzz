#include "fuzzer_utils.h"
#include <iostream>
#include <limits>
#include <cstring>

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - convert to float for linalg operations
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kFloat);
        
        // Ensure tensor is not empty
        if (input.numel() == 0) {
            return 0;
        }
        
        torch::Tensor result;
        
        // Parse which variant to test
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 6;
        }
        
        // Parse keepdim parameter
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 1;
        }
        
        switch (variant) {
            case 0: {
                // Default norm (Frobenius for matrices, 2-norm for vectors)
                result = torch::linalg_norm(input);
                break;
            }
            case 1: {
                // Norm with scalar ord
                if (offset + sizeof(float) <= Size) {
                    float ord_value;
                    std::memcpy(&ord_value, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    // Clamp to reasonable values to avoid NaN/Inf issues
                    if (std::isfinite(ord_value) && ord_value != 0.0f) {
                        ord_value = std::fmod(ord_value, 10.0f);
                        if (ord_value == 0.0f) ord_value = 2.0f;
                        try {
                            result = torch::linalg_norm(input, ord_value);
                        } catch (...) {
                            // Some ord values may not be valid for certain inputs
                        }
                    }
                }
                break;
            }
            case 2: {
                // Norm with string ord ("fro" or "nuc")
                uint8_t str_ord = 0;
                if (offset < Size) {
                    str_ord = Data[offset++] % 2;
                }
                try {
                    if (str_ord == 0) {
                        result = torch::linalg_norm(input, "fro");
                    } else {
                        // Nuclear norm requires 2D input
                        if (input.dim() >= 2) {
                            result = torch::linalg_norm(input, "nuc");
                        }
                    }
                } catch (...) {
                    // String norms have specific requirements
                }
                break;
            }
            case 3: {
                // Norm with dim parameter
                if (input.dim() > 0 && offset < Size) {
                    int8_t dim_val = static_cast<int8_t>(Data[offset++]);
                    int64_t dim = dim_val % input.dim();
                    try {
                        result = torch::linalg_norm(input, 2.0, dim, keepdim);
                    } catch (...) {
                        // Dimension may be invalid
                    }
                }
                break;
            }
            case 4: {
                // Vector norm with specific ord values
                if (offset < Size) {
                    uint8_t ord_type = Data[offset++] % 4;
                    try {
                        switch (ord_type) {
                            case 0:
                                result = torch::linalg_norm(input, 1.0);
                                break;
                            case 1:
                                result = torch::linalg_norm(input, 2.0);
                                break;
                            case 2:
                                result = torch::linalg_norm(input, std::numeric_limits<double>::infinity());
                                break;
                            case 3:
                                result = torch::linalg_norm(input, -std::numeric_limits<double>::infinity());
                                break;
                        }
                    } catch (...) {
                        // Some combinations may fail
                    }
                }
                break;
            }
            case 5: {
                // Matrix norm with dim tuple (requires 2D or more)
                if (input.dim() >= 2 && offset + 1 < Size) {
                    int8_t dim0_val = static_cast<int8_t>(Data[offset++]);
                    int8_t dim1_val = static_cast<int8_t>(Data[offset++]);
                    int64_t dim0 = dim0_val % input.dim();
                    int64_t dim1 = dim1_val % input.dim();
                    if (dim0 != dim1) {
                        std::vector<int64_t> dims = {dim0, dim1};
                        try {
                            result = torch::linalg_norm(input, 2.0, dims, keepdim);
                        } catch (...) {
                            // Invalid dim combinations
                        }
                    }
                }
                break;
            }
        }
        
        // Additional test: linalg_vector_norm
        if (offset < Size && (Data[offset++] & 1)) {
            try {
                // Flatten and compute vector norm
                torch::Tensor flat = input.flatten();
                result = torch::linalg_vector_norm(flat);
            } catch (...) {
                // May fail for certain inputs
            }
        }
        
        // Additional test: linalg_matrix_norm if input is 2D+
        if (input.dim() >= 2 && offset < Size && (Data[offset++] & 1)) {
            try {
                result = torch::linalg_matrix_norm(input);
            } catch (...) {
                // May fail for certain inputs
            }
        }
        
        // Additional test: standard torch::norm API
        if (offset < Size && (Data[offset++] & 1)) {
            try {
                // torch::norm with optional scalar p value
                result = torch::norm(input);
                
                if (input.dim() > 0 && offset < Size) {
                    int8_t dim_val = static_cast<int8_t>(Data[offset++]);
                    int64_t dim = dim_val % input.dim();
                    result = torch::norm(input, 2.0, dim, keepdim);
                }
            } catch (...) {
                // May fail for certain inputs
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