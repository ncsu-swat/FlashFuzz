#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>

// Helper to extract a scalar value from fuzzer data
template<typename T>
T extractScalar(const uint8_t* data, size_t& offset, size_t size) {
    T value = T(0);
    size_t bytes_needed = sizeof(T);
    if (offset + bytes_needed <= size) {
        std::memcpy(&value, data + offset, bytes_needed);
        offset += bytes_needed;
    }
    return value;
}

// Extract a float/double value with special handling for edge cases
double extractClampValue(const uint8_t* data, size_t& offset, size_t size) {
    if (offset >= size) return 0.0;
    
    uint8_t selector = data[offset++];
    
    // Use selector to choose interesting values
    switch (selector % 10) {
        case 0: return std::numeric_limits<double>::quiet_NaN();
        case 1: return std::numeric_limits<double>::infinity();
        case 2: return -std::numeric_limits<double>::infinity();
        case 3: return std::numeric_limits<double>::min();
        case 4: return std::numeric_limits<double>::max();
        case 5: return -std::numeric_limits<double>::max();
        case 6: return 0.0;
        case 7: return -0.0;
        default: {
            // Extract a regular double value
            return extractScalar<double>(data, offset, size);
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal data for tensor creation
        if (Size < 3) {
            return 0;
        }

        // Create primary tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract clamp minimum value
        double min_val = extractClampValue(Data, offset, Size);
        
        // Determine if we should use scalar or tensor version
        bool use_tensor_min = false;
        torch::Tensor min_tensor;
        
        if (offset < Size) {
            uint8_t mode = Data[offset++];
            use_tensor_min = (mode % 3 == 0); // ~33% chance to use tensor version
            
            if (use_tensor_min && offset < Size) {
                try {
                    // Create a tensor for the minimum value
                    min_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Handle broadcasting scenarios
                    if (offset < Size) {
                        uint8_t broadcast_mode = Data[offset++];
                        switch (broadcast_mode % 4) {
                            case 0:
                                // Make min_tensor a scalar tensor
                                if (min_tensor.numel() > 0) {
                                    min_tensor = min_tensor.flatten()[0];
                                }
                                break;
                            case 1:
                                // Try to reshape to broadcast-compatible shape
                                if (tensor.dim() > 0 && min_tensor.numel() > 0) {
                                    std::vector<int64_t> new_shape(tensor.dim(), 1);
                                    if (offset < Size && tensor.dim() > 0) {
                                        uint8_t dim_idx = Data[offset++] % tensor.dim();
                                        if (tensor.size(dim_idx) > 0) {
                                            new_shape[dim_idx] = std::min(min_tensor.numel(), 
                                                                         tensor.size(dim_idx));
                                        }
                                    }
                                    try {
                                        min_tensor = min_tensor.view(new_shape);
                                    } catch (...) {
                                        // If reshape fails, keep original
                                    }
                                }
                                break;
                            case 2:
                                // Use as-is for potential broadcast errors
                                break;
                            case 3:
                                // Try to match dtype
                                if (min_tensor.dtype() != tensor.dtype()) {
                                    try {
                                        min_tensor = min_tensor.to(tensor.dtype());
                                    } catch (...) {
                                        // Keep original if conversion fails
                                    }
                                }
                                break;
                        }
                    }
                } catch (...) {
                    // Fall back to scalar version if tensor creation fails
                    use_tensor_min = false;
                }
            }
        }
        
        // Store original for validation if needed
        torch::Tensor original;
        bool validate = false;
        if (offset < Size) {
            validate = (Data[offset++] % 4 == 0); // 25% chance to validate
            if (validate) {
                original = tensor.clone();
            }
        }
        
        // Apply clamp_min_ operation
        if (use_tensor_min && min_tensor.defined()) {
            // Tensor version of clamp_min_
            tensor.clamp_min_(min_tensor);
        } else {
            // Scalar version of clamp_min_
            tensor.clamp_min_(min_val);
        }
        
        // Validate the operation if requested
        if (validate && original.defined()) {
            // Check that all values are >= min_val (or min_tensor values)
            if (use_tensor_min && min_tensor.defined()) {
                // For tensor min, we need to handle broadcasting
                try {
                    auto comparison = tensor >= min_tensor;
                    if (!comparison.all().item<bool>()) {
                        // This shouldn't happen unless there's a bug
                        std::cerr << "Validation failed: Some values less than min_tensor" << std::endl;
                    }
                } catch (...) {
                    // Broadcasting might fail, which is ok for fuzzing
                }
            } else {
                // For scalar min, check all values
                if (!std::isnan(min_val)) {
                    auto comparison = tensor >= min_val;
                    if (!comparison.all().item<bool>()) {
                        // This shouldn't happen unless there's a bug
                        std::cerr << "Validation failed: Some values less than " << min_val << std::endl;
                    }
                }
            }
            
            // Check that values >= min were not changed
            if (!use_tensor_min || !min_tensor.defined()) {
                auto unchanged_mask = original >= min_val;
                auto changed_values = torch::where(unchanged_mask, original - tensor, 
                                                  torch::zeros_like(tensor));
                if (changed_values.abs().max().item<double>() > 1e-6) {
                    std::cerr << "Values that should be unchanged were modified" << std::endl;
                }
            }
        }
        
        // Additional operations to increase coverage
        if (offset < Size) {
            uint8_t extra_ops = Data[offset++];
            
            if (extra_ops % 4 == 0) {
                // Chain another clamp_min_ with different value
                double second_min = extractClampValue(Data, offset, Size);
                tensor.clamp_min_(second_min);
            } else if (extra_ops % 4 == 1) {
                // Test on different views
                if (tensor.numel() > 1) {
                    auto view = tensor.view(-1);
                    view.clamp_min_(min_val);
                }
            } else if (extra_ops % 4 == 2) {
                // Test on transposed tensor
                if (tensor.dim() >= 2) {
                    auto transposed = tensor.transpose(0, 1);
                    transposed.clamp_min_(min_val);
                }
            } else {
                // Test on slice
                if (tensor.dim() > 0 && tensor.size(0) > 1) {
                    auto slice = tensor.narrow(0, 0, 1);
                    slice.clamp_min_(min_val);
                }
            }
        }
        
        // Force computation to complete
        if (tensor.is_cuda()) {
            torch::cuda::synchronize();
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch errors are expected during fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}