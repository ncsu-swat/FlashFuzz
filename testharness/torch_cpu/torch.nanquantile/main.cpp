#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float for quantile operations
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Inject some NaN values to test nanquantile behavior
        if (offset < Size && input.numel() > 0) {
            uint8_t nan_control = Data[offset++];
            if (nan_control % 3 == 0) {
                // Inject NaN at random positions
                auto flat = input.flatten();
                int64_t nan_count = std::min((int64_t)(nan_control % 5 + 1), flat.numel());
                for (int64_t i = 0; i < nan_count && i < flat.numel(); i++) {
                    flat[i] = std::numeric_limits<float>::quiet_NaN();
                }
                input = flat.view(input.sizes());
            }
        }
        
        // Parse q (quantile value between 0 and 1)
        double q = 0.5;
        if (offset + sizeof(double) <= Size) {
            double raw_q;
            std::memcpy(&raw_q, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Handle special float values
            if (std::isnan(raw_q) || std::isinf(raw_q)) {
                q = 0.5;
            } else {
                // Ensure q is between 0 and 1
                q = std::abs(raw_q);
                q = q - std::floor(q);
            }
        }
        
        // Parse dim (optional)
        c10::optional<int64_t> dim = c10::nullopt;
        bool use_dim = false;
        int64_t dim_value = 0;
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_dim;
            std::memcpy(&raw_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Decide whether to use dim or not
            use_dim = (raw_dim & 0x1) != 0;
            
            if (use_dim && input.dim() > 0) {
                dim_value = (raw_dim >> 1) % input.dim();
                if (dim_value < 0) {
                    dim_value += input.dim();
                }
                dim = dim_value;
            }
        }
        
        // Parse keepdim
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Parse interpolation
        std::string interpolation = "linear";
        if (offset < Size) {
            uint8_t interp_selector = Data[offset++] % 5;
            switch (interp_selector) {
                case 0: interpolation = "linear"; break;
                case 1: interpolation = "lower"; break;
                case 2: interpolation = "higher"; break;
                case 3: interpolation = "midpoint"; break;
                case 4: interpolation = "nearest"; break;
            }
        }
        
        // Try different variants of nanquantile
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 4;
            
            try {
                switch (variant) {
                    case 0: {
                        // Basic nanquantile with just q (no dim)
                        torch::Tensor result = torch::nanquantile(input, q);
                        break;
                    }
                    case 1: {
                        // nanquantile with q and dim
                        if (dim.has_value()) {
                            torch::Tensor result = torch::nanquantile(input, q, dim.value(), keepdim);
                        } else {
                            torch::Tensor result = torch::nanquantile(input, q);
                        }
                        break;
                    }
                    case 2: {
                        // nanquantile with q, dim, and keepdim
                        if (dim.has_value()) {
                            torch::Tensor result = torch::nanquantile(input, q, dim.value(), keepdim);
                        } else {
                            torch::Tensor result = torch::nanquantile(input, q);
                        }
                        break;
                    }
                    case 3: {
                        // Full nanquantile with interpolation
                        if (dim.has_value()) {
                            torch::Tensor result = torch::nanquantile(input, q, dim.value(), keepdim, interpolation);
                        } else {
                            // When no dim, use dim=0 with flattened input for interpolation test
                            auto flat_input = input.flatten();
                            if (flat_input.numel() > 0) {
                                torch::Tensor result = torch::nanquantile(flat_input, q, 0, keepdim, interpolation);
                            }
                        }
                        break;
                    }
                }
            } catch (const c10::Error&) {
                // Expected errors for invalid inputs - silently ignore
            } catch (const std::runtime_error&) {
                // Expected errors for edge cases - silently ignore
            }
        } else {
            // Default case if we don't have enough data
            try {
                torch::Tensor result = torch::nanquantile(input, q);
            } catch (const c10::Error&) {
                // Silently ignore expected errors
            }
        }
        
        // Try with q as a tensor
        if (offset + 1 < Size) {
            // Create a tensor with q values
            std::vector<double> q_values;
            uint8_t num_q = Data[offset++] % 5 + 1; // 1 to 5 q values
            
            for (uint8_t i = 0; i < num_q && offset + sizeof(double) <= Size; i++) {
                double raw_q;
                std::memcpy(&raw_q, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                // Handle special float values and ensure q is between 0 and 1
                if (std::isnan(raw_q) || std::isinf(raw_q)) {
                    q_values.push_back(0.5);
                } else {
                    double q_val = std::abs(raw_q);
                    q_val = q_val - std::floor(q_val);
                    q_values.push_back(q_val);
                }
            }
            
            if (!q_values.empty()) {
                torch::Tensor q_tensor = torch::tensor(q_values, torch::kFloat64);
                
                // Try different variants with q as tensor
                if (offset < Size) {
                    uint8_t tensor_variant = Data[offset++] % 4;
                    
                    try {
                        switch (tensor_variant) {
                            case 0: {
                                // Basic nanquantile with q tensor
                                torch::Tensor result = torch::nanquantile(input, q_tensor);
                                break;
                            }
                            case 1: {
                                // nanquantile with q tensor and dim
                                if (dim.has_value()) {
                                    torch::Tensor result = torch::nanquantile(input, q_tensor, dim.value(), keepdim);
                                } else {
                                    torch::Tensor result = torch::nanquantile(input, q_tensor);
                                }
                                break;
                            }
                            case 2: {
                                // nanquantile with q tensor, dim, and keepdim
                                if (dim.has_value()) {
                                    torch::Tensor result = torch::nanquantile(input, q_tensor, dim.value(), keepdim);
                                } else {
                                    torch::Tensor result = torch::nanquantile(input, q_tensor);
                                }
                                break;
                            }
                            case 3: {
                                // Full nanquantile with all parameters and q tensor
                                if (dim.has_value()) {
                                    torch::Tensor result = torch::nanquantile(input, q_tensor, dim.value(), keepdim, interpolation);
                                } else {
                                    auto flat_input = input.flatten();
                                    if (flat_input.numel() > 0) {
                                        torch::Tensor result = torch::nanquantile(flat_input, q_tensor, 0, keepdim, interpolation);
                                    }
                                }
                                break;
                            }
                        }
                    } catch (const c10::Error&) {
                        // Expected errors for invalid inputs - silently ignore
                    } catch (const std::runtime_error&) {
                        // Expected errors for edge cases - silently ignore
                    }
                } else {
                    // Default case with q tensor
                    try {
                        torch::Tensor result = torch::nanquantile(input, q_tensor);
                    } catch (const c10::Error&) {
                        // Silently ignore expected errors
                    }
                }
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