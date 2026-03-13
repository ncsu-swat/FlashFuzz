#include "fuzzer_utils.h"
#include <iostream>
#include <vector>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    try {
        if (Size < 3) return 0; // Need at least bytes for tensor metadata + control flags

        size_t offset = 0;

        // 1. Create Input Tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // 2. Parse Control Flags for torch::sum arguments
        // We need flags for: dim selection, keepdim, and dtype selection.

        // Flag 1: Dimension selection strategy
        uint8_t dim_strategy = 0;
        if (offset < Size) {
            dim_strategy = Data[offset++];
        }

        // Flag 2: Keepdim boolean
        bool keepdim = false;
        if (offset < Size) {
            keepdim = (Data[offset++] % 2) == 1;
        }

        // Flag 3: Dtype selection for the result
        // 0: None (default), 1: Same as input, 2: Specific types (Int64, Float)
        torch::Dtype dtype_flag = torch::kFloat; // Default init
        bool has_dtype = false;
        if (offset < Size) {
            uint8_t dtype_sel = Data[offset++] % 4;
            has_dtype = true;
            switch (dtype_sel) {
                case 0: has_dtype = false; break; // Default (None)
                case 1: dtype_flag = input_tensor.dtype().toScalarType(); break; // Same as input
                case 2: dtype_flag = torch::kInt64; break; // Promote to Int64
                case 3: dtype_flag = torch::kFloat; break; // Promote to Float
            }
        }

        // 3. Determine Dimensions to Reduce
        // We use the strategy byte to decide how to pick 'dim'
        torch::optional<std::vector<int64_t>> dims_to_reduce;
        int64_t rank = input_tensor.dim();

        if (rank > 0) {
            // Strategy logic:
            // 0-49: Reduce all (std::nullopt equivalent to no dim arg)
            // 50-149: Reduce single valid dimension
            // 150-199: Reduce multiple dimensions (tuple)
            // 200-255: Reduce all explicitly

            if (dim_strategy < 50) {
                // Reduce all dimensions (call sum(tensor))
                dims_to_reduce = torch::optional<std::vector<int64_t>>(); // empty optional
            } else if (dim_strategy < 150) {
                // Single dimension
                int64_t dim_idx = 0;
                if (rank > 0) {
                     // Use another byte if available to pick the dim, else default to 0
                    uint8_t dim_byte = (offset < Size) ? Data[offset++] : 0;
                    dim_idx = static_cast<int64_t>(dim_byte % rank);
                }
                dims_to_reduce = std::vector<int64_t>{dim_idx};
            } else if (dim_strategy < 200) {
                // Multiple dimensions (tuple)
                std::vector<int64_t> dims;
                if (rank > 1) {
                    // Pick 2 distinct dimensions
                    uint8_t dim_byte = (offset < Size) ? Data[offset++] : 0;
                    int64_t d1 = static_cast<int64_t>(dim_byte % rank);
                    dims.push_back(d1);
                    
                    // Try to pick a second different dimension
                    if (rank > 1) {
                        uint8_t dim_byte2 = (offset < Size) ? Data[offset++] : 1;
                        int64_t d2 = static_cast<int64_t>(dim_byte2 % rank);
                        if (d2 != d1) dims.push_back(d2);
                    }
                }
                // If we couldn't populate multiple dims, fallback to single or all
                if (dims.empty() && rank > 0) dims.push_back(0);
                dims_to_reduce = dims;
            } else {
                // Explicitly reduce all via empty vector logic or just let API handle it
                // We will treat this as "reduce all"
                 dims_to_reduce = torch::optional<std::vector<int64_t>>();
            }
        } else {
            // Rank 0 tensor (scalar)
            // Sum on scalar returns scalar. dim argument is invalid or redundant.
            dims_to_reduce = torch::optional<std::vector<int64_t>>();
        }

        // 4. Invoke torch::sum
        torch::Tensor result;

        // Construct arguments
        c10::optional<c10::ScalarType> dtype_opt = c10::nullopt;
        if (has_dtype) {
            dtype_opt = dtype_flag;
        }

        if (!dims_to_reduce.has_value() || dims_to_reduce.value().empty()) {
            // Case: torch.sum(input) or reduce over all dims
            if (has_dtype) {
                result = torch::sum(input_tensor, dtype_opt.value());
            } else {
                result = torch::sum(input_tensor);
            }
        } else {
            // Case: torch.sum(input, dim, keepdim, dtype)
            result = torch::sum(input_tensor, dims_to_reduce.value(), keepdim, dtype_opt);
        }

        // 5. Basic Validation (keep harness running)
        // Check if result is defined (basic sanity)
        if (!result.defined()) {
            // Should not happen for valid inputs, but good to check
            std::cerr << "Warning: torch::sum returned undefined tensor." << std::endl;
        }

    } catch (const c10::Error& e) {
        // Catch PyTorch errors specifically to avoid catching unrelated system errors
        // We log but return 0 to allow libFuzzer to continue exploring other paths
        // std::cout << "C10 Error caught: " << e.what_without_backtrace() << std::endl;
        return 0;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        // std::cout << "Std exception caught: " << e.what() << std::endl;
        return 0;
    }

    return 0;
}