#include "fuzzer_utils.h"
#include <iostream>

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
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor - use float type for special functions
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        input = input.to(torch::kFloat32);
        
        // Apply various torch.special functions based on remaining data
        if (offset < Size) {
            uint8_t special_op_selector = Data[offset++] % 18;
            
            try {
                switch (special_op_selector) {
                    case 0:
                        torch::special::entr(input);
                        break;
                    case 1:
                        torch::special::erf(input);
                        break;
                    case 2:
                        torch::special::erfc(input);
                        break;
                    case 3:
                        torch::special::erfinv(input);
                        break;
                    case 4:
                        torch::special::expit(input);
                        break;
                    case 5:
                        torch::special::expm1(input);
                        break;
                    case 6:
                        torch::special::exp2(input);
                        break;
                    case 7:
                        torch::special::gammaln(input);
                        break;
                    case 8:
                        torch::special::digamma(input);
                        break;
                    case 9:
                        torch::special::psi(input);
                        break;
                    case 10:
                        torch::special::log1p(input);
                        break;
                    case 11:
                        torch::special::logit(input);
                        break;
                    case 12:
                        torch::special::i0(input);
                        break;
                    case 13:
                        torch::special::i0e(input);
                        break;
                    case 14:
                        torch::special::i1(input);
                        break;
                    case 15:
                        torch::special::i1e(input);
                        break;
                    case 16:
                        torch::special::ndtri(input);
                        break;
                    case 17:
                        torch::special::ndtr(input);
                        break;
                }
            } catch (const std::exception& e) {
                // Silently catch expected failures from single-param operations
            }
            
            // Try some multi-parameter special functions if we have more data
            if (offset + 1 < Size) {
                uint8_t multi_param_selector = Data[offset++] % 6;
                torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);
                other = other.to(torch::kFloat32);
                
                try {
                    switch (multi_param_selector) {
                        case 0:
                            torch::special::xlogy(input, other);
                            break;
                        case 1:
                            torch::special::xlog1py(input, other);
                            break;
                        case 2:
                            // multigammaln requires input > (p-1)/2 where p is the second param
                            torch::special::multigammaln(input.abs() + 1.0, 2);
                            break;
                        case 3:
                            // polygamma(n, input) where n >= 0
                            torch::special::polygamma(1, input);
                            break;
                        case 4:
                            torch::special::polygamma(0, input);
                            break;
                        case 5:
                            // zeta requires careful input handling
                            torch::special::zeta(input.abs() + 1.1, other.abs() + 0.1);
                            break;
                    }
                } catch (const std::exception& e) {
                    // Silently catch expected failures from multi-parameter operations
                }
            }
            
            // Try scalar versions of some functions
            if (offset < Size) {
                uint8_t scalar_selector = Data[offset++] % 4;
                float scalar_val = static_cast<float>(Data[offset % Size]) / 255.0f;
                
                try {
                    switch (scalar_selector) {
                        case 0:
                            torch::special::xlogy(input, scalar_val + 0.01f);
                            break;
                        case 1:
                            torch::special::xlog1py(input, scalar_val);
                            break;
                        case 2:
                            torch::special::xlogy(scalar_val, input.abs() + 0.01f);
                            break;
                        case 3:
                            torch::special::xlog1py(scalar_val, input.abs());
                            break;
                    }
                } catch (const std::exception& e) {
                    // Silently catch expected failures from scalar operations
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