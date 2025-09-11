#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply various torch.special functions based on remaining data
        if (offset < Size) {
            uint8_t special_op_selector = Data[offset++] % 20;
            
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
                        torch::special::logsumexp(input, {0}, false);
                        break;
                    case 13:
                        torch::special::log_softmax(input, 0, torch::nullopt);
                        break;
                    case 14:
                        torch::special::softmax(input, 0, torch::nullopt);
                        break;
                    case 15:
                        if (offset + 1 < Size) {
                            torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);
                            torch::special::xlog1py(input, other);
                        } else {
                            torch::special::xlog1py(input, input);
                        }
                        break;
                    case 16:
                        torch::special::i0(input);
                        break;
                    case 17:
                        torch::special::i0e(input);
                        break;
                    case 18:
                        torch::special::i1(input);
                        break;
                    case 19:
                        torch::special::i1e(input);
                        break;
                }
            } catch (const std::exception& e) {
                // Catch exceptions from the special operations but continue fuzzing
            }
            
            // Try some multi-parameter special functions if we have more data
            if (offset + 1 < Size) {
                uint8_t multi_param_selector = Data[offset++] % 5;
                torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);
                
                try {
                    switch (multi_param_selector) {
                        case 0:
                            torch::special::zeta(input, other);
                            break;
                        case 1:
                            torch::special::xlogy(input, other);
                            break;
                        case 2:
                            torch::special::xlog1py(input, other);
                            break;
                        case 3:
                            torch::special::multigammaln(input, 2);
                            break;
                        case 4:
                            torch::special::polygamma(1, input);
                            break;
                    }
                } catch (const std::exception& e) {
                    // Catch exceptions from multi-parameter operations but continue fuzzing
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
