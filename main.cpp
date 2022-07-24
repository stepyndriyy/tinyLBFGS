#include <iostream>
#include "LBFGS.cpp"


int main() {
    /*
    const LBFGS::func_grad_eval func = [](const std::vector<double> &x, double &f, std::vector<double> &g) {
        f = (x[0] - 7)*(x[0] - 7) +
            (x[1] - 1)*(x[1] - 1);
        g[0] = 2*(x[0] - 7);
        g[1] = 2*(x[1] - 1);
    };
    std::vector<double> x = {10, 10};
    */

    
    const LBFGS::func_grad_eval func = [](const std::vector<double> &x, double &f, std::vector<double> &g) {
        f = 10000*(x[0] * x[0] - x[1])*(x[0] * x[0] - x[1]) + (x[0] - 1)*(x[0] - 1);
        g[0] = 40000 * x[0]*x[0]*x[0] - 40000 * x[0] * x[1] + 2 * x[0] - 2;
        g[1] = 20000 * (x[1] - x[0]*x[0]);
    };
    std::vector<double> x = {-1, -2};
    
    LBFGS::Optimizer opt{func};
    

    int count = opt.run(x);
    std::cout << "iteration count: " << count << '\n';
    std::cout << x[0] << " " << x[1] << "\n"; 
    return std::abs(x[0]-1)>1e-3 || std::abs(x[1]-1)>1e-3;
}