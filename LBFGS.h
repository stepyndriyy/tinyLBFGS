#pragma once

#include <functional>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>

namespace LBFGS {
    typedef std::vector<double> vector;
    typedef std::function<void(const vector &x, double &f, vector &g)> func_grad_eval;
    typedef std::deque<vector> history;

    struct Optimizer {
        Optimizer(func_grad_eval func_grad) : func_grad(func_grad) {}
        int run(vector &sol);

        struct IHessian {
            void mult(const vector &g, vector &result) const; 
            void add_correction(const vector &s, const vector &y);

            const int history_depth;

            history S = {};
            history Y = {};
            double gamma = 1.;
        } invH = { 2 };

        const func_grad_eval func_grad;


        // L-BFGS user parameters
        int maxiter = 10000; // maximum number of quasi-Newton updates
        double gtol = 1e-4; // the iteration stops when ||g||/max(1,||x||) <= gtol

        // Line search user parameters: the step size must satisfy Wolfe conditions with these parameters
        double c1  = 1e-4; // sufficient decrease constant (Armijo rule)
        double c2 = 0.9; // curvature condition constant
        double line_search_failiure = 1e-3; // constant on line search fail

        bool verbose = true;    
         
    };

}