#include <functional>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>
#include "LBFGS.h"

namespace LBFGS {

    double dot(const vector &a, const vector &b) {
        assert(a.size() == b.size());
        double dot = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
        }
        return dot;
    }

    double norm(const vector &v) {
        return std::sqrt(dot(v, v));
    }
   
    // backtracking line search with strong Wolfe conditions
    bool line_search_backtracking_primitive(func_grad_eval func_grad, const vector &x, const vector &p, double &alpha, const double c1, const double c2,  int maxiter=20) {
        const int n = static_cast<int>(x.size());
        int iter = 0;
        
        double fx; 
        vector g(n);
        func_grad(x, fx, g);

        double dot_g_p = dot(g, p);
        vector x_next(n), g_next(n);
        double f_next;
        for (int i = 0; i < n; ++i) {
            x_next[i] = x[i] + alpha * p[i];
        }
        func_grad(x_next, f_next, g_next);
        while (
            (f_next >= fx + (c1 * alpha * dot_g_p) ||
            std::abs(dot(g_next, p)) >= c2*std::abs(dot_g_p)) &&
            iter < maxiter
        ) { 
            alpha *= 0.5;
            for (int i = 0; i < n; ++i) {
                x_next[i] = x[i] - alpha * p[i];
            }
            func_grad(x_next, f_next, g_next);
            iter++;
        }

        if (iter>=maxiter) {
            return false;
        }
        return true;
        
    }


    // Add a correction pair {s, y} to the optimization history
    void Optimizer::IHessian::add_correction(const vector &s, const vector &y) {
        const int n = static_cast<int>(s.size());
        const int m = static_cast<int>(S.size());

        assert(static_cast<int>(Y.size()) == m);
        assert(static_cast<int>(y.size()) == n);

        if (m==history_depth) {
            S.pop_back();
            Y.pop_back();
        }
        S.push_front(s);
        Y.push_front(y);

        double yy = dot(y, y);
        assert(std::abs(yy) > 0);

        gamma = dot(y, s)/yy;
        assert(std::isfinite(gamma));
        
    }

    // L-BFGS two-loop recursion algorithm
    void Optimizer::IHessian::mult(const vector &g, vector &result) const {
        const int n = static_cast<int>(g.size());
        const int m = static_cast<int>(S.size());
        assert(static_cast<int>(Y.size()) == m);

        result = g;

        if (!m) return;

        vector a(m);
        for (int i=0; i<m; i++) {
            const vector &y = Y[i];
            const vector &s = S[i];

            double sy = dot(s, y);
            assert(std::abs(sy) > 0);
            a[i] = dot(s, result)/sy;
            assert(std::isfinite(a[i]));
            
            for (int j=0; j<n; j++)
                result[j] -= a[i]*y[j];
        }

        for (int j=0; j<n; j++)
            result[j] *= gamma;

        for (int i=m; i--;) {
            const vector &y = Y[i];
            const vector &s = S[i];
            double b = dot(y, result)/dot(s, y);
            assert(std::isfinite(b));
            for (int j=0; j<n; j++)
                result[j] += (a[i]-b)*s[j];
        }
    }

    // Run an L-BFGS optimization
    int Optimizer::run(vector &x) {
        const int n = static_cast<int>(x.size());
        double f;
        vector g(n), p(n);

        func_grad(x, f, g);
        for (int i=0; i<maxiter; i++) {

            if (norm(g)/std::max(1., norm(x)) < gtol) {
                if (verbose) std::cerr << "||g||/max(1,||x||)  <= " << gtol << std::endl;
                return i + 1;
            } 
            
            invH.mult(g, p);

        
            vector xprev = x, gprev = g;
            double alpha = 1;
            bool line_search_fl = false;
            if (!(line_search_fl = line_search_backtracking_primitive(func_grad, x, p, alpha, c1, c2))) {
                alpha = line_search_failiure; // move a bit
            }
            for (int j=0; j<n; j++)
                x[j] = xprev[j]-p[j]*alpha;
            func_grad(x, f, g);
     
            vector s(n), y(n);
            for (int j=0; j<n; j++) {
                s[j] = x[j]-xprev[j];
                y[j] = g[j]-gprev[j];
            }
            invH.add_correction(s, y);

        }
        return -1;
    }
}
