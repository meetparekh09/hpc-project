#include <stdio.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "utils.h"
#include "read_file.h"


/******************************************** Function Definitions ********************************************/

void gradient(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
 double *cost, double *gradient, int n, int m);

double hypothesis(double *x, double *theta, int n);

void gradient_descent(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
void (*g)(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
 double *cost, double *grad, int n, int m),
int n, int m, int num_iters);

void stochastic_gradient_descent(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
void (*g)(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
 double *cost, int n, int m, double alpha),
int n, int m, int num_iters);

void gradient_update(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
 double *cost, int n, int m, double alpha);

void omp_gradient(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
  double *cost, double *grad, int n, int m);

void omp_gradient_update(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
   double *cost, int n, int m, double alpha);

double omp_hypothesis(double *x, double *theta, int n);

/***************************************************************************************************************/


/******************************************** Stochastic Gradient Descent ********************************************/

void stochastic_gradient_descent(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
void (*g)(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
 double *cost, int n, int m, double alpha),
int n, int m, int num_iters) {
    double alpha = 1.0/n;
    double cost = 0.0;
    double prev_cost = 0.0;

    for(int i = 0; i < num_iters; i++) {
        g(x, y, theta, &hypothesis, &cost, n, m, alpha);
        if(fabs(prev_cost - cost) < 0.1) {
            printf("Iterations to converge :: %d, Cost :: %lf\n", i, cost);
            break;
        }
        // if(i % 100 == 0)
            printf("Iter :: %d, Cost :: %f\n", i, cost);
        prev_cost = cost;
        cost = 0.0;
    }

}


void gradient_update(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
 double *cost, int n, int m, double alpha) {

     for(int i = 0; i < m*n; i+=n) {
         double h_val = h(x+i, theta, n);
         double diff = y[i/n] - h_val;
         *cost += diff*diff;
     }

    for(int i = 0; i < m*n; i+=n) {
        double h_val = h(x+i, theta, n);
        double diff = y[i/n] - h_val;

        for(int j = i; j < i + n; j++) {
            theta[j%n] += alpha*diff*x[j];
        }
    }

    *cost = (*cost)/2;
}


void omp_gradient_update(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
 double *cost, int n, int m, double alpha) {
     double d_cost = 0.0;

     #ifdef _OPENMP
     #pragma omp parallel for reduction(+:d_cost)
     #endif
     for(int i = 0; i < m*n; i+=n) {
         double h_val = h(x+i, theta, n);
         double diff = y[i/n] - h_val;
         d_cost += diff*diff;
     }

    for(int i = 0; i < m*n; i+=n) {
        double h_val = h(x+i, theta, n);
        double diff = y[i/n] - h_val;

        for(int j = i; j < i + n; j++) {
            theta[j%n] += alpha*diff*x[j];
        }
    }

    *cost = (d_cost)/2;
}


/***************************************************************************************************************/

/******************************************** Gradient Descent ********************************************/

void gradient_descent(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
void (*g)(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
 double *cost, double *grad, int n, int m),
int n, int m, int num_iters) {
    double alpha = 1.0/m/n;
    double cost = 0.0;
    double prev_cost = 0.0;
    double *grad = (double*)malloc(n*sizeof(double));

    for(int i = 0; i < n; i++) {
        grad[i] = 0.0;
    }

    for(int i = 0; i < num_iters; i++) {
        g(x, y, theta, &hypothesis, &cost, grad, n, m);
        if(fabs(prev_cost - cost) < 0.1) {
            printf("Iterations to converge :: %d, Cost :: %lf\n", i, cost);
            break;
        }
        // if(i % 100 == 0)
            printf("Iter :: %d, Cost :: %f\n", i, cost);
        for(int j = 0; j < n; j++) {
            theta[j] += alpha*grad[j];
            grad[j] = 0.0;
        }
        prev_cost = cost;
        cost = 0.0;
    }
}


void gradient(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
 double *cost, double *grad, int n, int m) {

    for(int i = 0; i < m*n; i+=n) {
        double h_val = h(x+i, theta, n);
        double diff = y[i/n] - h_val;
        *cost += diff*diff;

        for(int j = i; j < i + n; j++) {
            grad[j%n] += diff*x[j];
        }
    }

    *cost = (*cost)/2;
}

void omp_gradient(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
 double *cost, double *grad, int n, int m) {
     double d_cost = 0.0;

    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:d_cost)
    #endif
    for(int i = 0; i < m*n; i+=n) {
        double h_val = h(x+i, theta, n);
        double diff = y[i/n] - h_val;
        d_cost += diff*diff;

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(int j = i; j < i + n; j++) {
            grad[j%n] += diff*x[j];
        }
    }

    *cost = (d_cost)/2;
}

/***************************************************************************************************************/


/******************************************** Hypothesis ********************************************/

double hypothesis(double *x, double *theta, int n) {
    double val = 0.0;
    for(int i = 0; i < n; i++) {
        val += theta[i]*x[i];
    }

    return val;
}

double omp_hypothesis(double *x, double *theta, int n) {
    double val = 0.0;

    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:val)
    #endif
    for(int i = 0; i < n; i++) {
        val += theta[i]*x[i];
    }

    return val;
}

/***************************************************************************************************************/


/******************************************** Main Function ********************************************/

int main() {
    Timer t;
    int n = 100;
    int m = 1000000;
    int num_iters = 10;

    double *x = (double*)malloc(m*n*sizeof(double));
    double *y = (double*)malloc(m*sizeof(double));
    double *theta = (double*)malloc(n*sizeof(double));

    // #ifdef _OPENMP
    // int t_num = omp_get_num_threads();
    // printf("Number of threads :: %d\n", t_num);
    // #endif

    printf("Reading Data :: \n");

    t.tic();
    read_x(x, m, n);
    read_y(y, m);

    for(int i = 0; i < n; i++) {
        theta[i] = drand48();
    }
    double time = t.toc();
    printf("Time to Initialize :: %lf\n", time);

    printf("==============================================================================================\n");
    printf("Convergence in OMP Gradient Descent :: \n\n\n");

    t.tic();
    gradient_descent(x, y, theta, hypothesis, omp_gradient, n, m, num_iters);
    time = t.toc();

    printf("Time for OMP Gradient Descent :: %lf\n", time);
    printf("\n\n\n");


    for(int i = 0; i < n; i++) {
        theta[i] = drand48();
    }

    printf("==============================================================================================\n");
    printf("Convergence in Gradient Descent :: \n\n\n");

    t.tic();
    gradient_descent(x, y, theta, hypothesis, gradient, n, m, num_iters);
    time = t.toc();

    printf("Time for Gradient Descent :: %lf\n", time);
    printf("\n\n\n");


    for(int i = 0; i < n; i++) {
        theta[i] = drand48();
    }

    printf("==============================================================================================\n");
    printf("Convergence in Stochastic Gradient Descent :: \n\n\n");

    t.tic();
    stochastic_gradient_descent(x, y, theta, hypothesis, gradient_update, n, m, num_iters);
    time = t.toc();

    printf("Time for Stochastic Gradient Descent :: %lf\n", time);
    printf("\n\n\n");


    for(int i = 0; i < n; i++) {
        theta[i] = drand48();
    }

    printf("==============================================================================================\n");
    printf("Convergence in OMP Stochastic Gradient Descent :: \n\n\n");

    t.tic();
    stochastic_gradient_descent(x, y, theta, omp_hypothesis, omp_gradient_update, n, m, num_iters);
    time = t.toc();

    printf("Time for OMP Stochastic Gradient Descent :: %lf\n", time);
    printf("\n\n\n");
    printf("==============================================================================================\n");
}
