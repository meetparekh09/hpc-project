#include <stdio.h>
#include <math.h>
#include "utils.h"


void gradient(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
 double *cost, double *gradient, int n, int m);
double hypothesis(double *x, double *theta, int n);
void gradient_descent(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
int n, int m, int num_iters);



void gradient_descent(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
int n, int m, int num_iters) {
    double alpha = 1.0/m/n;
    double cost = 0.0;
    double *grad = (double*)malloc(n*sizeof(double));

    for(int i = 0; i < n; i++) {
        grad[i] = 0.0;
    }

    for(int i = 0; i < num_iters; i++) {
        gradient(x, y, theta, &hypothesis, &cost, grad, n, m);
        printf("Iter :: %d, Cost :: %f\n", i, cost);
        for(int j = 0; j < n; j++) {
            theta[j] += alpha*grad[j];
            grad[j] = 0.0;
        }
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

double hypothesis(double *x, double *theta, int n) {
    double val = 0.0;
    for(int i = 0; i < n; i++) {
        val += theta[i]*x[i];
    }

    return val;
}

int main() {
    int n = 100;
    int m = 10000;
    int num_iters = 1000;

    double *x = (double*)malloc(m*n*sizeof(double));
    double *y = (double*)malloc(m*sizeof(double));
    double *theta = (double*)malloc(n*sizeof(double));

    for(int i = 0; i < m*n; i+=n) {
        y[i/n] = drand48();
        for(int j = i; j < i + n; j++) {
            x[j] = drand48();
            if(i == 0) theta[j] = drand48();
        }
    }

    gradient_descent(x, y, theta, hypothesis, n, m, num_iters);


}
