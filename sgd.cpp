#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "read_file.h"


/******************************************** Function Definitions ********************************************/

void gradient(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
 double *cost, double *gradient, int n, int m);
double hypothesis(double *x, double *theta, int n);
void gradient_descent(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
int n, int m, int num_iters);
void stochastic_gradient_descent(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
int n, int m, int num_iters);
void gradient_update(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
 double *cost, int n, int m, double alpha);

/***************************************************************************************************************/


/******************************************** Stochastic Gradient Descent ********************************************/

void stochastic_gradient_descent(double *x, double *y, double *theta, double (*h)(double *x, double *theta, int n),
int n, int m, int num_iters) {
    double alpha = 1.0/n;
    double cost = 0.0;

    for(int i = 0; i < num_iters; i++) {
        gradient_update(x, y, theta, &hypothesis, &cost, n, m, alpha);
        if(i % 100 == 0)
            printf("Iter :: %d, Cost :: %f\n", i, cost);
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


/***************************************************************************************************************/

/******************************************** Gradient Descent ********************************************/

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

        if(i % 100 == 0)
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

/***************************************************************************************************************/


/******************************************** Hypothesis ********************************************/

double hypothesis(double *x, double *theta, int n) {
    double val = 0.0;
    for(int i = 0; i < n; i++) {
        val += theta[i]*x[i];
    }

    return val;
}

/***************************************************************************************************************/


/******************************************** Main Function ********************************************/

int main() {
    int n = 100;
    int m = 10000;
    int num_iters = 1000;

    double *x = (double*)malloc(m*n*sizeof(double));
    double *y = (double*)malloc(m*sizeof(double));
    double *theta = (double*)malloc(n*sizeof(double));

    read_x(x, m, n);
    read_y(y, m);

    for(int i = 0; i < n; i++) {
        theta[i] = drand48();
    }


    printf("==============================================================================================\n");
    printf("Convergence in Gradient Descent :: \n\n\n");
    gradient_descent(x, y, theta, hypothesis, n, m, num_iters);
    printf("\n\n\n");


    for(int i = 0; i < m*n; i+=n) {
        y[i/n] = drand48();
        for(int j = i; j < i + n; j++) {
            x[j] = drand48();
            if(i == 0) theta[j] = drand48();
        }
    }


    printf("==============================================================================================\n");
    printf("Convergence in Stochastic Gradient Descent :: \n\n\n");
    stochastic_gradient_descent(x, y, theta, hypothesis, n, m, num_iters);
    printf("\n\n\n");
    printf("==============================================================================================\n");
}
