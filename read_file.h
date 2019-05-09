#ifndef _READ_FILE_H_
#define _READ_FILE_H_

#include <stdio.h>
#include <stdlib.h>

#define ROW_LIMIT 1000000
#define COL_LIMIT 100

void read_y(double *y, long row_limit);
void read_x(double *x, long row_limit, int col_limit);

void read_y(double *y, long row_limit) {
    FILE *fp = fopen("data/y.csv", "r");

    if(row_limit > ROW_LIMIT) {
        row_limit = ROW_LIMIT;
    }

    for(long i = 0; i < row_limit; i++) {
        fscanf(fp, "%lf", y+i);
    }
}

void read_x(double *x, long row_limit, int col_limit) {
    char *base_string = "data/xcol";
    char *file_type = ".csv";
    char filename[20];

    if(row_limit > ROW_LIMIT) {
        row_limit = ROW_LIMIT;
    }
    if(col_limit > COL_LIMIT) {
        col_limit = COL_LIMIT;
    }

    for(int i = 0; i < col_limit; i++) {
        sprintf(filename, "%s%d%s", base_string, i+1, file_type);
        // printf("Reading %d column from %s\n", i+1, filename);
        FILE *fp = fopen(filename, "r");
        for(long j = 0; j < row_limit; j++) {
            fscanf(fp, "%lf", x + i + j*col_limit);
        }
    }
}

#endif
