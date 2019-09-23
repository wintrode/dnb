#ifndef __DNB_DETTOOLS_H
#define __DNB_DETTOOLS_H


#include <vector>
#include <algorithm>

using namespace std;

#define NO_TGT -1
#define NO_NONTGT -2

typedef struct _cost_param {
	double prior;
	double m0;
	double f0;
	double m1;
	double f1;

} cost_param;

int det_eer(vector<double> &tgtscores, vector<double> &ntgtscores, cost_param *cost, double *eer);


#endif
