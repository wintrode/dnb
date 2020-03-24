#include <vector>

#include <iostream>
#include <fstream>
#include <string>

#include <string.h>

#include "dettools.h"

using namespace std;

int main(int argc, char ** argv) {
    
    vector<double> tgt;
    vector<double> nontgt;
    double eer=1.0;

    istream *in;
    ifstream inf;

    if (argc < 2 || !strcmp("-", argv[1])) {
	in = &cin;
    }
    else {
        inf.open(argv[1]);
	in = &inf;
	if (!inf.is_open()) {
	    cerr << "Unable to open " << argv[1] << ".\n";
	    exit(1);
	}
    }


    double val;

    string tp;
    while(!in->eof()) {
	*in >> val;
	if (in->eof()) break;
	*in >> tp;
	if (in->eof()) break;
	if (!strcmp("target", tp.c_str())) 
	    tgt.push_back(val);
	else
	    nontgt.push_back(val);
    }
    if (in != cin) 
	inf.close();
    
    det_eer(tgt, nontgt, 0, &eer);

    cout << "EER " << eer << "\n";


}
