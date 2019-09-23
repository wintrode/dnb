
#include <iostream>
#include <fstream>

#include <set>
#include <map>
#include <vector>
#include <string>

#include <stdio.h>

using namespace std;

int read_set(const char *filename, set<int>& strset, int(*idlookup)(string&)) {
    ifstream inf;
    int id;
    int count = 0;
    string word;

    inf.open(filename);
    if (inf.is_open()) {
	while(!inf.eof()) { 
	    getline(inf, word);
	    if (inf.eof()) break;
	    if ( word.size() < 1 ) continue;
		
	    id = (*idlookup)(word);
	    strset.insert(id);
	    count++;
	}
	inf.close();
    }

    return count;

}

int read_map(const char *filename, map<int, double>& strmap, int(*idlookup)(string&)) {
    ifstream inf;
    int id;
    int count = 0;
    string word;
    double val;
    inf.open(filename);
    if (inf.is_open()) {
	while(!inf.eof()) { 
	    inf >> word;
	    if ( word.size() < 1 ) continue;
	    if (inf.eof()) break;
	    inf >> val;
	    if (inf.eof()) break;

		
	    id = (*idlookup)(word);
	    strmap[id] = val;
	    count++;
	}
	inf.close();
    }

    return count;

}

int read_counts(const char *filename, map<int, double>& strmap, int(*idlookup)(string&)) {
    ifstream inf;
    int id;
    int count = 0;
    string word;
    double val;
    inf.open(filename);
    if (inf.is_open()) {
	while(!inf.eof()) { 
	    inf >> word;
	    if ( word.size() < 1 ) continue;
	    if (inf.eof()) break;
	    inf >> val;
	    if (inf.eof()) break;

	    id = (*idlookup)(word);
	    if (strmap.find(id) == strmap.end()) 
		strmap[id] = 0.0;
	    strmap[id] += val;
	    count++;
	}
	inf.close();
    }
    return strmap.size();
}

int write_map(const char *filename, std::map<int, double>& strmap,
	      std::vector<string>& vocab) {

    FILE *fp;
    //open FILE, ">:utf8", $fweights;
    fp = fopen(filename, "w");
    map<int,double>::iterator si;
    int count = 0;
    if (fp) {

	for (si=strmap.begin(); si != strmap.end(); si++) {
	    if (si->first < vocab.size()) {
		fprintf(fp, "%s %g\n", vocab[(*si).first].c_str(), (*si).second);
		count++;
	    }
	}
		
	fclose(fp);
	
    }
    
    return count;
    
}
