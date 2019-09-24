
#include <iostream>
#include <fstream>
#include <sstream>

#include <set>
#include <map>
#include <vector>
#include <string>

#include <stdio.h>
#include <string.h>

#include "util.h"

using namespace std;

vector<string> &split(const string &s, char delim, vector<string> &elems) {
    std::stringstream ss(s);
    std::string item;
    elems.clear();
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}



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

int read_corpus(const char *cfile, corpus *corp) {
    ifstream inf;
    int ND = 0;
    inf.open(cfile);
    if (!inf.is_open()) {
	cerr << "Unable to open " << cfile << ".\n";
	return 0;
    }
    
    string line, tp,utt;
    int cpos;
    int cp2;
    int slen;
    int dataPos;

    while(1) {
	getline(inf, line); 
	if (inf.eof()) break;
	vector<int> flist;	

	if ( line.size() < 1 ) continue;

	cpos = line.find('\t', 0);

	if (cpos == string::npos) continue;
	tp = line.substr(0, cpos);
	
	cp2 = line.find('\t', cpos+1);
	if (cp2 == string::npos) continue;

	utt = line.substr(cpos+1, cp2-(cpos+1));
	corp->rawdata.push_back(line.substr(cp2+1));
	corp->rawids.push_back(utt);

	dataPos = corp->rawdata.size()-1;

	// what if tp =~ /:/?

	if (strchr(tp.c_str(), ':')) {
	    //
	    vector<string> tlist;
	    split(tp, ':', tlist);
	    
	    for (int tt=0; tt < tlist.size(); tt++) {
		corp->topicMap[tlist[tt]] = 1;
		if (corp->tffiles.count(tlist[tt]) == 0) {
		    corp->tffiles[tlist[tt]] = flist;
		}
		corp->tffiles[tlist[tt]].push_back(dataPos);
	    }

	    
	}
	else {
	    corp->topicMap[tp] = 1;
	    if (corp->tffiles.count(tp) == 0) {
		corp->tffiles[tp] = flist;
	    }
	    corp->tffiles[tp].push_back(dataPos);
	}
	ND++;
    }
    inf.close();

    return ND;

}
