#ifndef __DNB_UTIL_H
#define __DNB_UTIL_H

#include <set>
#include <map>
#include <string>


typedef struct _train_info {
    std::string* tf;
    std::string id;
    std::string label;
    int index;
    int tid;
} tr_info;

typedef struct _corp_info {
    std::map<std::string, std::vector<int> > tffiles;
    std::vector<std::string> rawdata;
    std::vector<std::string> rawids;
    std::map<std::string, int> topicMap;
} corpus;


std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

int read_set(const char *filename, std::set<int>& strset, int(*idlookup)(std::string&));

int read_map(const char *filename, std::map<int, double>& strmap, int(*idlookup)(std::string&));

int read_counts(const char *filename, std::map<int, double>& strmap, int(*idlookup)(std::string&));

int write_map(const char *filename, std::map<int, double>& strmap,
	      std::vector<std::string>& vocab);

int read_corpus(const char *cfile, corpus *corp);

#endif
