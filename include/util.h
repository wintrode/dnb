#ifndef __DNB_UTIL_H
#define __DNB_UTIL_H

#include <set>
#include <map>
#include <string>


int read_set(const char *filename, std::set<int>& strset, int(*idlookup)(std::string&));

int read_map(const char *filename, std::map<int, double>& strmap, int(*idlookup)(std::string&));

int read_counts(const char *filename, std::map<int, double>& strmap, int(*idlookup)(std::string&));

int write_map(const char *filename, std::map<int, double>& strmap,
	      std::vector<string>& vocab);

#endif
