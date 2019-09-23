#define HAVE_CXX_STDHEADERS

#include <iostream>

#include <fstream>


#include <stdio.h>

#include <sstream>
#include <map>
#include <vector>

#include <string>

#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>

#include "dettools.h"
#include "util.h"

#include <time.h>

stringstream ss (stringstream::in | stringstream::out);

double diffclock(clock_t clock1,clock_t clock2)
{
	double diffticks=clock1-clock2;
	double diffms=(diffticks*10)/CLOCKS_PER_SEC;
	return diffms;
}


using namespace std;

typedef pair<int, map< int, double> > mpair;
typedef pair<int, vector<double> > vpair;

FILE* outp = NULL;


double zero = 1e-15;

int maxWordId = 0;
map<string,int> wordids;
vector<string> wvec;

vector<string> rawdata;
vector<string> rawids;

map<string,string> featStrings;

typedef struct _train_info {
    string* tf;
    string id;
    string label;
    int index;
    int tid;
} tr_info;


map<int, vector<double> > *topcount;

vector<double> *tprop;
vector<double> *tcount;
map<int,double> *prior;
vector<double> *wcount;
map<int,double> *vcount;
int *counts;
vector< map<int, map<int, double> > > docCount;

double *corpcount;
double *vocabSize;

vector<string> topics;
map<string, int> topicMap;

vector<tr_info> *partitions;

double *llrs;

map<string, vector<int> > tffiles;
set<int> stopwords;
set<int> tstopwords;
set<int> vocab;


vector<map<int,double> > testdocs;

vector< map<int,double> >  *fcache;

map<int,double> empty;

#define MCE 0
#define L1 1
#define L2 2
#define TPL 3

int lfunc = L1;

// INITIALIZE ALL THESE...


struct option opts[] = 
{
   {"train", 1, 0, 't'},
	{"test", 1, 0, 's'},
	{"workdir", 1, 0, 'w'},
	{"prdata", 1, 0, 'p'},
	{"output", 1, 0, 'o'},
	{"model", 1, 0, 'm'},

	{"feature-weights", 1, 0, 'f'},
	{"target", 1, 0, 'T'},
	{"iterations", 1, 0, 'i'},
	{"ignore", 1, 0, 'I'},
	{"K", 1, 0, 'K'},
	{"debug", 1, 0, 'd'},
	{"bernoulli", 1, 0, 'b'},
	{"epsilon", 1, 0, 'e'},
	{"unbalanced", 1, 0, 'u'},
	{"pr-freq", 1, 0, 'F'},
	{"word-classes", 1, 0, 'W'},
	{"class-priors", 1, 0, 'c'},
	{"normalize", 1, 0, 'z'},
	{"prune", 1, 0, 'P'},
	{"vocab", 1, 0, 'V'},
	{0,0,0,0}
};
   

//my @fcache = ();

int topicCount;
map<int,double> weights;


char *test = 0;
char *train = 0;
char *workdir = 0;
char *prdata  = 0 ;
char *output = 0 ;
char *model = 0;

char *weightFile = 0 ;
char *wordclasses= 0;
char *target = 0 ;
char *stoplist = 0;
char *vocabfile = 0;

char *tstoplist = 0;
char *ignoreW = 0;

int iterations;
int K;
double epsilon = 0.1;
double beta = 1;
int bernoulli = 0;
int debug = 0;
int unbalanced  = 0;
int classPriors  = 0;
double prune = -1;
string argerr;
int znorm = 0;

int singleMax = 1;

void initialize_arrays(int ND) {
	topcount = new map<int, vector<double> >[K+1];
	tprop = new vector<double>[K+1];
	tcount = new vector<double>[K+1];

	prior= new map<int, double>[K+1]; 
	wcount = new vector<double>[K+1];
	vcount = new map<int, double>[K+1];

	corpcount = new double[K+1];
	vocabSize = new double[K+1];
//	topics = new string[topicCount];

	partitions = new vector<tr_info>[K];

  	llrs = new double[topicCount];

	fcache = new vector< map<int,double> >[K+1];

}

void delete_arrays(int ND) {
	int i;
	delete[] topcount;
	
	delete[] tprop;
	delete[] tcount;
	delete[] prior;
	delete[] wcount;
	delete[] vcount;


	delete[] corpcount;
	delete[] vocabSize;
	//delete[] topics;

	delete [] partitions;
	delete [] llrs;
	
	delete [] fcache;

}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    elems.clear();
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


int parse_args(int argc, char **argv) {

	int idx,opt;

	while ((opt = getopt_long(argc,argv, "G:t:s:w:p:o:f:T:i:K:b:e:F:W:S:ducP:zV:m:", opts, &idx)) != -1) {
		switch (opt) {
		case 't':
			train = optarg; break;
		case 's':
			test = optarg; break;
		case 'w':
			workdir = optarg; break;
		case 'p':
			prdata = optarg; break;
		case 'o':
			output = optarg; break;
		case 'm':
			model = optarg; break;
		case 'f':
			weightFile = optarg; break;
		case 'T':
			target = optarg; break;
		case 'i':
			iterations = atoi(optarg); break;
		case 'K':
			K = atoi(optarg); break;
		case 'b':
			bernoulli = 1; break;
		case 'W':
			wordclasses = optarg; break;
		case 'S':
			stoplist = optarg; break;
		case 'V':
			vocabfile = optarg; break;
		case 'G':
			tstoplist = optarg; break;
		case 'I':
			ignoreW = optarg; break;
		case 'd':
			debug = 1; break;
		case 'u':
			unbalanced = 1; break;
		case 'c':
			classPriors = 1; break;
		case 'z':
			znorm = 1; break;
		default:
			argerr = "Unknown argument\n";
			return 1;	
		}

	}
	return 0;
}






inline int word_id(string &word) {
	map<string,int>::iterator iit;

	iit = wordids.find(word);
	if (iit == wordids.end()) {
		wordids[word] = maxWordId;
		wvec.push_back(word);
		return maxWordId++;
	}
	else 
		return (*iit).second;
}



double get_llr(map<int,double> *tdoc, int topic, int part, double *results) {
    //my ($dref, topic, part, $idx) = @_;
    vector<double> vempty;
    double llr = 0;
	
    int i,w=0;
	
    for (i=0; i < topicCount; i++) { // tp (keys(%topics)) {
		results[i] = 0.0;
    }

    int word;
    double wt;

    map<int, double>::iterator j;
    
    map<int, vector<double> >::iterator jj;

    double p,q,logp,logq,d,f,qp,pp;
    string tp,t2;
    
    double pr;
    
    double *parray = new double[topicCount];

	string cw = "buenas";
	int checkwd = word_id(cw);
		
	double chk = 0.0;
		

	
    for (j=tdoc->begin(); j != tdoc->end(); j++) {
		word = (*j).first;   // (sort(keys(%tdoc))) {
	
		//if (word == checkwd)
		//	chk += (*j).second;

		if (weights.count(word) > 0) {
			wt = weights[word];
		}
		else  {
			wt = 1;
		}
		//	# what to do for OOV weights
		
		if (wt == 0.0) 
			continue;
		
		if (fcache[part].size() < topicCount) {
			fcache[part].resize(topicCount);//, map<int,double>());
			//fcache[i] = empty;
		}
	
        if (prior[part].count(word) == 0) {
			pr = prior[part][word] = 1 / (corpcount[part] + vocabSize[part]);
		}
		else {
			pr = prior[part][word];
		}

		
	q = 0;
	for (i=0; i < topicCount; i++) { // tp (keys(%topics)) {
	    parray[i] = 0.0;
	    if (fcache[part][i].count(word) > 0) {
			f = fcache[part][i][word];
			
			results[i] += wt * (*j).second * f;
			
			continue;					
	    }
		
	    jj = topcount[part].find(word);
	    if (jj == topcount[part].end()) 
		d = 0.0;
	    else 
		d = (*jj).second[i];

		if (word == checkwd && i == 0)  {
			cout << "Topic " << i << " " << vocabSize[part] <<  " " << pr << " " << wcount[part][i] << "\n";
			chk = d; 
		}
		
		//if (word == checkwd) 
		//	chk = d; 

	    
	    p = (d + vocabSize[part] * pr) /
		(wcount[part][i] + vocabSize[part]);
	    if (p == 0.0) {
		p = zero;
		cerr << "Canot have a zero value in log prob calc.\n";
	    }
	    
	    q += p;
	    parray[i] = p;	
	}
	
			
	// # compute log(p(w|!t)) over all topics
	for (i=0; i < topicCount; i++) { // tp (keys(%topics)) {
	    if (fcache[part][i].count(word) > 0) {
		continue;					
	    }
	    
	    d = (q - parray[i]) / (topicCount-1);
	    if (d == 0) {
		d = zero;
	    }
	    
	    f = log(parray[i]) - log(d);	
		
	    fcache[part][i][word] = f;
	    
	    results[i] += wt * (*tdoc)[word] * f;
		//if (word == checkwd)
			//	chk =  parray[0];
		
	}
	
	
	
    }

    delete[] parray;
    
    //cout << "Check word count "<< chk << "\n";

    if (classPriors) {
	qp = 0;
	for (i=0; i < topicCount; i++) { // tp (keys(%topics)) {
	    qp += tcount[part][i];
	}
	// gold morgan ellis, speech and audio signal processing
	// pierce -- whither speech recognition
	
	for (i=0; i < topicCount; i++) { // tp (keys(%topics)) {
	    q = qp - tcount[part][i];
	    
	    q /= (topicCount- 1);
	    pp = tcount[part][i];
	    results[i] += log(pp) - log(q);
	}

    }
    
    return 0.0;
  

}

bool comparator ( const pair<int,double>& l, const pair<int,double>& r)
   { return l.first < r.first; }

class gt {
vector<double>& _x;
public:
	gt( vector<double>& x ) : _x(x) {}	
	bool comp( int j, int k ) const { return _x[j] > _x[k]; }
	bool operator()	( int j, int k ) const { return _x[j] > _x[k]; }
};



vector<int> truth;
vector<map<int,int> > truthMap;

double fmSum = 0;
int fpsum = 0;
int tpsum = 0;
int fnsum = 0;

//#ifndef ORIGINAL_LOSS
double *maxLLRs = 0;  //new double[topicCount];
int *maxtopics = 0;  //new int[topicCount];
int *maxindexs = 0;  //new int[topicCount];

double ** maxLLR2 = 0; // = new double*[topicCount];
int ** maxindex2 = 0; //= new int*[topicCount];

double *sumLLR = 0;

double targetLLR;
double maxLLR;
int maxtopic;

double* targetLLRs = 0;

vector<double> mus;
vector<double> sigmas;

vector<string> testfiles;


inline int read_testdocs(char *test) {
    ifstream tfin;
    ifstream docin;
    string tfdata,word,topic;
    double count;
    tfin.open(test);
    
    map<int, double> doc;
    
    int x = 0;
    int wid;
    
    string line;
    string utt;
    vector<string> parts;
    int cpos, cp2;


    while (1)  {
	getline(tfin, line); 
	if (tfin.eof()) break;
	vector<int> flist;	
	// HERE
	if (debug) 
	    cerr << line << "\n";
	if ( line.size() < 1 ) continue;
	
	cpos = line.find('\t', 0);
	
	if (cpos == string::npos) continue;
	topic = line.substr(0, cpos);
	
	cp2 = line.find('\t', cpos+1);
	if (cp2 == string::npos) continue;
	
	utt = line.substr(cpos+1, cp2-(cpos+1));
	tfdata = line.substr(cp2+1);

	//rawids.push_back(utt);
	//dataval = rawdata.size()-1;
    
	vector<string> tlist;
	
	if (strchr(topic.c_str(), ':')) {
	    split(topic, ':', tlist);
	    topic = tlist[0];
	}
	else {
	    tlist.push_back(topic);
	}
		
	testfiles.push_back(utt);
	//docin.open(tffile.c_str());

	testdocs.push_back(doc);
	ss.clear();
	ss.str(tfdata);
	while (1) {
	    if (ss.eof()) break;
	    
	    ss >> word;
	    split(word, ':', parts);
	    word = parts[0];
	    count = strtod(parts[1].c_str(), 0);

	    if (bernoulli) count = 1;
	    wid = word_id(word);
	    
	    if (stopwords.find(wid) != stopwords.end() ) continue;
	    if (tstopwords.find(wid) != tstopwords.end() ) continue;
	    
	    if (vocabfile && vocab.find(wid) == vocab.end()) continue;
	    
	    testdocs[x][wid] = count;
	    
	}	
	docin.close();
	
	truth.push_back(topicMap[topic]);
	
	truthMap.push_back(map<int,int>());
	for (int tt=0; tt < tlist.size(); tt++) {
	    truthMap[x][topicMap[tlist[tt]]] = 1;
	}
	
	
	x++;
    }	
    tfin.close();
}


double score_test(int N, char *test, char *prdata, char *output, double* test_results) {

    int len;
    
    int m=0;
    
    int x = 0;
    int wid;
    string word;
    double count;
    
    string tffile, topic;
    if (testdocs.size() == 0) { 
	x = read_testdocs(test);		
    }

    double errors = 0;

    vector<vector<double> > llrscore;
    vector<vector<double> > tgtscrvec;
    vector<vector<double> > nontgtscrvec;

    vector<int> vi;

    vector<double> tmpl;

    int tgtID = 0;
    for (m=0; m < topicCount; m++) {
	llrscore.push_back(tmpl);
	tgtscrvec.push_back(tmpl);
	nontgtscrvec.push_back(tmpl);
	if (target && !strcmp(target, topics[m].c_str()))
	    tgtID = m;
    }	

    map<int, double> *testdoc;
    
    double llr;
    double max, prec, recall, y, lastp, last_r, dr;
    int maxtgt;
    
    double lw;
    int A=0;
    int k=0;
    int tid;
    //int j=0;

    vector<int>tp;
    vector<int>fn;
    vector<int> fp;
    
    tp.resize(topics.size(), 0);
    fn.resize(topics.size(), 0);
    fp.resize(topics.size(), 0);

    for (x=0; x <testdocs.size(); x++) {
	testdoc = &(testdocs[x]);
	
	max = -9999;
	maxtgt = m;
	
	memset(llrs, 0, sizeof(double) * topicCount);

	get_llr(testdoc, -1, K, llrs);

	tid = truth[x];
	
	if (output)
	  fprintf(outp, "test %s %s", testfiles[x].c_str(), topics[truth[x]].c_str());
	
	m=0;
	for (m=0; m < topicCount; m++) {
	    if (znorm)
		llrs[m] = (llrs[m] - mus[m]) / (sqrt(sigmas[m]));
	    
	    llrscore[m].push_back(llrs[m]);
	    
	    if (llrs[m] > max) {	
		max = llrs[m];
		maxtgt = m;
	    }
	    
	    if (output) 
		fprintf(outp, " %0.4f",llrs[m]);
	
	    	    
	    if (m == truth[x] || truthMap[x].find(m) != truthMap[x].end()) {
		tgtscrvec[m].push_back(llrs[m]);
	    }
	    else {
		nontgtscrvec[m].push_back(llrs[m]);
	    }
	}
	

	for (m = 0; m < topicCount; m++) {
	    if (m == truth[x] || truthMap[x].find(m) != truthMap[x].end()) {	
		if (m == maxtgt)
		    tp[m]++;
		else
		    fn[m]++;
	    }
	    else {
		if (m == maxtgt)
		    fp[m]++;

	    }

	}

	if ( maxtgt != truth[x] && truthMap[x].find(maxtgt) == truthMap[x].end()) {
	    errors++;
	}
	if (output) {
	    fprintf(outp, "\n");
	}
	    
	
	vi.push_back(x);
    }
	
    
    
	
    int ridx = 0;
    
    double eer;
    
    int  correct,total,denom;
    double auc,maxp,error;
	
    vector<int>::iterator vit;
    
    vector<double> rarray;
    vector<double> parray;
    
    for (m=0; m < topicCount; m++) {
	if (target && m != tgtID)
	    continue;
	
    	if (prdata) {
	    //	open PRDATA, "> $prdata.$tgt";
    	}
    
		
	
    	if (tgtscrvec[m].size() > 0) {
	  
	  double f1prec = 0;
	  if (tp[m] + fp[m] > 0) {
	    f1prec = (double) tp[m] / (double) (tp[m] + fp[m]);
	  }
	  else {
	    f1prec = 0;
	  }
	  double f1rec = (double) tp[m] / (double) (tp[m] + fn[m]);
	  tpsum += tp[m]; 
	  fnsum += fn[m];
	  fpsum += fp[m];
	  double fmeas = (f1prec + f1rec == 0) ? 0: ((2.0 * f1rec * f1prec) / (f1rec + f1prec));


	  //	  cout << topics[m] << " F1=" << fmeas << " TP=" << tp[m] << " FP=" << fp[m] <<  " FN=" << fn[m] << " " << f1prec  << " " << f1rec <<"\n";
	  fmSum += fmeas;
	  
	  det_eer(tgtscrvec[m], nontgtscrvec[m], 0, &eer);
    //		eer = det_eer(det);
	  lastp = 1.0;
	  last_r = 0.0;
	    
	  auc = 0;
	  correct = 0;
	  total = 0;
	  //@rarray = ();
	  //@parray = ();
	  rarray.clear();
	  parray.clear();
	  denom = tgtscrvec[m].size();
	
	  sort( vi.begin(), vi.end(), gt(llrscore[m]) );

			
	  for (vit = vi.begin(); vit != vi.end(); vit++) {
			
	      total++;
	      if (truth[*vit] == m ||
		  truthMap[*vit].find(m) != truthMap[*vit].end()) {
		  correct++;
		  recall = (double) correct / (double) denom;
		  prec = (double) correct / (double)total;
		  rarray.push_back(recall);
		  parray.push_back(prec);
		  
	      }
			
	  }
	    
	  maxp = 0;
	  //# smooth precision scores
	  for (x = parray.size()-1; x >= 0; x--) {
	      if (parray[x] < maxp) {
		  parray[x] = maxp;
	      }
	      if (parray[x] > maxp) {
		  maxp = parray[x];
	      }
	  }
	  
	  for (x = 0; x < parray.size(); x++) {
	      recall = rarray[x];
	      prec = parray[x];
	      dr = recall - last_r;
	      y = prec;
	      
	      auc += dr * y;
	      auc += (lastp - y) * dr /2; // # trapezoid rule
	      last_r = recall;
	      lastp = prec;
	  }
	  
	  /*if (defined($prdata)) {	    
	    for ($x=0; $x < scalar(@parray); $x++) {
	    print PRDATA "$rarray[$x] $parray[$x]\n";
	    }
	    close PRDATA;
	    }*/
	  
	  if (output) {
	      //close OUTPUT;
	  }
	  
		}
	else {
	    auc = -1.0;
	    eer = 1.0;
	}
	test_results[ridx++] = auc;
	test_results[ridx++] = eer;
	
	//exit(1);
	
    }
    
    
//  #  return ($auc, $eer, $errors / scalar(@testdocs));
    return (double)	errors / (double) testdocs.size();
   
}

vector < vector <double> > trainscores;

int docx = 0;

double score_train(int N, int part, char *prdata, char *output, double* test_results) {

	int len;

	int m=0;

    int x = 0;
	int wid;
	string word;
	double count;

	double errors = 0;

    vector<vector<double> > llrscore;
    vector<vector<double> > tgtscrvec;
    vector<vector<double> > nontgtscrvec;

	vector<int> vi;

    vector<double> tmpl;

    int tgtID = 0;
    mus.resize(topicCount);
    sigmas.resize(topicCount);
    
    for (m=0; m < topicCount; m++) {
	llrscore.push_back(tmpl);
	tgtscrvec.push_back(tmpl);
    	nontgtscrvec.push_back(tmpl);
    	if (target && !strcmp(target, topics[m].c_str()))
	    tgtID = m;
    	//mus[m]=0.0;
    	sigmas[m]=0.0;
    }	
    
    map<int, double> *testdoc;
    
    double llr;
    double max, prec, recall, y, lastp, last_r, dr;
    int maxtgt;
    
    double lw;
    int A=0;
    int k=0;
    int tid;
    int index;
    
    
    if (output) {
      fprintf(outp, "set file truth", tid);
      for (m=0; m < topicCount; m++) {
	fprintf(outp, " %s",topics[m].c_str());
      }
      fprintf(outp, "\n");
    }
	

    for (x=0; x <counts[part]; x++) {
    	
	tid = partitions[part][x].tid;
	index = partitions[part][x].index;
	testdoc = &(docCount[tid][index]);

    	
    	if (target) {
    		
    		llr = get_llr(testdoc, tgtID, part, llrs);

    		// calculate testLoss    		
    		if (output) {
    			//print OUTPUT "llr\n";
    			//cout << llr << "\n";
    		}
    		
    		if (tid == tgtID) {
    			tgtscrvec[tgtID].push_back( llr );
    			if (llr <= 0) {
    				errors++;
    			}
			}
			else {
				nontgtscrvec[tgtID].push_back(llr);
				    			
				if (llr > 0) {
					errors++;
				}
			}
	    
			llrscore[tgtID].push_back(llr);

	    }
	    else {
		max = -9999;
		maxtgt = m;
		
		get_llr(testdoc, -1, part, llrs);
		
		if (output) {
		    fprintf(outp, "train %s %s", partitions[part][x].id.c_str(), topics[tid].c_str());
		}
		
		m=0;
		for (m=0; m < topicCount; m++) {
		    mus[m] += llrs[m];
		    llrscore[m].push_back(llrs[m]);
		    if (trainscores[docx].size() != topicCount)
			trainscores[docx].resize(topicCount);
		    trainscores[docx][m] = llrs[m];
		    
		    if (llrs[m] > max) {	
			max = llrs[m];
			maxtgt = m;
		    }
		    if (output) {
			fprintf(outp, " %0.4f",llrs[m]);
		    }
		    
		    if (m == tid) {
			tgtscrvec[m].push_back(llrs[m]);
		    }
		    else {
			nontgtscrvec[m].push_back(llrs[m]);
		    }
		}
		docx++;
		
		if ( maxtgt != tid) {
		    errors++;
		}
		
		if (output) {
		    fprintf(outp, "\n");
		}
	    }
		
	vi.push_back(x);
    }
    
 	
    
    int ridx = 0;
    
    double eer;
    
    int  correct,total,denom;
    double auc,maxp,error;
    
    vector<int>::iterator vit;
    
    vector<double> rarray;
    vector<double> parray;
    
    for (m=0; m < topicCount; m++) {
	if (target && m != tgtID)
	    continue;
	
    	if (prdata) {
	    //	open PRDATA, "> $prdata.$tgt";
    	}
	
	
	
    	if (tgtscrvec[m].size() > 0) {
	    
	    det_eer(tgtscrvec[m], nontgtscrvec[m], 0, &eer);
	    //		eer = det_eer(det);
	    lastp = 1.0;
	    last_r = 0.0;
	    
	    auc = 0;
	    correct = 0;
	    total = 0;
	    //@rarray = ();
	    //@parray = ();
	    rarray.clear();
	    parray.clear();
	    denom = tgtscrvec[m].size();
	    
	    sort( vi.begin(), vi.end(), gt(llrscore[m]) );

	    
	    for (vit = vi.begin(); vit != vi.end(); vit++) {
			
		total++;
		if (partitions[part][*vit].tid == m) {
		    correct++;
		    recall = (double) correct / (double) denom;
		    prec = (double) correct / (double)total;
		    rarray.push_back(recall);
		    parray.push_back(prec);
					
		}
		
	    }
	    
	    maxp = 0;
	    //# smooth precision scores
	    for (x = parray.size()-1; x >= 0; x--) {
		if (parray[x] < maxp) {
		    parray[x] = maxp;
		}
		if (parray[x] > maxp) {
		    maxp = parray[x];
		}
	    }
	    
	    for (x = 0; x < parray.size(); x++) {
		recall = rarray[x];
		prec = parray[x];
		dr = recall - last_r;
		y = prec;
		
		auc += dr * y;
		auc += (lastp - y) * dr /2; // # trapezoid rule
		last_r = recall;
		lastp = prec;
	    }
	    
	    /*if (defined($prdata)) {	    
	      for ($x=0; $x < scalar(@parray); $x++) {
	      print PRDATA "$rarray[$x] $parray[$x]\n";
	      }
	      close PRDATA;
	      }*/
	    
	    if (output) {
		//close OUTPUT;
	    }
	    
	}
	else {
	    auc = -1.0;
	    eer = 1.0;
	}
	test_results[ridx++] = auc;
	test_results[ridx++] = eer;
	
	//exit(1);
	
    }
    
    //if (prdata) {	    
    /*open PNG, "| gnuplot > $prdata.$N.png";
		
      print PNG "set terminal png\n";
      print PNG "set xlabel 'Recall';\n";
      print PNG "set ylabel 'Precision';\n";
      print PNG "set yrange [0:1.01]\n";
      print PNG "plot ";
      @ks2 = sort(keys(%topics));
      @ks = ();
      foreach $k (@ks2) {
      if (scalar(@{$tgtscra{$k}}) > 0) {
      push @ks, $k;
      }
      }
      for ($x=0; $x < scalar(@ks); $x++) {
      printf PNG "'$prdata.$ks[$x]' using 1:2 title '$ks[$x]' with lines %d", $x+1;
      if ($x+1 < scalar(@ks)) {
      printf PNG ",\\\n";
      }
      }
      print PNG ";\n";
      close PNG;
    */
    //}
    
    //  #  return ($auc, $eer, $errors / scalar(@testdocs));
    return (double)	errors / (double) counts[part];
    
}


int main(int argc, char ** argv) {
    int i,j,k;
    vector<double> vempty;
    
    if (parse_args(argc, argv)) {
	cerr << "Error parsing args: " << argerr << "\n";
	exit(1);
    }
    
    if (K <= 0) K = 1;
    
    FILE * fp;
    string word, val, line;
    
    int dataval;

    int len;
    int wid;
    ifstream inf;
    
    if (stoplist) {
	read_set(stoplist, stopwords, &word_id);
	
    }
    
    if (vocabfile) {
	read_set(vocabfile, vocab, &word_id);
    }
    
    
    if (tstoplist) 
	read_set(tstoplist, tstopwords, &word_id);
    
    
    if (weightFile) 
	read_map(weightFile, weights, &word_id);
    
    
    if (!train) {
    	cerr << "Must specify either training corpus or existing model\n";
	exit(1);
    }
    
    int ND = 0;
    inf.open(train);
    if (!inf.is_open()) {
	cerr << "Unable to open " << train << ".\n";
	exit(1);
    }
    
    string tp,utt;
    int cpos;
    int cp2;
    int slen;
    while(1) {
	getline(inf, line); 
	if (inf.eof()) break;
	vector<int> flist;	
// HERE
	if (debug) 
	    cerr << line << "\n";
	if ( line.size() < 1 ) continue;

	cpos = line.find('\t', 0);

	if (cpos == string::npos) continue;
	tp = line.substr(0, cpos);
	
	cp2 = line.find('\t', cpos+1);
	if (cp2 == string::npos) continue;

	utt = line.substr(cpos+1, cp2-(cpos+1));
	rawdata.push_back(line.substr(cp2+1));
	rawids.push_back(utt);

	dataval = rawdata.size()-1;

	// what if tp =~ /:/?

	if (strchr(tp.c_str(), ':')) {
	    //
	    vector<string> tlist;
	    split(tp, ':', tlist);
	    
	    for (int tt=0; tt < tlist.size(); tt++) {
		topicMap[tlist[tt]] = 1;
		if (tffiles.count(tlist[tt]) == 0) {
		    tffiles[tlist[tt]] = flist;
		}
		tffiles[tlist[tt]].push_back(dataval);
	    }

	    
	}
	else {
	    topicMap[tp] = 1;
	    if (tffiles.count(tp) == 0) {
		tffiles[tp] = flist;
	    }
	    tffiles[tp].push_back(dataval);
	}
	ND++;
    }
    inf.close();
    
    topicCount = topicMap.size();
    
    initialize_arrays(ND);
	
    map<string, int>::iterator ii;
    map<int, double>::iterator di;
    
    int m=0;
    for ( ii = topicMap.begin(); ii != topicMap.end(); ii++) {
	topics.push_back((*ii).first);
	(*ii).second = m++;
	}
    //	topicMap.clear();
    
    int p;
    
    
    for (p=0; p <= K; p++) {
	tprop[p]=vector<double>();//vempty;
	tprop[p].resize(topicCount, 0.0);
	
	tcount[p]=vector<double>();//vempty;;
	tcount[p].resize(topicCount, 0.0);
	
	wcount[p]=vector<double>();//vempty;;
	wcount[p].resize(topicCount, 0.0);
	
	vcount[p] = empty;
	
	corpcount[p] = 0;
	
    }
    
    counts = new int[K];
    memset(counts, 0, sizeof(int) * K);
    tr_info	 pinfo;
    
    // this partitioning strategy works if single labels
    for (i=0; i < topicCount; i++) {
	tp = topics[i];
	for (j=0; j < tffiles[tp].size(); j++) {
	    p = j % K;
	    
	    tcount[p][i]++;
	    tcount[K][i]++;
	    
	    pinfo.tf = &(rawdata[tffiles[tp][j]]);
	    //fprintf(stderr, "LINE %s\n", pinfo.tf->c_str());
	    pinfo.id = rawids[tffiles[tp][j]];
	    pinfo.label = topics[i];
	    pinfo.index = j;
	    pinfo.tid = i;
	    partitions[p].push_back(pinfo);
	    
	    counts[p]++;
	    
	}
    }
    
    
    //# compute LM data over partitions
    tr_info *info;
    string topic;
    int index;
    double count;
    
    int ah=0;
    
    map<int, map<int, double> > empty1;
    
    // initialize docCount
    for (i=0; i < topicCount; i++) {
	docCount.push_back(empty1);
    }

    cerr << "Reading train" << "\n";    
    vector<string> parts;
    for (i=0; i < K; i++) {
	for (j=0; j < counts[i]; j++) {
	    info = &(partitions[i][j]);
	    topic = info->label;
	    index = info->index;
	    

	    // READ TF values
	    //inf.open(info->tf.c_str());
	    //if (!inf.is_open()) {
	    //	cerr << "Unable to open " << info->tf << "\n";
	    //	exit(1);
	    //}
	    ss.clear();
	    ss.str(info->tf->c_str());
	    while (1) {
		if (ss.eof()) break;
		
		ss >> word;
		split(word, ':', parts);
		word = parts[0];
		count = strtod(parts[1].c_str(), 0);
		
		//fprintf(stderr, "FOUND %s %s %s\n", info->id.c_str(), word.c_str(), parts[1].c_str());
		//inf >> word;
		//inf >> count; 
		
		if (bernoulli) count = 1;
		
		if (ss.eof()) break;
		
		wid = word_id(word);
		
		if (stopwords.find(wid) != stopwords.end()) continue;
		
		if (vocabfile && vocab.find(wid) == vocab.end()) continue;
		
		if (docCount[info->tid].count(index) == 0)
		    docCount[info->tid].insert(mpair(index, map<int,double>()));// = empty;
		
		docCount[info->tid][index][wid] = count;
		
		for (k=0; k < K; k++) {
		    if (k != i) {
			
			if (topcount[k].count(wid) == 0) {
			    topcount[k][wid] = vector<double>();
			      topcount[k][wid].resize(topicCount, 0.0);
			}
			
			
			topcount[k][wid][info->tid] += count;
			wcount[k][info->tid] += count;
			vcount[k][wid] += count;
			corpcount[k] += count;
		    }
		}
		
		
		
		
		if (topcount[K].count(wid) == 0) {
		    topcount[K][wid] = vector<double>();
		    topcount[K][wid].resize(topicCount, 0.0);
		}
		
		
		topcount[K][wid][info->tid] += count;
		wcount[K][info->tid] += count;
		vcount[K][wid] += count;
		corpcount[K] += count;
		
		//if (!strcmp(topic.c_str(),"Education") && !strcmp(word.c_str(), "ah")) {
		//	printf ("T %f %d\n", 				topcount[K][wid][info->tid], ah++);
		//}
		
		
	    }	
	    //inf.close();
	    
	    
	}
	
    }
    
    // #there are many better smoothing methods, does it matter?
    
    if (output) {
	outp = fopen(output, "w");
	if (outp == NULL) {
	    fprintf(stderr, "Unable to open output file: %s\n", output);
	    output = NULL;
	}
    }
    
    map<int, vector<double> >::iterator mit;
    
    for (i=0; i < K; i++) {
	//@{$vocab[$i]} = keys(%{$topcount[$i]});
	vocabSize[i] = topcount[i].size();
	
	//foreach word (@{$vocab[$i]}) {
	for (mit = topcount[i].begin(); mit != topcount[i].end(); mit++) {
	    wid = (*mit).first;
	    prior[i][wid] = (vcount[i][wid] + 1) / (corpcount[i] + vocabSize[i]);
	}
	
	for (j=0; j < topicCount; j++) {  //foreach topic (keys(%topics)) {
	    //topic = topics[j];
	    if (unbalanced) {
		tprop[i][j] = wcount[i][j] / corpcount[i];
	    }
	    else {
		tprop[i][j] = 1;
	    }
	    
	}
	
    }
    
    vocabSize[K] = topcount[K].size();
    
    for (mit = topcount[K].begin(); mit != topcount[K].end(); mit++) {
	wid= (*mit).first;
	prior[K][wid] = (vcount[K][wid] + 1) / (corpcount[K] + vocabSize[K]);
	
	// start at 1
    }
    
    // # starting point
    double *test_results = new double[topicCount*2]; 
    double error, auc, eer, loss;
    
    
    int *tdocCount = new int[topicCount];
    
    double aauc, aeer;
    
    int ttestcount = 0;
    
    if (K > 1) {
	double *avg_tr = new double[topicCount * 2];
	double avg_err = 0;
	for (j=0; j < topicCount*2; j++) {  //foreach topic (keys(%topics)) {
	    avg_tr[j]= 0;
	}
	
	
	if (trainscores.size() != ND) {
	    trainscores.resize(ND);
	}
	docx = 0;
	for (j=0; j < K; j++) {
	    error = score_train(-1, j, prdata, output, test_results);
	    avg_err += error;
	    
	    for (m=0; m < topicCount*2; m+=2) {  //foreach topic (keys(%topics)) {
		if (test_results[m] < 0) {
		    avg_tr[m] = -1;
		    continue;
		}
		
		avg_tr[m] += test_results[m];
		avg_tr[m+1] += test_results[m+1];
	    }
	    
	}
	
	
	for (m=0; m < topicCount; m++) {
	    mus[m] /= ND;	
	}
	
	for (docx=0; docx < ND; docx++) {		
	    for (m=0; m < topicCount; m++) {
		sigmas[m]+= (mus[m] - trainscores[docx][m])*(mus[m] - trainscores[docx][m]) / (ND-1);
	    }
	}
	
	
	printf("Initial n error");
	for (j=0; j < topicCount; j++) {  //foreach topic (keys(%topics)) {
	    topic = topics[j];
	    auc = avg_tr[j*2];
	    if (auc < 0 || 
		(ignoreW && 
		 !strcmp(ignoreW, topic.c_str()))) {
		continue;
	    }
	    
	    cout << " auc-" << topic << " eer-" << topic;
	    
	}
	cout <<" auc-avg eer-avg\n";
	ttestcount = topicCount;
	aauc  = 0;
	aeer = 0;
	printf("Train 0 %0.4f", avg_err / (double) K);
	for (j=0; j < topicCount; j++) {  //foreach topic (keys(%topics)) {
	    topic = topics[j];
	    auc = avg_tr[j*2] / K;
	    eer = avg_tr[j*2+1] / K;
	    if (auc < 0 || 
		(ignoreW && 
		 !strcmp(ignoreW, topic.c_str()))) {
		ttestcount--;
		continue;
	    }
	    printf(" %0.4f %0.4f", auc, eer);
		aauc += auc;
		aeer += eer;
	}
	printf(" %0.4f %0.4f", aauc / (double) ttestcount , aeer / (double) ttestcount);
	
	printf("\n");
	
    }
    
    if (test) {
	
	error = score_test(-1, test, prdata, output, test_results);
	ttestcount = topicCount;
	
	if (K == 1) {	
	    printf("Initial n error");
	    for (j=0; j < topicCount; j++) {  //foreach topic (keys(%topics)) {
		topic = topics[j];
		auc = test_results[j*2];
		if (auc < 0 || 
		    (ignoreW && 
		     !strcmp(ignoreW, topic.c_str()))) {
		    continue;
		}
		cout << " auc-" << topic << " eer-" << topic;
		
	    }
	    cout <<" auc-avg eer-avg\n";
	}
	aauc  = 0;
	aeer = 0;
	printf("Test 0 %0.4f", error);
	for (j=0; j < topicCount; j++) {  //foreach topic (keys(%topics)) {
	    topic = topics[j];
	    auc = test_results[j*2];
	    eer = test_results[j*2+1];
		
	    if (auc < 0 || 
		(ignoreW && 
		 !strcmp(ignoreW, topic.c_str()))) {
		ttestcount--;
		continue;
	    }
	    printf(" %0.4f %0.4f", auc, eer);
	    aauc += auc;
	    aeer += eer;
	}
	
	printf(" %0.4f %0.4f", aauc / (double) ttestcount , aeer / (double) ttestcount);
	
	printf("\n");
	    
	double microF = (2*(double)tpsum) / 
	    (2*(double)tpsum + (double)fpsum + (double)fnsum);
	
	printf ("MicroF %0.4f MacroF %0.4f\n", microF, fmSum / (double) ttestcount);
	
    }
    
    
    
    //	printf("runtime score %0.4f classify %0.4f\n", scoretime / (double)iterations, getllr / ((double) iterations*(double)K));
    
    //# model
    // write final weights ? and model?
    
    FILE *mfp = 0;
    if (model) {
	mfp = fopen(model, "w");
	fprintf(mfp, "Word VocabSize");
	for (i=0; i < topicCount; i++) 
	    fprintf(mfp, " %s", topics[i].c_str());
	
	fprintf(mfp, "\n");
	/*if (K > 0) {
	  fprintf(mfp, "ZNorm Mean");
	  for (i=0; i < topicCount; i++) 
	  fprintf(mfp, " %g", zmean[i]);
	  fprintf(mfp, "\n");
	  
	  fprintf(mfp, "ZNorm SD");
	  for (i=0; i < topicCount; i++) 
	  fprintf(mfp, " %g", zsd[i]);
	  fprintf(mfp, "\n");
	  }*/
	
	fprintf(mfp, "%g %g", corpcount[K], vocabSize[K]);
	for (i=0; i < topicCount; i++) 
	    fprintf(mfp, " %g", wcount[K][i]);
	fprintf(mfp, "\n");
	
	
	for (mit = topcount[K].begin(); mit != topcount[K].end(); mit++) {
	    wid = mit->first;
	    word = wvec[wid];
	    fprintf(mfp, "%s %g", word.c_str(), vcount[K][wid]);
	    for (i=0; i < topicCount; i++) 
		fprintf(mfp, " %g", topcount[K][wid][i]);
	    fprintf(mfp, "\n");
	    
	    
	}
	fclose(mfp);
    }
    
    delete_arrays(ND);
    
    delete [] counts;
    
    delete [] test_results; 
    
    empty.clear();
    vempty.clear();
    
}
