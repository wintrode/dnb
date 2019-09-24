#define HAVE_CXX_STDHEADERS

#include <iostream>
#include <fstream>
#include <sstream>

#include <stdio.h>
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

double diffclock(clock_t clock1,clock_t clock2)
{
    double diffticks=clock1-clock2;
    double diffms=(diffticks*10)/CLOCKS_PER_SEC;
    return diffms;
}


using namespace std;

typedef struct _token_t {
    int wid;
    double weight;
    int location;
} token_t;



typedef pair<int, map< int, double> > mpair;
typedef pair<int, vector<double> > vpair;
typedef pair<int, vector<token_t> > tpair;


double zero = 1e-15;
int maxWordId = 0;
map<string,int> wordids;

/*typedef struct _train_info {
    string tf;
    string label;
    int index;
    int tid;
} tr_info;
*/

map<int, vector<double> > *topcount;

vector<double> *tprop;
vector<double> *tcount;
map<int,double> *prior;
vector<double> *wcount;
map<int,double> *vcount;

double *corpcount;
double *vocabSize;

vector<string> topics;


vector<tr_info> *partitions;

double *llrs;
double *parray;

//map<string, int> topicMap;
//map<string, vector<string> > tffiles;

corpus trainCorp;
set<int> stopwords;

vector<map<int,double> > testdocs;
vector<vector<token_t> > testseqs;

vector< map<int,double> >  *fcache;

map<int,double> empty;

#define MCE 0
#define L1 1
#define L2 2
#define TPL 3
#define PER_TOPIC 4

enum {NONE, EXP, LIN, GAUSS};

int lfunc = MCE;

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
    {"initial-weights", 1, 0, 'I'},
    {"topic-weights", 1, 0, '1'},
    {"target", 1, 0, 'T'},
    {"iterations", 1, 0, 'i'},
    {"K", 1, 0, 'K'},
    {"debug", 1, 0, 'd'},
    {"start", 1, 0, 'R'},
    {"end", 1, 0, 'N'},
    {"beta", 1, 0, 'b'},
    {"epsilon", 1, 0, 'e'},
    {"lambda", 1, 0, 'L'},
    {"pr-freq", 1, 0, 'F'},
    {"word-classes", 1, 0, 'W'},
    {"class-priors", 1, 0, 'c'},
    {"znorm", 1, 0, 'z'},
    {"znorm-train", 1, 0, 'Z'},
    {"prune", 1, 0, 'P'},
    {"loss-func", 1, 0, 'l'},
    {"decay", 1, 0, 'D'},
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
char *TweightFile = 0 ;

char *initialWeightFile = 0 ;
char *wordclasses= 0;
char *target = 0 ;
char *stoplist = 0;

double start = 0.0;
double endPos = 1.0;

int iterations;
int K;
double epsilon = 0.1;
double beta = 1;
double lambda = 2;
int prFreq = 1;
int debug = 0;

int classPriors  = 0;
double prune = -1;
string argerr;
int znorm = 0;
int znormTrain = 0;

int singleMax = 1;

int wfunc = EXP;

FILE *outp = 0;

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
  	parray = new double[topicCount];

  	
	fcache = new vector< map<int,double> >[K+1];

}

void delete_arrays(int ND) {
    int i;
    
    int j;
    
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
    delete[] parray;
    
    delete [] fcache;
    
}


int parse_args(int argc, char **argv) {

    int idx,opt;
    
    while ((opt = getopt_long(argc,argv, "t:s:w:p:o:f:T:i:K:b:e:F:W:S:ducP:zl:I:B1:L:m:R:N:D:", opts, &idx)) != -1) {
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
	case '1':
	    TweightFile = optarg; break;
	case 'I':
	    initialWeightFile = optarg; break;
	case 'T':
	    target = optarg; break;
	case 'i':
	    iterations = atoi(optarg); break;
	case 'K':
	    K = atoi(optarg); break;
	case 'b':
	    beta = strtod(optarg, 0); break;
	case 'P':
	    prune = strtod(optarg, 0); break;
	case 'e':
	    epsilon = strtod(optarg, 0); break;
	case 'L':
	    lambda = strtod(optarg, 0); break;
	case 'R':
	    start = strtod(optarg, 0); break;
	case 'N':
	    endPos = strtod(optarg, 0); break;
	case 'F':
	    prFreq = atoi(optarg); break;
	case 'W':
	    wordclasses = optarg; break;
	case 'S':
	    stoplist = optarg; break;
	case 'd':
	    debug = 1; break;
	case 'c':
	    classPriors = 1; break;
	case 'z':
	    znorm = 1; break;
	case 'Z':
	    znormTrain = 1; break;
	case 'D':
	    wfunc = atoi(optarg);
	    break;
	case 'l':
	    if (optarg && optarg[0] == '0')
		lfunc = MCE;
	    else if (optarg && optarg[0] == '1')
		lfunc = L1;
	    else if (optarg && optarg[0] == '2')
		lfunc = L2;
	    else if (optarg && optarg[0] == '3')
		lfunc = TPL;
	    else if (optarg && optarg[0] == '4')
		lfunc = PER_TOPIC;
	    else
		lfunc = L1;
	    break;
	default:
	    argerr = "Unknown argument\n";
	    return 1;	
	}
	
    }
    return 0;
}


vector<string> wvocab;

inline int word_id(string &word) {
	map<string,int>::iterator iit;

	iit = wordids.find(word);
	if (iit == wordids.end()) {
		wordids[word] = maxWordId;
		wvocab.push_back(word);
		return maxWordId++;
	}
	else 
		return (*iit).second;
}

inline double calcF(int part, int topic, int word) {
	int i;
	map< int, vector<double> >::iterator jj;
	double p, q, pr,d;
	q = 0;
	
	if (prior[part].count(word) == 0) {
	    pr = prior[part][word] = 1 / (corpcount[part] + vocabSize[part]);
	}
	else {
	    pr = prior[part][word];
	}
	
	
	for (i=0; i < topicCount; i++) { // tp (keys(%topics)) {
	    //tp = topics[i];
	    parray[i]=0.0;
	    
	    
		// what if topcount == 0
	    jj = topcount[part].find(word);
	    if (jj == topcount[part].end()) 
		d = 0.0;
		else 
		    d = (*jj).second[i];
	    
	    p = (d + vocabSize[part] * pr) /
		(wcount[part][i] + vocabSize[part]);
	    if (p == 0.0) {
		p = zero;
		cerr << "Canot have a zero value in log prob calc.\n";
	    }
	    
	    q += p;
	    parray[i] = p;	
	}
	
	double f,r ;		
	// # compute log(p(w|!t)) over all topics
	for (i=0; i < topicCount; i++) { // tp (keys(%topics)) {
	
		d = (q - parray[i]) / (topicCount-1);
		if (d == 0) {
			d = zero;
		}

		f = log(parray[i]) - log(d);	
		
		fcache[part][i][word] = f;

		if (f == topic)
			r = f;
	
	}
	
	return r;
	
}

double get_llr(vector<token_t> *tdoc, int topic, int part, double *results) {
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
	map<int, double>::iterator v;
	int x;
	map<int, vector<double> >::iterator jj;

	double p,q,logp,logq,d,f,qp,pp;
	string tp,t2;
	
	double pr;
	  
	token_t linfo;
	
	int startidx = start * tdoc->size();
	int endidx = endPos * tdoc->size();

	double wsum = 0;
	double tmp; 
	for (x=0; x < tdoc->size(); x++) {

	      linfo = (*tdoc)[x];
	      word = linfo.wid;   // (sort(keys(%tdoc))) {
			 
	      if (linfo.location < startidx || linfo.location >= endidx) {
		   
		continue;
	      }
	      if (weights.count(word) > 0) {
		  wt = weights[word];
		  
		  if (wfunc == EXP) 
		      wt = exp ( -wt * ( double) linfo.location / (double)tdoc->size());
		  else if (wfunc == LIN) {
		      tmp = (double) linfo.location / (double)tdoc->size();
		      tmp = (1 - wt * tmp);
		      if (tmp < 0) tmp = 0;
		      wt = tmp;
		  }
		  //		  else if (wfunc == INV) 
		  //  wt = 1 / (wt *
		  ///		( ((double) linfo.location / (double)tdoc->size()) + (1/wt)));
		  else if (wfunc == GAUSS)  {
		      tmp = (double) linfo.location / (double)tdoc->size();
		      wt = exp (- wt * wt * tmp * tmp / 2.0);
		  }
		  else if (wfunc == NONE) {
		      //wt = 
		      // no op
		  }
		  else {
		      cerr << "Unexpected weight function.  Exiting.";
		      exit(1);
		  }
		      
		  
		  
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
	      
	      
	      if (topic < 0) {
		  //	# compute log(p(w|t)) over all topics
		  
		  q = 0;
		  for (i=0; i < topicCount; i++) { // tp (keys(%topics)) {
		      //tp = topics[i];
		      parray[i]=0.0;
		      
		      v = fcache[part][i].find(word);				
		      if (v != fcache[part][i].end()) {
			results[i] += wt * linfo.weight * (*v).second;// /(double) tdoc->size();;
			  continue;
		      }
		      
		      // what if topcount == 0
		      jj = topcount[part].find(word);
		      if (jj == topcount[part].end()) 
			  d = 0.0;
		      else 
			  d = (*jj).second[i];
		      
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
		      
		      //tp = topics[i];
		      //if (parray[i] == 0.0) {
		      if (fcache[part][i].count(word) > 0) {
			  continue;					
		      }
		      
		      d = (q - parray[i]) / (topicCount-1);
		      if (d == 0) {
			  d = zero;
		      }
		      
		      f = log(parray[i]) - log(d);	

		      fcache[part][i][word] = f;
		      
		      results[i] += wt * linfo.weight * f;// /(double) tdoc->size();;
		      
		  }
	
	      }
	      
	}


	
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
		results[i] +=	(log(pp) - log(q));
	    }
	}
	
	if (topic >= 0) {
	    return results[topic];
	}
	else {
	    return 0.0;
	}
	
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


inline int read_testdocs(char *test) {
    ifstream tfin;
    ifstream docin;
    string tfdata,word,topic,line;
    double count;
    tfin.open(test);
    
    map<int, double> doc;
    vector<token_t> dsq;
    
    token_t linfo;
    
    int x = 0;
    int wid;
    int location;
    int doclen = 1;
    string utt;
    vector<string> parts;
    int cpos, cp2;

    cerr << "Reading test...";
    stringstream ss (stringstream::in | stringstream::out);
    
    while (1)  {
	getline(tfin, line); 
	if (tfin.eof()) break;
	vector<int> flist;	
	// HERE
	if (debug) 
	    cerr << line << "\n";

	std::cerr << "WORD " << line << "\n";
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
		
	testdocs.push_back(doc);
	testseqs.push_back(dsq);

	ss.clear();
	ss.str(tfdata);

	while (1) {
	    if (ss.eof()) break;
	    
	    ss >> word;
	    std::cerr << "WORD " << word << "\n";
	    split(word, ':', parts);
	    word = parts[0];
	    count = strtod(parts[1].c_str(), 0);

	    if (!strcmp(word.c_str(), "#DOCLEN")) {
		doclen = atoi(parts[1].c_str());
		continue;
	    }
	    
	    if (parts.size()>2)
		location = atoi(parts[2].c_str());
	    else 
		location = 1;


	    //docin.open(tffile.c_str());
	    if (x + 1 % 50 == 0) {
		cerr << x << "...";
	    }


	    //docin >> doclen;

	    /*while (!docin.eof()) {
		docin >> word;
		if (docin.eof()) break;
		docin >> count;
		if (docin.eof()) break;
		docin >> location;
		if (docin.eof()) break;*/
	    
	    wid = word_id(word);

	    
	    if (stopwords.find(wid) != stopwords.end() ) continue;
	    
	    //	if (defined($wclasses[word])) {
	    //    word = $wclasses[word];
	    //}
	    if (testdocs[x].count(wid)==0)
		testdocs[x][wid]=0;
	    
	    linfo.wid = wid;
	    linfo.location = location;
	    linfo.weight = count;
	    
	    count =  count * exp(-lambda * (double)location / (double)doclen);
	    
	    testdocs[x][wid] += count;
	    testseqs[x].push_back(linfo);
	}	
	docin.close();
	
	truth.push_back(trainCorp.topicMap[topic]);
	truthMap.push_back(map<int,int>());
	for (int tt=0; tt < tlist.size(); tt++) {
	    truthMap[x][trainCorp.topicMap[tlist[tt]]] = 1;
	}
	
	x++;
    }	
    tfin.close();
    cerr << "done.\n";
}

vector < vector <double> > trainscores;

int docx = 0;

vector<double> mus;
vector<double> sigmas;

vector<double> mu;
vector<double> sigma;


double score_test(int N, char *test, char *prdata, char *output, double* test_results, double *test_loss) {

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
    vector<token_t> *testseq;
    
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
    

    *test_loss = 0;
    
    if (lfunc != MCE) {
	for (k=0; k < topicCount; k++) {
	    maxLLRs[k] = -9999;
	    maxtopics[k] = 0;
	    maxindexs[k] = 0;
	    
	    for (m=0; m < topicCount; m++) {
		maxLLR2[k][m] = -9999;
		maxindex2[k][m] = 0;
	    }
	}
    }
    
    for (x=0; x <testdocs.size(); x++) {
    	testdoc = &(testdocs[x]);
    	testseq = &(testseqs[x]);

	max = -9999;
	maxtgt = m;
	
	memset(llrs, 0, sizeof(double)*topicCount);
	
	// classification
	get_llr(testseq, -1, K, llrs);
	
	tid = truth[x];

	if (output)
	    fprintf(outp, "test %d %s", x, topics[truth[x]].c_str());


	maxLLR = -9999;
	for (m=0; m < topicCount; m++) {

	    if (znorm)
		llrs[m] = (llrs[m] - mus[m]) / sigmas[m];

	    if (output)
		fprintf(outp, " %0.6f", llrs[m]);

	    
	    if (lfunc != MCE) {//#ifndef ORIGINAL_LOSS
		if (m == tid) {
		    targetLLRs[x] = llrs[m];
		}
		else {
		    // actual topic is tid
		    // llr is score for topic tid against classifier m
		    
		    if (llrs[m] > maxLLRs[m]) {
			maxLLRs[m] = llrs[m];
			maxindexs[m] = x;
			maxtopics[m] = tid;
		    }
		    if (llrs[m] > maxLLR2[m][tid]) {
			maxLLR2[m][tid] = llrs[m];
			maxindex2[m][tid] = x;
		    }
			
		}
	    }
	    else {
		if (m == tid) {
		    targetLLR = llrs[m];
		}
		else {
		    if (llrs[m] > maxLLR) {
			maxLLR = llrs[m];
			maxtopic = m;
		    }
		}
	    }
	}
	
	if (lfunc == MCE) { // one pass through documents is enough			    

	    lw = 1 / (1 + exp(-beta * 
			      ( maxLLR - targetLLR)));

	    //cerr << "MCE " << lw << "\n";	    
	    *test_loss += lw;
	    A++;
	    
	}

	if (output)
	    fprintf(outp, "\n");
	
	
	m=0;
	//	eval.update(llrs, truthMap[x]);

	// compute argmax
	for (m=0; m < topicCount; m++) {
	    llrscore[m].push_back(llrs[m]);
	    
	    if (llrs[m] > max) {	
		max = llrs[m];
		maxtgt = m;
	    }
	    if (output) {
		//cout << topics[m] <<  " " << llrs[m] << " ";
		//cout << " " <<  llrs[m] << " ";
	    }

	}
			
	for (m=0; m < topicCount; m++) {
	    if (m == truth[x] || truthMap[x].find(m) != truthMap[x].end()) {
		tgtscrvec[m].push_back(llrs[m]);
		if (m == maxtgt)
		    tp[m]++;
		else
		    fn[m]++;
	    }
	    else {
		nontgtscrvec[m].push_back(llrs[m]);
		if (m == maxtgt)
		    fp[m]++;
	    }
	    
	    
	}

	if ( maxtgt != truth[x] && truthMap[x].find(maxtgt) == truthMap[x].end()) {
	    errors++;
	}
	
	if (output) {
	    //print OUTPUT "$maxtgt\n";
	    //cout << topics[maxtgt] << "\n";
	}
	
	vi.push_back(x);
    }

    int j=0;

    if (lfunc != MCE) {
	for (m=0; m < topicCount; m++) {
	    sumLLR[m] = 0;
	    for (j=0; j < topicCount; j++) {
		if (j == m) continue;
		sumLLR[m] += maxLLR2[m][j];
	    }
	}
	
	
	for (x=0; x <testdocs.size(); x++) {
	    testdoc = &(testdocs[x]);
			
	    tid = truth[x];
	    
	    //loss is highest incorrect score for this topic - this doc score
	    if (lfunc == L1)
		lw = 1 / (1 + exp(-beta * 
				  ( maxLLRs[tid] - targetLLRs[x])));
	    else 				// loss is sum (or average) of highest incorrect scores for this topic (1 from each other topic)
		lw = 1 / (1 + exp(-beta * 
				  ( sumLLR[tid]/ (topicCount-1) - targetLLRs[x])));
	    
	    
	    *test_loss += lw;
	    A++;
	}
    }
    
    //    results = ();
    int ridx = 0;
    //    test_results[ridx++] = errors / testdocs.size();
    
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
	    
	    
	    //    cout << topics[m] << " F1=" << fmeas << " TP=" << tp[m] << " FP=" << fp[m] <<  " FN=" << fn[m] << " " << f1prec  << " " << f1rec <<"\n";
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
    
    *test_loss /= (double)A;

    return (double)	errors / (double) testdocs.size();
   
}


int main(int argc, char ** argv) {
    int i,j,k;
    vector<double> vempty;
    
    fprintf(stderr, "# ARGS:");
    for (i=0; i < argc; i++) {
	fprintf(stderr, " %s", argv[i]);
    }
    fprintf(stderr, "\n");

    if (parse_args(argc, argv)) {
	cerr << "Error parsing args: " << argerr << "\n";
	exit(1);
    }
    
    
    if (K <=0) K=1;
    FILE * fp;
    string word, val, line;
	
    int len;
    int wid;
    ifstream inf;
    
    if (stoplist) {
	//fp = fopen(stoplist, "r");
	read_set(stoplist, stopwords, &word_id);
    }
    
    if (initialWeightFile) 
	read_map(initialWeightFile, weights, &word_id);
        
    if (!train) {
    	cerr << "Must specify either training corpus or existing model\n";
	exit(1);
    }

    /*int ND = 0;
    inf.open(train);
    if (!inf.is_open()) {
	cerr << "Unable to open " << train << ".\n";
	exit(1);
    }


    string tp;
    while(!inf.eof()) {
	inf >> tp;
	if (inf.eof()) break;
	inf >> val;
	if (inf.eof()) break;

	vector<string> flist;	
	// HERE

	if (strchr(tp.c_str(), ':')) {
	    //
	    vector<string> tlist;
	    split(tp, ':', tlist);
	    
	    for (int tt=0; tt < tlist.size(); tt++) {
		topicMap[tlist[tt]] = 1;
		if (tffiles.count(tlist[tt]) == 0) {
		    tffiles[tlist[tt]] = flist;
		}
		tffiles[tlist[tt]].push_back(val);
	    }
		    
		    
	}
	else {
	    topicMap[tp] = 1;
	    if (tffiles.count(tp) == 0) {
		tffiles[tp] = flist;
	    }
	    tffiles[tp].push_back(val);
	}

	ND++;
    }
    inf.close();
    */
    int ND = read_corpus(train, &trainCorp);
    topicCount = trainCorp.topicMap.size();

    initialize_arrays(ND);
	
    string tp;

    map<string, int>::iterator ii;
    map<int, double>::iterator di;

    int m=0;
    for ( ii = trainCorp.topicMap.begin(); ii != trainCorp.topicMap.end(); ii++) {
	topics.push_back((*ii).first);
	(*ii).second = m++;
    }

    int p;
    
    // partition data
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
    
    int *counts = new int[K];
    memset(counts, 0, sizeof(int) * K);
    tr_info	 pinfo;
    
    for (i=0; i < topicCount; i++) {
	tp = topics[i];
	for (j=0; j < trainCorp.tffiles[tp].size(); j++) {
	    p = j % K;
	    
	    tcount[p][i]++;
	    tcount[K][i]++;
	    
	    pinfo.tf = &(trainCorp.rawdata[trainCorp.tffiles[tp][j]]);
	    //fprintf(stderr, "LINE %s\n", pinfo.tf->c_str());
	    pinfo.id = trainCorp.rawids[trainCorp.tffiles[tp][j]];
	    pinfo.label = topics[i];
	    pinfo.index = j;
	    pinfo.tid = i;
	    partitions[p].push_back(pinfo);
	    
	    counts[p]++;
	}
    }
    
    map<string*,int> seen;
    
//# compute LM data over partitions
    tr_info *info;
    string topic;
    int index;
    double count;
    int location;
    
    token_t linfo;
    
    int ah=0;
    
    map<int, map<int, double> > empty1;
    map<int, vector<token_t> > empty2;
    
    
    vector< map<int, map<int, double> > > docCount;
    vector< map<int, vector<token_t> > > docSeq;
    
    // initialize docCount
    for (i=0; i < topicCount; i++) {
	docCount.push_back(empty1);
	docSeq.push_back(empty2);
    }

    int doclen = 1;

    stringstream ss (stringstream::in | stringstream::out);
    vector<string> parts;
    
    cerr << "Reading train"  << "...";
    int idx = 0;
    for (i=0; i < K; i++) {
	for (j=0; j < counts[i]; j++) {
	    info = &(partitions[i][j]);
	    topic = info->label;
	    index = info->index;
	    
	    idx++; 
	    if (idx % 1000 == 0) 
		cerr << idx << "...";

	    
	    ss.clear();
	    ss.str(info->tf->c_str());
	    while (1) {
		if (ss.eof()) break;
		
		ss >> word;
		split(word, ':', parts);
		word = parts[0];
		if (!strcmp(word.c_str(), "#DOCLEN")) {
		    doclen = atoi(parts[1].c_str());
		    continue;
		}

		count = strtod(parts[1].c_str(), 0);
		if (parts.size()>2)
		    location = atoi(parts[2].c_str());
		else 
		    location = 1;
		
		//fprintf(stderr, "FOUND %s %s %s\n", info->id.c_str(), word.c_str(), parts[1].c_str());
		//inf >> word;
		//inf >> count; 

		/*inf.open(info->tf.c_str());
	    
		if (!inf.is_open()) {
		    cerr << "Unable to open " << info->tf << "\n";
		    exit(1);
		}
		
		inf >> doclen;
		
		while (!inf.eof()) {
		
		    inf >> word;
		    if (inf.eof()) break;
		    inf >> count;
		    if (inf.eof()) break;
		    inf >> location; 
		    
		    if (inf.eof()) break;*/
		
		wid = word_id(word);
		
		if (stopwords.find(wid) != stopwords.end()) continue;
		
		//	    if (defined($wclasses[word])) {
		//	word = $wclasses[word];
		//  }
		    
		
		    
		linfo.wid = wid;
		linfo.weight = count;
		linfo.location = location;
		
		
		if (docCount[info->tid].count(index) == 0) {
		    docCount[info->tid].insert(mpair(index, map<int,double>()));// = empty;
		    docSeq[info->tid].insert(tpair(index, vector<token_t>()));// = empty;
		}
		
		if (docCount[info->tid][index].count(wid) == 0)
		    docCount[info->tid][index][wid] = 0;
	       
		
		//count =  count * exp(-lambda * (double)location / (double)doclen);
		if (wfunc == EXP) 
		    count = count * exp ( -lambda * ( double) location / (double)doclen);
		else if (wfunc == LIN) {
		    double tmp = (double) location / (double)doclen;
		    tmp = 1 - (lambda * tmp);
		    if (tmp < 0) tmp = 0;
		    count *= tmp; 
		}
		else if (wfunc == GAUSS) { 
		    double p = ((double) location / (double)doclen); 
		    count *= exp(-(lambda * lambda) *  p * p / 2.0);
		    
		}
		else if (wfunc == NONE) {
		    count *= lambda; // no position
		}
		else {
		    cerr << "Unexpected weight function.  Exiting.";
		    exit(1);
		}
	

		docCount[info->tid][index][wid] += count;
		
		docSeq[info->tid][index].push_back(linfo);
		
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
		
		if (seen.find(info->tf) == seen.end()) {
		    // don't count documents twice....
		    vcount[K][wid] += count;
		    corpcount[K] += count;
		    
		    //seen[info->tf]=1;
		}
		
	    }
	    
	    inf.close();
	    
	    
	}
	
    }
    
    cerr << "done.\n";
    
    // #there are many better smoothing methods, does it matter?
    
    map<int, vector<double> >::iterator mit;
    
    for (i=0; i < K; i++) {
	vocabSize[i] = topcount[i].size();
	
	for (mit = topcount[i].begin(); mit != topcount[i].end(); mit++) {
			wid = (*mit).first;
			prior[i][wid] = (vcount[i][wid] + 1) / (corpcount[i] + vocabSize[i]);
	}
	
	for (j=0; j < topicCount; j++) {  //foreach topic (keys(%topics)) {
	    //topic = topics[j];
	    tprop[i][j] = 1;
	    
	}
	
    }
    
    //@{$vocab[$K]} = keys(%{$topcount[$K]});
    vocabSize[K] = topcount[K].size();
    
    for (mit = topcount[K].begin(); mit != topcount[K].end(); mit++) {
	wid= (*mit).first;
	prior[K][wid] = (vcount[K][wid] + 1) / (corpcount[K] + vocabSize[K]);
	
	// start at 1
	if (!initialWeightFile) {
	    weights[wid] = lambda;
	}
    }
    
    // # starting point
    double *test_results = new double[topicCount*2]; 
    double error, auc, eer, loss;
    
    //#ifndef ORIGINAL_LOSS
    maxLLRs = new double[topicCount];
    maxtopics = new int[topicCount];
    maxindexs= new int[topicCount];
    
    double *minLLRs = new double[topicCount];
    int *mintopics = new int[topicCount];
    int *minindexs= new int[topicCount];
    
	
    maxLLR2 = new double*[topicCount];
    maxindex2= new int*[topicCount];
    
    for (j = 0; j < topicCount; j++) {
	maxLLR2[j] = new double[topicCount];
	maxindex2[j] = new int[topicCount];
    }
    
    sumLLR = new double[topicCount];
    
    double *targetLoss = new double[topicCount];
    
    int *tdocCount = new int[topicCount];
    
    double **trainScores  = new double*[topicCount];
    for (i=0; i < topicCount; i++) {
	trainScores[i] = new double[ND];
    }
    if (lfunc != MCE) {
	targetLLRs = new double[ND];// / K + 1];
    }
    //#endif
    
    double aauc, aeer;
    double testLoss;
    int ttestcount = 0;
    
    mus.resize(topicCount);
    sigmas.resize(topicCount);
    mu.resize(topicCount);
    sigma.resize(topicCount);
    
    for (j=0; j < topicCount; j++) {  //foreach topic (keys(%topics)) {
	mus[j] = 0.0;
	sigmas[j] = 1.0;
    }
    
    if (output) 
	outp = fopen(output,"w");

    if (test) {
	cerr << "Scoring test...\n";
	error = score_test(-1, test, prdata, output, test_results, &testLoss);
	ttestcount = topicCount;
	cerr << "Done scoring test...\n";
	
	printf("Initial n error loss zeros test-loss");
	for (j=0; j < topicCount; j++) {  //foreach topic (keys(%topics)) {
	    topic = topics[j];
	    auc = test_results[j*2];
	    if (auc < 0) {
		continue;
	    }
	    cout << " auc-" << topic << " eer-" << topic;
	    
	}
	cout <<" auc-avg eer-avg\n";
	
	aauc = aeer = 0;
	
	printf("Initial 0 %0.4f 1 0 %0.4f", error, testLoss); 
	for (j=0; j < topicCount; j++) {  //foreach topic (keys(%topics)) {
	    topic = topics[j];
	    auc = test_results[j*2];
	    eer = test_results[j*2+1];
	    if (auc < 0) {
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
    
    map<int, vector<double> >::iterator jj;
    
    double d;
    
    int N,tid;
    double avgLoss,A, lw,v, oldweight;
    
    vector<double> partialSum;
    
    map<int, double> *doc;
    vector<token_t> *docsq;
    vector<token_t> *docsq2;
    map<int, double> *doc2;
    
    double getllr = 0.0;
    double scoretime = 0.0;
    double wtmp;
    
    partialSum.resize(weights.size(), 0.0);
    
    mus.resize(topicCount);
    sigmas.resize(topicCount);
    mu.resize(topicCount);
    sigma.resize(topicCount);
    
    double lastLoss = -1;
    int backwards = 0;
    for (N= 0; N < iterations; N++) {
	avgLoss = 0;
	A = 0;
	
	for(di = weights.begin(); di != weights.end(); di++) {
	    partialSum[(*di).first] = 0.0;
	}
	
	/* tmp disable znorm calcs for speed */
	
	if (trainscores.size() != ND) {
	    trainscores.resize(ND);
	}
	
	docx = 0;
	
	//# for each document, calculate loss
	for (i=0; i < K; i++) {
	    
	    j=0;
	    
	    //#max score of off-topic
	    //	
	    //#ifndef ORIGINAL_LOSS
	    if (lfunc != MCE) {
		for (k=0; k < topicCount; k++) {
		    maxLLRs[k] = -9999;
		    maxtopics[k] = 0;
		    maxindexs[k] = 0;
		    minLLRs[k] = 0;
		    mintopics[k] = 0;
		    minindexs[k] = 0;
		    
		    
		    if (lfunc == TPL || lfunc == PER_TOPIC) {
			if (lfunc == TPL) 
			    maxLLRs[k] = 0.0;
			tdocCount[k] = 0;
			for (m=0; m < topicCount; m++) {
			    maxLLR2[k][m] = 0;
			    //							maxindex2[k][m] = 0;
			}
		    }
		    else {
			for (m=0; m < topicCount; m++) {
			    maxLLR2[k][m] = -9999;
			    maxindex2[k][m] = 0;
			}	
		    }
		}
		
		
	    }
	    //#endif	
	    for (j = 0; j < counts[i]; j++) {
		
		tid = partitions[i][j].tid;
		tdocCount[tid]++;
		index = partitions[i][j].index;
		
		doc = &(docCount[tid][index]);
		docsq = &(docSeq[tid][index]);
		
		clock_t begin = clock();
				
		get_llr(docsq, -1, i, /*j,*/ llrs);
		
		clock_t end = clock();
		
		getllr += diffclock(end, begin);

		/* temp disable znorm for speed */
		
		if (trainscores[docx].size() != topicCount)
		    trainscores[docx].resize(topicCount);
		
		
		for (m=0; m < topicCount; m++) {
		    trainscores[docx][m] = llrs[m];
		    mus[m] += llrs[m];
		    if (N > 0 && znormTrain) 
			llrs[m] = (llrs[m] - mu[m] ) / sigma[m];
		}
		
		
		docx++;
		if (lfunc == TPL || lfunc == PER_TOPIC) {
		    for (m=0; m < topicCount; m++) {
			maxLLR2[tid][m] += llrs[m];
			//		maxLLRs[m] = 0;
			trainScores[m][j] = llrs[m];
		    }
		}
				
		maxLLR = -9999999;
		for (m=0; m < topicCount; m++) {
		    
		    if (lfunc != MCE && lfunc != TPL) {//#ifndef ORIGINAL_LOSS
			if (m == tid) {
			    targetLLRs[j] = llrs[m];
			}
			else {
			    // actual topic is tid
			    // llr is score for topic tid against classifier m
			    
			    if (llrs[m] > maxLLRs[m]) {
				maxLLRs[m] = llrs[m];
				maxtopics[m] = tid;
				maxindexs[m] = index;
			    }
			    if (llrs[m] > maxLLR2[m][tid]) {  // top off-topic score 
				maxLLR2[m][tid] = llrs[m];
				maxindex2[m][tid] = index;
			    }
			    
			}
			
			
		    }
		    else {
			if (m == tid) {
			    targetLLR = llrs[m];
			}
			else {
			    if (llrs[m] > maxLLR) {
				maxLLR = llrs[m];
				maxtopic = m;
			    }
			}
		    }
		}
		
		if (lfunc == MCE) { // one pass through documents is enough			    
		    //cerr << "MCE Loss...\n";
		    lw = 1 / (1 + exp(-beta * 
				      ( maxLLR - targetLLR)));
				    
		    
		    if (debug && docx % 10 == 1) {
			cerr << "Loss " << lw << " Max " << maxLLR << " corrr " << targetLLR << "\n";
		    }
		    
		    int idx;
		    token_t linfo;
		    for (idx = 0; idx < docsq->size(); idx++) {
			linfo = (*docsq)[idx];
			wid = linfo.wid;
			if (weights.count(wid)==0)
			    continue;
			wtmp = weights[wid];
			
			if (wfunc == EXP) {
			    partialSum[wid] +=
				lw * (1-lw) * linfo.weight *
				(- (double) linfo.location / (double)docsq->size()) *
				exp(- wtmp * (double) linfo.location / (double)docsq->size()) *
				(fcache[i][maxtopic][wid] - fcache[i][tid][wid]);
			}
			else if (wfunc == NONE) {
			    partialSum[wid] +=
				lw * (1-lw) * linfo.weight *
				(fcache[i][maxtopic][wid] - fcache[i][tid][wid]);
			}
			else if (wfunc == GAUSS) {
			    double p = (double) linfo.location / (double)docsq->size();
			    partialSum[wid] +=
				lw * (1-lw) * linfo.weight *
				exp(-wtmp*wtmp * p * p /2.0)*
				(-wtmp * p * p) * 
				(fcache[i][maxtopic][wid] - fcache[i][tid][wid]);
					    
			}
			
		    }
		    
		    // lw -- was this doc well recognized or poorly
		    
		    avgLoss += lw;
		    A++;
		    
		}
		
	    }
			
	    if (lfunc == MCE) continue;
	
	    if (lfunc == TPL) {
		lw = 0;
		for (m = 0; m < topicCount; m++) {
		    if (tdocCount[m] == 0) { // no docs in this topic for this partition
			continue;
		    }
		    targetLoss[m] = -maxLLR2[m][m] / (double) tdocCount[m]; 	
		    for (j=0; j < topicCount; j++) {
			if (j == m) continue;
			targetLoss[m] += maxLLR2[m][j] / (double) (counts[i] - tdocCount[m]);
			
		    }
		    lw += targetLoss[m];
				
		    
		}
		
		
		lw = 1 / (1 + exp(-beta * lw)); 
		
		avgLoss += lw;
		A++;
		
		for (m = 0; m < topicCount; m++) {
		    if (tdocCount[m] == 0) { // no docs in this topic for this partition
			continue;
		    }
		    
		    for (jj = topcount[i].begin(); jj != topcount[i].end(); jj++) {
			wid = (*jj).first;
			d = (*jj).second[m];
			
			partialSum[wid] += lw * (1 - lw) * fcache[i][m][wid] * 
			    (/*(corpcount[i] - d) / (double) (counts[i] - tdocCount[m]) */-
			     d / (double) tdocCount[m]);
		    }
		    
		}
		
		continue;
	    }
	    
			
	    j=0;
	    
	    // sum of off-topic max
	    for (m=0; m < topicCount; m++) {
		sumLLR[m] = 0;
		for (j=0; j < topicCount; j++) {
		    if (j == m) continue;
		    sumLLR[m] += maxLLR2[m][j];
		}
	    }
	    
	    
	    for (j = 0; j < counts[i]; j++) {
		topic = partitions[i][j].label;
		tid = partitions[i][j].tid;
		index = partitions[i][j].index;
		
		doc = &(docCount[tid][index]);
		docsq = &(docSeq[tid][index]);
		//loss is highest incorrect score for this topic - this doc score
		if (lfunc == L1)
		    lw = 1 / (1 + exp(-beta * 
				      ( maxLLRs[tid] - targetLLRs[j])));
		else 				// loss is sum (or average) of highest incorrect scores for this topic (1 from each other topic)
		    lw = 1 / (1 + exp(-beta * 
				      ( sumLLR[tid]/ (topicCount-1) - targetLLRs[j])));
		/*else
		  lw = 1 / (1 + exp(-beta * 
				  ( avgMax[tid] - targetLLR[j])));
		*/
		
		
				
		avgLoss += lw;
		A++;
		
		// FIX THIS
		if (lfunc == L1) {

		  docsq2 = &(docSeq[maxtopics[tid]][maxindexs[tid]]);
		  //doc2 = &(docCount[maxtopics[tid]][maxindexs[tid]]);

		    int idx;
		    token_t linfo;
		    for (idx = 0; idx < docsq->size(); idx++) {
		      linfo = (*docsq)[idx];
		      wid = linfo.wid;
		      if (weights.count(wid)==0)
			continue;
		      wtmp = weights[wid];
		      
		      if (wfunc == EXP) {
			partialSum[wid] +=
			  lw * (1-lw) * linfo.weight *
			  (- (double) linfo.location / (double)docsq->size()) *
			  exp(- wtmp * (double) linfo.location / (double)docsq->size()) *
			  (fcache[i][tid][wid]);
		      }
		      else if (wfunc == NONE) {
			partialSum[wid] +=
			    lw * (1-lw) * linfo.weight *
			    (fcache[i][tid][wid]);
		      }
		      else if (wfunc == GAUSS) {
			double p = (double) linfo.location / (double)docsq->size();
			partialSum[wid] +=
			  lw * (1-lw) * linfo.weight *
			  exp(-wtmp*wtmp * p * p /2.0)*
			  (-wtmp * p * p) * 
				(fcache[i][tid][wid]);
			
			}
		      
		    }
	

		    for (idx = 0; idx < docsq2->size(); idx++) {
		      linfo = (*docsq2)[idx];
		      wid = linfo.wid;
		      if (weights.count(wid)==0)
			continue;
		      wtmp = weights[wid];
		      
		      if (wfunc == EXP) {
			partialSum[wid] +=
			  lw * (1-lw) * linfo.weight *
			  (- (double) linfo.location / (double)docsq2->size()) *
			  exp(- wtmp * (double) linfo.location / (double)docsq2->size()) *
			  (fcache[i][tid][wid]);
		      }
		      else if (wfunc == NONE) {
			partialSum[wid] +=
			    lw * (1-lw) * linfo.weight *
			    (fcache[i][tid][wid]);
		      }
		      else if (wfunc == GAUSS) {
			double p = (double) linfo.location / (double)docsq2->size();
			partialSum[wid] +=
			  lw * (1-lw) * linfo.weight *
			  exp(-wtmp*wtmp * p * p /2.0)*
			  (-wtmp * p * p) * 
			  (fcache[i][tid][wid]);
			
		      }
		      
		    }

		    
		}
		else {
		    // AAH need to do calculus to figure out update	
		    for (m=0; m < topicCount; m++) {
			if (m == tid) continue;
			doc2 = &(docCount[m][maxindex2[m][tid]]);
			for (di = doc2->begin(); di != doc2->end(); di++) {
			    wid = (*di).first;
			    if (doc->count(wid) == 0) continue; 
			    /*						v = 0;
									else
									v = (*doc2)[wid];
			    */							
			    partialSum[wid] +=
				lw * (1-lw) * ((*di).second) *
				fcache[i][tid][wid] / (topicCount-1);
						}
		    }
		    
		    for (di = doc->begin(); di != doc->end(); di++) {
			wid = (*di).first;
			//if (doc->count(wid) > 0) continue;
			
			partialSum[wid] +=
			    lw * (1-lw) * (-(*di).second) *
			    fcache[i][tid][wid];
			
		    }
		    
		    
		}
	    }

	    
	}
	
	/* tmp disable znorm */
	
		for (m=0; m < topicCount; m++) {
			mus[m] /= ND;	
			mu[m] = mus[m];
		}
	       
		for (docx=0; docx < ND; docx++) {		
			for (m=0; m < topicCount; m++) {
			    sigmas[m]+= (mus[m] - trainscores[docx][m])*(mus[m] - trainscores[docx][m]) / (ND-1);
			}
		}
		for (m=0; m < topicCount; m++) {
		sigmas[m] = sqrt(sigmas[m]);
			sigma[m] = sigmas[m];
		}
	
		
	avgLoss /= (double) A;
	
	if (lastLoss >=0 && avgLoss > lastLoss) {
	    backwards++;
	    if (backwards > 2) {
		fprintf(stderr, "Loss increasing\n");
		//exit(1);
	    }
	}
	else {
	    backwards = 0;
	    
	}
	lastLoss = avgLoss;
	
	
	double wtsum = 0.0;
	double nweight;
	int zeros = 0;
	
	
	for (di = weights.begin(); di != weights.end(); di++) {
	    wid = (*di).first;
	    oldweight = (*di).second;
	    
	    // techinically beta is part of partial sum.
	    // but lets save some multiplies
	    (*di).second = oldweight - epsilon * beta * partialSum[wid] / (double)ND;
	    
	    //	    if (debug && N % 10 == 0 ) {
	    //	cerr << wvocab[wid] << " L:" << (*di).second << " DL:" << partialSum[wid] << "\n";
	    //}
	    
	    //if (!strcmp(wvocab[wid].c_str(), "tardes"))
	    //cout << "Partial "<< wid << " " << (partialSum[wid] / (double)ND)<< " " << oldweight << " " << (*di).second << "\n";
	    if (prune > 0.0 && (*di).second < prune)
		(*di).second = 0.0;
	    
	    
	    if ((*di).second <= 0.0) {
		(*di).second = 0.0;
		zeros++;
	    }
					
	    wtsum += (*di).second;
	    
	}
		
	/*	if (normalizeWeights) {
		for (di = weights.begin(); di != weights.end(); di++) {
		wid = (*di).first;
		if ((*di).second == 0)
		continue;
		(*di).second /= wtsum;
		}
		}*/
	
	if (test) {
			
		    fmSum = 0;
		    fpsum = 0;
		    tpsum = 0;
		    fnsum = 0;

			clock_t c1 = clock();
			if  (N % prFreq == 0) {
				error = score_test(N, test, prdata, output, test_results, &testLoss);
			}
			else {
				error  = score_test(N, test, 0,0, test_results, &testLoss);
			}
			clock_t c2 = clock();
		
			scoretime += diffclock(c2, c1);
			
			
			aauc = aeer = 0.0;
			ttestcount = topicCount;
			printf("Iteration %d %0.4f %0.4f %d %0.4f", N, error, avgLoss, zeros, testLoss);	
			for (j=0; j < topicCount; j++) { // foreach $label (sort(keys(%topics))) {
				auc = test_results[2*j];
				eer = test_results[2*j + 1];
				if (auc < 0) {
					ttestcount--;
					continue;
				}
				printf(" %0.4f %0.4f", auc, eer);
				aauc += auc;
				aeer += eer;
			}
			printf(" %0.4f %0.4f", aauc / (double) ttestcount , aeer / (double) ttestcount);
				
			printf( "\n");
			double microF = (2*(double)tpsum) / 
			    (2*(double)tpsum + (double)fpsum + (double)fnsum);
			
			printf ("MicroF %0.4f MacroF %0.4f\n", microF, fmSum / (double) ttestcount);
		}
	
	}

	
	printf("runtime score %0.4f classify %0.4f\n", scoretime / (double)iterations, getllr / ((double) iterations*(double)K));
	
//# model
// write final weights ? and model?
	map<int, double>::iterator si, vv;

	if (weightFile) {
	    write_map(weightFile, weights, wvocab);
	    
	}

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
		word = wvocab[wid];
		fprintf(mfp, "%s %g", word.c_str(), vcount[K][wid]);
	    for (i=0; i < topicCount; i++) 
		fprintf(mfp, " %g", topcount[K][wid][i]);
	    fprintf(mfp, "\n");
	    
	    
	    }
	    fclose(mfp);
	}

	
	if (output && outp)
	    fclose(outp);
	
	
	delete_arrays(ND);
	
	delete [] counts;

	delete [] test_results; 
	
	if (lfunc != MCE) {
	    delete [] maxLLRs; 
	    delete [] maxtopics; 
	    delete [] maxindexs; 
	    
	    delete [] targetLLRs;
	}
	
	
	empty.clear();
	vempty.clear();
	
}
