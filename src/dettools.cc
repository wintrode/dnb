
#include "dettools.h"

#include <stdio.h>

typedef struct _det_t {
	double score;
	double pm;
	double pf;
	double cost;
	double value;
	double cnorm;
} det_t;

typedef struct _trial_t {	
	double score;
	int type;
} trial_t;

class gt2 {
public:
	gt2(int x) {}	
	bool comp(  const  trial_t &j, const trial_t &k ) 
			const { 
				double cmp = (j.score - k.score);
				return (cmp == 0.0) ? (j.type ==1 && k.type ==0) : (cmp > 0.0);
			}
	bool operator()	( const trial_t &j, const trial_t &k ) 
			const { 
				double cmp = (j.score - k.score);
				return (cmp == 0.0) ? (j.type ==1 && k.type ==0) : (cmp > 0.0);
			}
};


int det_eer(vector<double> &tgtscores, vector<double> &ntgtscores, cost_param *cost_params, double *eer) {

// transliteration of NIST DET_TOOLS
/*  my ($ntrue, $true_scores, $nfalse, $false_scores, $cost_params) = @_;
#  $ntrue is the number of trials for true targets (i.e., the target is present).
#  $true_scores is the array of system output scores for true targets.
#  $nfalse is the number of trials for false targets (i.e., the target is absent).
#  $false_scores is the array of system output scores for false targets.
#  $cost_params is an optional hash of parameters used to compute VALUE:
#      $cost_params->{prior} is the prior probability of the target
#      $cost_params->{m0} is the constant cost of a miss, Cm0
#      $cost_params->{m1} is the proportional cost of a miss, Cm1*Pmiss
#      $cost_params->{f0} is the constant cost of a false alarm, Cf0
#      $cost_params->{f1} is the proportional cost of a false alarm, Cf1*Pfa
#
#    The number of true or false system outputs may be less than the number
#    of true or false trials.  The corresponding missing outputs are presumed
#    to be rejections of the target (i.e., misses for true target trials and
#    correct rejections for false target trials).
*/
	
	if (tgtscores.empty()) 
		return NO_TGT;

	if (ntgtscores.empty()) 
		return NO_NONTGT;

	double cost_m0, cost_m1,cost_f0, cost_f1,cnorm,vnorm;
	int i;
	if (cost_params) {
		cost_m0 = cost_params->prior *cost_params->m0; 
		cost_m1 = cost_params->m1 ? cost_params->prior * cost_params->m1 : 0;
		cost_f0 = (1-cost_params->prior) * cost_params->f0;
		cost_f1 = cost_params->f1 ? (1-cost_params->prior)*cost_params->f1 : 0;
	
		cnorm = 1/(cost_m0+cost_m1 < cost_f0+cost_f1 ? cost_m0+cost_m1 : cost_f0+cost_f1);
		vnorm = cost_m0 > 0 ? 1/cost_m0 : -1;
	}
	else {
		cnorm = -1;
		vnorm = -1;
 	}

//#collate true and false trials in decreasing order of score (false before true when scores equal)
  	vector<trial_t> trials;
	trial_t t;
  	for (i=0; i < tgtscores.size(); i++) { 
		t.score = tgtscores[i];
		t.type=1;
		trials.push_back(t);
	}

  	for (i=0; i < ntgtscores.size(); i++) { 
		t.score = ntgtscores[i];
		t.type=0;
		trials.push_back(t);
	}

	sort(trials.begin(), trials.end(), gt2(1));

	  /*@trials = sort {my $cmp;
		  $cmp = $b->{SCORE} <=> $a->{SCORE};
		  return $cmp if $cmp;
		  return $a->{TYPE} cmp $b->{TYPE}} @trials;*/

//#accumulate error statistics for each score, increase false alarm before decreasing miss
  double pm, pf;
	pm = 1.0; pf = 0.0;
  double dpm, dpf;
	dpm = (double)1/(double)tgtscores.size();
	dpf = 1.0/(double)ntgtscores.size();
  
	double cost = (cost_params) ? cost_m0+cost_m1 : -1;

	
	det_t d;
	d.score = trials[0].score;
	d.pm = pm;
	d.pf = pf;
	d.cost = cost;
	d.value = (cost_params) ? (1-vnorm)*cost : -1;
	d.cnorm = (cost_params) ? cnorm*cost : -1;
	
	vector<det_t> det;
  	det.push_back(d);
	
	/*my @det = ({SCORE=>@trials ? $trials[0]{SCORE} : undef,
	      PM=>$pm,
	      PF=>$pf,
	      COST=>$cost, CNORM=>$cnorm ? $cnorm*$cost : undef,
	      VALUE=>$vnorm ? 1-$vnorm*$cost : undef});*/

  	for (i=0; i < trials.size(); i++) {
		if (!trials[i].type)
			pf += dpf;
		else 
			pm -= dpm;

     	if (pf > 1-0.5*dpf) pf = 1;
		if (pm < 0.5*dpm) pm = 0;

		
		if (pf != det[0].pf) {
      		
			cost = (cost_params) ? det[0].pm*(cost_m0+cost_m1*det[0].pm)+pf*(cost_f0+cost_f1*pf) : -1;
		
			d.score = trials[i].score;
			d.pm = det[0].pm;
			d.pf = pf;
			d.cost = cost;
			d.value = (cost_params) ? (1-vnorm)*cost : -1;
			d.cnorm = (cost_params) ? cnorm*cost : -1;

			det.insert(det.begin(),d);
		}
 
		if (pm != det[0].pm) {
    		cost = (cost_params) ? pm*(cost_m0+cost_m1*pm)+pf*(cost_f0+cost_f1*pf) : -1;
 
			d.score = trials[i].score;
			d.pm = pm;
			d.pf = pf;
			d.cost = cost;
			d.value = (cost_params) ? (1-vnorm)*cost : -1;
			d.cnorm = (cost_params) ? cnorm*cost : -1;

			det.insert(det.begin(),d);
		}					
	}

	i = int(det.size()/2);
	int delta;
  	for (delta = int((i+1)/2); 1; delta = int((delta+1)/2)) {
		//printf("I %d %d\n", i, delta);
	
		if ( det[i].pm <= det[i].pf) {
			if (det[i+1].pf <= det[i+1].pm) {
				*eer = det[i+1].pf <= det[i].pm ? det[i].pf : det[i+1].pf;
				return 0;
			}
			
			i += delta;
		} 
		else if (det[i].pf <= det[i].pm) {
			i -= delta;
		}
  	}

	return -1;

}
