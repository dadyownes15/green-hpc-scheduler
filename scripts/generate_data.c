/*############################################################################*/
/*               A Workload Model for Parallel Computer Systems               */
/*               Uri Lublin (uril@cs.huji.ac.il)                              */
/*               Dror Feitelson (feit@cs.huji.ac.il)                          */
/*############################################################################*/
/* This program creates a sample of jobs from the workload model I (Uri)
 * suggested in my master (M.Sc) thesis.
 * The program create a sample of SIZE jobs each one is represented by a line 
 * in the output. Each line contains data about the job's 
 * arrive-time , number-of-nodes , runtime , and type.
 * The load can be changed by changing the values of the model parameters.
 * Remember to set the number-of-nodes' parameters to fit your system.
 * There is an explanation beside each one of the parameters.
 * I distinguish between the two different types (batch or interactive) if the
 * constant INCLUDE_JOBS_TYPE is on (with value 1).
 * The program starts by initializing the model's parameters (according to the
 * following constants definitions -- parameters values). Notice that if 
 * INCLUDE_JOBS_TYPE is on then there are different values to batch-parameters 
 * and interactive-parameters , and if this flag is off then they both get
 * the same value (and the arbitrary type interactive).
 * The program calculates for each job its arrival arrival time (using 2 gamma 
 * distributions , number of nodes (using a two-stage-uniform distribution) 
 * runtime (using the number of nodes and hyper-gamma distribution) and type.
 *
 * Some definitions :
 *    'l'   - the mean load of the system 
 *    'r'   - the mean runtime of a job
 *    'ri'  - the runtime of job 'i'
 *    'n'   - the mean number of nodes of a job
 *    'ni'  - the number of nodes of job 'i'
 *    'a'   - the mean inter-arrival time of a job
 *    'ai'  - the arrival time (from the beginning of the simulation) of job 'i'
 *    'P'   - the number of nodes in the system (the system's size).
 * Given ri,ni,ai,N or r,n,a,N we can calculate the expected load on the system
 *
 *           sum(ri * ni)  
 * load =  ---------------
 *           P * max(ai)
 * 
 * We can calculate an approximation (exact if 'ri' and 'ni' are independent)
 *
 *                     r * n
 * approximate-load = -------
 *                     P * a
 *
 *
 * My model control the r,m,a.
 * One can change them by changing the values of the model's parameters.
 *
 * ----------------------
 * Changing the runtime :
 * ----------------------
 * Let g ~ gamma(alpha,beta) then the expectation and variation of g are :
 * E(g) = alpha*beta  ; Var(g) = alpha*(beta^2)
 * so the coefficient of variation is CV  =  sqrt(Var(g))/E(g)  =  1/sqrt(alpha).
 * one who wishes to enlarge the gamma random value without changing the CV 
 * can set a larger value to the parameter beta . If one wishes that the CV will 
 * change he may set a larger value to alpha parameter.
 * Let hg ~ hyper-gamma(a1,b1,a2,b2,p) then the expectation and variation are:
 * E(hg) = p*a1*b1 + (1-p)*a2*b2
 * Var(hg) = p^2*a1*b1^2 + (1-p)^2*a2*b2^2
 * One who wishes to enlarge the hyper-gamma (or the runtime) random value may 
 * do that using one (or more) of the three following ways:
 * 1. enlarge the first gamma.
 * 2. and/or enlarge the second gamma.
 * 3. and/or set a smaller value to the 'p' parameter.(parameter p of hyper-gamma
 *    is the proportion of the first gamma).
 *    since in my model 'p' is dependent on the number of nodes (p=pa*nodes+pb) 
 *    this is done by diminishing 'pa' and/or diminishing 'pb' such that 'p' 
 *    will be smaller. Note that changing 'pa' affects the correlation 
 *    between the number of nodes a job needs and its runtime.
 * 
 * ------------------------
 * Changing the correlation between the number of nodes a job needs and its 
 * ------------------------
 * runtime:
 * The parameter 'pa' is responsible for that correlation. Since its negative
 * as the number of nodes get larger so is the runtime. 
 * One who wishes no correlation between the nodes-number and the runtime should
 * set 'pa' to be 0 or a small negative number close to 0.
 * One who wishes to have strong such a correlation should set 'pa' to be not so 
 * close to 0 , for example -0.05 or -0.1 . Note that this affect the runtime 
 * which will be larger. One can take care of that by changing the other runtime 
 * parameters (a1,b1,a2,b2,pb). 
 *
 * -----------------------------
 * Changing the number of nodes:
 * -----------------------------
 * Let u ~ uniform(a,b) then the expectation of u is E(u) = (a+b)/2.
 * Let tsu ~ two-stage-uniform(Ulow,Umed,Uhi,Uprob) then the expectation of tsu
 * is E(tsu) = Uprob*(Ulow+Umed)/2 + (1-Uprob)*(Umed+Uhi)/2  =  
 *           = (Uprob*Ulow + Umed + (1-Uprob)*Uhi)/2.
 * 'Ulow' is the log2 of the minimal number of nodes a job may run on.
 * 'Uhi' is the log2 of the system size
 * 'Umed' is the changing point of the cdf function and should be set to
 * 'Umed' = 'Uhi' - 2.5. ('2.5' could be change between 1.5 to 3.5).
 * For example if the system size is 8 (Uhi = log2(8) = 3) its make no sense to
 * set Umed to be 0 or 0.5 . In this case I would set Umed to be 1.5 , and maybe 
 * set Ulow to be 0.8 , so the probability of 2 would not be too small.
 * 'Uprob' is the proportion of the first uniform (Ulow,Umed) and should be set 
 * to a value between 0.7 to 0.95.
 *
 * One who wishes to enlarge the mean number of nodes may do that using one of  
 * the two following ways:
 * 1. set a smaller value to 'prob'
 * 2. and/or enlarge 'med'
 * Remember that changing the mean number of nodes will affect on the runtime 
 * too.
 *
 *
 * --------------------------
 * Changing the arrival time:
 * --------------------------
 * The arrival time is calculated in two stages:
 * First stage : calculate the proportion of number of jobs arrived at every 
 *              time interval (bucket). this is done using the gamma(anum,bnum) 
 *              cdf and the CYCLIC_DAY_START. The weights array holds the points
 *              of all the buckets.
 * Second stage: foreach job calculate its inter-arrival time:
 *              Generate a random value from the gamma(aarr,barr).
 *              The exponential of the random value is the 
 *              points we have. While we have less points than the current 
 *              bucket , "pay" the appropriate number of points and move to the
 *              next bucket (updating the next-arrival-time).
 *
 * The parameters 'aarr' and 'barr' represent the inter-arrival-time in the
 * rush hours. The parameters 'anum' and 'bnum' represent the number of jobs
 * arrived (for every bucket). The proportion of bucket 'i' is: 
 * cdf (i+0.5 , anum , bnum) - cdf(i-0.5 , anum , bnum).
 * We calculate this proportion foreach i from BUCKETS to (24+BUCKETS) , 
 * and the buckets (their indices) that are larger than or equal to 24 are move 
 * cyclically to the right place:
 * (24 --> 0 , 25 --> 1 , ... , (24+BUCKETS) --> BUCKETS  )
 *
 * One who wishes to change the mean inter-arrival time may do that in the 
 * following way:
 * enlarge the rush-hours-inter-arrival time .
 * This is done by enlarging the value of aarr or barr according to the wanted
 * change of the CV. 
 * One who wishes to change the daily cycle may change the values of 'anum' 
 * and/or 'bnum'
 *
 *
 * The functions that randomly choose a value from gamma 
 * distribution were written according to
 *   Raj Jain. 
 *   'THE ART OF COMPUTER SYSTEMS PERFORMANCE ANALYSIS Techniques for
 *    Experimental Design, Measurement, Simulation, and Modeling'.
 *   Jhon Wiley & Sons , Inc.
 *   1991.
 *   Chapter 28 - RANDOM-VARIATE GENERATION (pages 484,485,490,491)
 *
 * The functions that calculate gamma distribution's cumulative distribution 
 * function (cdf) were written according to 
 *   William H. Press , Brian P. Flannery , Saul A. Teukolsky and 
 *   William T. Vetterling.
 *   NUMERICAL RECIPES IN PASCAL The Art of Scientific Computing.
 *   Cambridge University Press
 *   1989
 *   Chapter 6 - Special Functions (pages 180-183).
 *
 */

/*############################################################################*/
/*                    BEGINNING OF THE PROGRAM                                */
/*############################################################################*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <limits.h>

/*#############################################################################*/
/*                          CONSTANTS                                          */
/*#############################################################################*/

#define MAX_END_TIME 3*365*24*60*60 
#define TOO_MUCH_TIME 12        /* no more than two days =exp(12) for runtime  */
#define TOO_MUCH_ARRIVE_TIME 13 /* no more than 5 days =exp(13) for arrivetime */

#define BUCKETS 48            /* number of buckets in one day -- 48 half hours */
/* we now define (calculate) how many seconds are in one hour,day,bucket       */
#define SECONDS_IN_HOUR 3600                       /*    seconds in one hour   */
#define SECONDS_IN_DAY (24*SECONDS_IN_HOUR)        /*     seconds in one day   */
#define SECONDS_IN_BUCKET (SECONDS_IN_DAY/BUCKETS) /*  seconds in one bucket   */
#define HOURS_IN_DAY    24                       /* number of hours in one day */

/* The hours of the day are being moved cyclically.  
 * instead of [0,23] make the day [CYCLIC_DAY_START,(24+CYCLIC_DAY_START-1)]
 */
#define CYCLIC_DAY_START 11

#define ITER_MAX 1000 /* max iterations number for the iterative calculations */
#define EPS 1E-10     /* small epsilon for the iterative algorithm's accuracy */

/* INCLUDE_JOBS_TYPE tells us if we should differ between batch and interactive 
 * jobs or not
 * If the value is 1 , then we use both batch and interactive values for
 * the parameters and the output sample includes both interactive and batch jobs.
 * We choose the type of the job to be the type whose next job arrives to the
 * system first (smaller next arrival time).
 * If the value is 0 , then we use the "whole sample" parameters. The output 
 * sample includes jobs from the same type ( arbitrarily we choose interactive).
 * We force the type to be interactive by setting the next arrival time of batch
 * jobs to be ULONG_MAX -- always larger than the interactive's next arrival.
 */
#define INCLUDE_JOBS_TYPE 1 /* (1 for yes , 0 for no ) */

/* represent BATCH or INTERACTIVE parameters in appropriate array */
#define ACTIVE 0
#define BATCH  1

/* there are two optional output formats: Standard Workload format, or short format.
 * to get the standard format, define the constant SWF to be 1.
 * the short format includes only four fields: arrival time, nodes, runtime, and type.
 */
#define SWF 1


/*############################################################################*/
/*                      MODEL PARAMETERS                                      */
/*############################################################################*/

/* The parameters for number of nodes:
 * serial_prob is the proportion of the serial jobs
 * pow2_prob   is the proportion of the jobs with power 2 number of nodes
 * ULow , UMed , UHi and Uprob are the parameters for the two-stage-uniform 
 * which is used to calculate the number of nodes for parallel jobs.
 * ULow is the log2 of the minimal size of job in the system (you can add or 
 * subtract 0.2 to give less/more probability to the minimal size) .
 * UHi  is the log2 of the maximal size of a job in the system (system's size)
 * UMed should be in [UHi-1.5 , UHi-3.5]
 * Uprob should be in [0.7 - 0.95]
 */
#define SERIAL_PROB_BATCH 0.2927
#define POW2_PROB_BATCH   0.6686
#define ULOW_BATCH        1.2 /* smallest parallel batch job has 2 nodes */
#define UMED_BATCH        5
#define UHI_BATCH         7  /* biggest batch job has 128 nodes */
#define UPROB_BATCH       0.875

#define SERIAL_PROB_ACTIVE 0.1541
#define POW2_PROB_ACTIVE  0.625
#define ULOW_ACTIVE       1  /* smallest interactive parallel job has 2 nodes */
#define UMED_ACTIVE       3
#define UHI_ACTIVE        5.5  /* biggest interactive job has 45 nodes */
#define UPROB_ACTIVE      0.705

/* The parameters for the running time
 * The running time is computed using hyper-gamma distribution.
 * The parameters a1,b1,a2,b2 are the parameters of the two gamma distributions
 * The p parameter of the hyper-gamma distribution is calculated as a straight 
 * (linear) line p = pa*nodes + pb.
 * 'nodes' will be calculated in the program, here we defined the 'pa','pb' 
 * parameters.
 */
#define A1_BATCH          6.57
#define B1_BATCH          3
#define A2_BATCH          639.1
#define B2_BATCH          0.0156
#define PA_BATCH          -0.003
#define PB_BATCH          0.6986

#define A1_ACTIVE         3.8351 
#define B1_ACTIVE         3
#define A2_ACTIVE         7.073 
#define B2_ACTIVE         1.5
#define PA_ACTIVE         -0.0118 
#define PB_ACTIVE         0.9156 


/* The parameters for the inter-arrival time
 * The inter-arriving time is calculated using two gamma distributions.
 * gamma(aarr,barr) represents the inter_arrival time for the rush hours. It is 
 * independent on the hour the job arrived at.
 * The cdf of gamma(bnum,anum) represents the proportion of number of jobs which 
 * arrived at each time interval (bucket).
 * The inter-arrival time is calculated using both gammas
 * Since gamma(aarr,barr) represents the arrive-time at the rush time , we use
 * a constant ,ARAR (Arrive-Rush-All-Ratio), to set the alpha parameter (the new
 * aarr) so it will represent the arrive-time at all hours of the day.
 */

#define AARR_BATCH        6.0415
#define BARR_BATCH        0.8531
#define ANUM_BATCH        6.1271
#define BNUM_BATCH        5.2740
#define ARAR_BATCH        1.0519

#define AARR_ACTIVE       6.5510
#define BARR_ACTIVE       0.6621
#define ANUM_ACTIVE       8.9186
#define BNUM_ACTIVE       3.6680
#define ARAR_ACTIVE       0.9797



/* 
 * Here are the model's parameters for typeless data (no batch nor interactive)
 * We use those parameters when INCLUDE_JOBS_TYPE is off (0)
 */

#define SERIAL_PROB       0.244
#define POW2_PROB         0.576
#define ULOW              0.8       /* The smallest parallel job is 2 nodes */
#define UMED              4.5
#define UHI               8         /* SYSTEM SIZE is 2^UHI == 128          */
#define UPROB             0.86

#define A1                4.2
#define B1                0.94
#define A2                312
#define B2                0.03
#define PA                -0.0054
#define PB                0.78

#define AARR              10.2303
#define BARR              0.4871
#define ANUM              8.1737
#define BNUM              3.9631
#define ARAR              1.0225


/* start the simulation at midnight (hour 0) */
#define START             0



/*############################################################################*/
/*                      PROTOTYPES                                            */
/*############################################################################*/

void init(double* a1,double* b1,double* a2,double* b2,double* pa,double* pb,
	  double* aarr,double* barr,double* anum,double* bnum , 
	  double* SerialProb, double* Pow2Prob , 
	  double* ULow , double* UMed , double* UHi , double* Uprob , 
	  double weights[2][BUCKETS]);
unsigned int calc_number_of_nodes(double SerialProb,double Pow2Prob,double ULow,
				  double UMed,double Uhi,double Uprob);
unsigned long time_from_nodes(double alpha1, double beta1, 
			      double alpha2, double beta2, 
			      double pa    , double pb   , unsigned int nodes);
unsigned long arrive(int *type , double weights[2][BUCKETS] , 
		     double aarr[2] , double barr[2]);
void arrive_init(double *aarr , double*barr , double *anum, double *bnum,
			int start_hour , double weights[2][BUCKETS]);
void calc_next_arrive(int type ,double weights[2][BUCKETS] ,double aarr[2],
		      double barr[2]);
double hyper_gamma(double a1, double b1, double a2, double b2, double p);
double gamrnd(double alpha , double beta);
double gamrnd_int_alpha(unsigned n , double beta);
double gamrnd_alpha_smaller_1(double alpha, double beta);
double betarnd_less_1(double alpha , double beta);
double gamcdf(double x , double alpha , double beta);
double gser(double x , double a);
double gcf(double x , double a);
double two_stage_uniform(double low, double med, double hi, double prob);


/*###########################################################################*/
/*                      GLOBAL VARIABLES                                     */
/*###########################################################################*/

/* current time interval (the bucket's number) */
int current[2];

/* the number of seconds from the beginning of the simulation*/
unsigned long time_from_begin[2]; 


/*----------------------------------------------------------------------------*/
/*                      THE MAIN FUNCTION                                     */
/*----------------------------------------------------------------------------*/
int main()
{
  int i;
  double a1[2],b1[2],a2[2],b2[2],pa[2],pb[2];
  double aarr[2],barr[2],anum[2],bnum[2];
  double SerialProb[2] , Pow2Prob[2] , ULow[2] , UMed[2] , UHi[2] , Uprob[2];
  unsigned long arr_time , run_time;
  unsigned int nodes;
  int type;
  double weights[2][BUCKETS] ; /* the appropriate weight (points) for each    */
			       /* time-interval used in the arrive function   */
  long seed;
  FILE *outfile;
  char *output_filename = "workload.swf";

  seed = (long)time(NULL);
  srand48(seed);

  outfile = fopen(output_filename, "w");
  if (outfile == NULL) {
      fprintf(stderr, "Couldn't open file %s for writing.\n", output_filename);
      return 1;
  }

  /* set all the parameters' values */
  init(a1,b1,a2,b2,pa,pb,aarr,barr,anum,bnum , 
       SerialProb, Pow2Prob , ULow , UMed , UHi , Uprob , weights);

  /* for each job, calculate its type (if needed) , arrival time
   * number of nodes and run time, and print them.
   */
#if SWF
  /*
   * print SWF header comments
   */
  fprintf(outfile, "; Version: 2\n");
  fprintf(outfile, "; Acknowledge: Uri Lublin, Hebrew University\n");
  fprintf(outfile, "; Information: http://www.cs.huji.ac.il/labs/parallel/workload\n");
  fprintf(outfile, "; MaxNodes: %d\n", 1 << (int)UHI);
  fprintf(outfile, "; MaxRuntime: %d\n", (int)exp((double)TOO_MUCH_TIME));
#endif

  for (i=0; ; i++) {
    arr_time = arrive(&type,weights,aarr,barr);
    if (arr_time > MAX_END_TIME) {
        break;
    }
    nodes = calc_number_of_nodes(SerialProb[type] , Pow2Prob[type],
				 ULow[type], UMed[type], UHi[type], Uprob[type]);
    run_time = time_from_nodes(a1[type] , b1[type] , a2[type] , b2[type],
			       pa[type] , pb[type] , nodes);
#if SWF
    fprintf(outfile, "%5d %7lu -1 %7lu %3d -1 -1 -1 -1 -1 -1 -1 1 -1 -1 -1 %d -1 -1 -1\n",
	   i+1, arr_time, run_time, nodes, type);
#else
    fprintf(outfile, "%lu\t%u\t%lu\t%d\n", arr_time , nodes , run_time , type);
#endif
  }

  fclose(outfile);
  printf("Workload data saved to %s\n", output_filename);

  return 0;
}

/*----------------------------------------------------------------------------*/
/*   INIT                                                                     */
/*----------------------------------------------------------------------------*/

/* 
 * This function initializes all the parameters of the workload model, and 
 * the arrival process.
 */
void init(double* a1,double* b1,double* a2,double* b2,double* pa,double* pb,
	  double* aarr,double* barr,double* anum,double* bnum , 
	  double* SerialProb, double* Pow2Prob , 
	  double* ULow , double* UMed , double* UHi , double* Uprob , 
	  double weights[2][BUCKETS] )
{
  if (INCLUDE_JOBS_TYPE) { /* seperate batch from interactive */
    SerialProb[BATCH] = SERIAL_PROB_BATCH; 
    Pow2Prob[BATCH] = POW2_PROB_BATCH;
    ULow[BATCH] =  ULOW_BATCH;
    UMed[BATCH] =  UMED_BATCH;
    UHi[BATCH]  =  UHI_BATCH;
    Uprob[BATCH] = UPROB_BATCH;
    
    SerialProb[ACTIVE] = SERIAL_PROB_ACTIVE;
    Pow2Prob[ACTIVE] = POW2_PROB_ACTIVE;
    ULow[ACTIVE] = ULOW_ACTIVE;
    UMed[ACTIVE] = UMED_ACTIVE;
    UHi[ACTIVE]   = UHI_ACTIVE;
    Uprob[ACTIVE] = UPROB_ACTIVE;

    a1[BATCH] =  A1_BATCH;    b1[BATCH] =  B1_BATCH;
    a2[BATCH] =  A2_BATCH;    b2[BATCH] =  B2_BATCH;
    pa[BATCH] =  PA_BATCH;    pb[BATCH] = PB_BATCH; 

    a1[ACTIVE] = A1_ACTIVE;   b1[ACTIVE] = B1_ACTIVE; 
    a2[ACTIVE] = A2_ACTIVE;   b2[ACTIVE] = B2_ACTIVE; 
    pa[ACTIVE] = PA_ACTIVE;   pb[ACTIVE] = PB_ACTIVE; 

    aarr[BATCH] = AARR_BATCH*ARAR_BATCH;     barr[BATCH] = BARR_BATCH; 
    anum[BATCH] = ANUM_BATCH;                bnum[BATCH] = BNUM_BATCH;

    aarr[ACTIVE] = AARR_ACTIVE*ARAR_ACTIVE;  barr[ACTIVE] = BARR_ACTIVE; 
    anum[ACTIVE] = ANUM_ACTIVE;              bnum[ACTIVE] = BNUM_ACTIVE;
  }
  else { /* whole sample -- make all interactive */
  SerialProb[BATCH] = SerialProb[ACTIVE] = SERIAL_PROB;
  Pow2Prob[BATCH] = Pow2Prob[ACTIVE]     = POW2_PROB;
  ULow[BATCH]    = ULow[ACTIVE]          = ULOW ; 
  UMed[BATCH]     = UMed[ACTIVE]         = UMED ;
  UHi[BATCH]      = UHi[ACTIVE]          = UHI ;
  Uprob[BATCH]    = Uprob[ACTIVE]        = UPROB ;

  a1[BATCH]   = a1[ACTIVE]   = A1; 
  b1[BATCH]   = b1[ACTIVE]   = B1; 
  a2[BATCH]   = a2[ACTIVE]   = A2; 
  b2[BATCH]   = b2[ACTIVE]   = B2;  
  pa[BATCH]   = pa[ACTIVE]   = PA; 
  pb[BATCH]   = pb[ACTIVE]   = PB; 

  aarr[BATCH] = aarr[ACTIVE] = AARR * ARAR; 
  barr[BATCH] = barr[ACTIVE] = BARR; 
  anum[BATCH] = anum[ACTIVE] = ANUM; 
  bnum[BATCH] = bnum[ACTIVE] = BNUM;
  }

  arrive_init(aarr,barr,anum,bnum,START,weights);
  if ( ! INCLUDE_JOBS_TYPE )     /* make all jobs interactive */
    time_from_begin[BATCH] = ULONG_MAX;
  
}

/*----------------------------------------------------------------------------*/
/*    CALC_NUMBER_OF_NODES  (NUMBER OF NODES)                                 */
/* -------------------------------------------------------------------------- */
/*
 * we distinguish between serial jobs , power2 jobs and other.
 * for serial job (with probability SerialProb) the number of nodes is 1
 * for all parallel jobs (both power2 and other jobs) we randomly choose a 
 * number (called 'par') from two-stage-uniform distribution.
 * if the job is a power2 job then we make it an integer (par = round(par)), 
 * the number of nodes will be 2^par but since it must be an integer we return
 * round(pow(2,par)).
 * if we made par an integer then 2^par is ,obviously, a power of 2.
 */
unsigned int calc_number_of_nodes(double SerialProb,double Pow2Prob,double ULow,
				  double UMed,double Uhi,double Uprob)
{
  double u,par;
  
  u = drand48();
  if (u <= SerialProb) /* serial job */
    return 1;
  par = two_stage_uniform(ULow,UMed,Uhi,Uprob);
  if (u <= (SerialProb + Pow2Prob))        /* power of 2 nodes parallel job */
    par = (unsigned int)(par + 0.5);                /* par = round(par)     */
  return ((unsigned int)(pow(2,par) + 0.5));        /* return round(2^par)  */
}

/*----------------------------------------------------------------------------*/
/* TIMES_FROM_NODES  (RUNTIME)                                                */
/* -------------------------------------------------------------------------- */
/* 
 * time_from_nodes returns a value of a random number from hyper_gamma 
 *    distribution.
 * The a1,b1,a2,b2 are the parameters of both gammas.
 * The 'p' parameter is calculated from the 'nodes' 'pa' and 'pb' arguments 
 * using the formula:   p = pa * nodes + pb.
 * we keep 'p' a probability by forcing its value to be in the interval [0,1].
 * if the value that was randomly chosen is too big (larger than 
 * TOO_MUCH_TIME) then we choose another random value.
 */
unsigned long time_from_nodes(double alpha1, double beta1, 
			      double alpha2, double beta2,
			      double pa , double pb , unsigned int nodes)
{
  double hg;
  double p = pa*nodes + pb;
  if (p>1)
    p=1;
  else if (p<0)
    p=0;
  do {
    hg = hyper_gamma(alpha1 , beta1 , alpha2 , beta2 , p); 
  } while (hg > TOO_MUCH_TIME);
  return exp(hg);
}

/*############################################################################*/
/*                          ARRIVE PROCESS                                    */
/*############################################################################*/

/* The arrive process.
 * 'arrive' returns arrival time (from the beginning of the simulation) of 
 * the current job.
 * The (gamma distribution) parameters 'aarr' and 'barr' represent the 
 * inter-arrival time at rush hours.
 * The (gamma distribution) parameters 'anum' and 'bnum' represent the number
 * of jobs arriving at different times of the day. Those parameters fit a day 
 * that contains the hours [CYCLIC_DAY_START..24+CYCLIC_DAY_START] and are 
 * cyclically moved back to 0..24
 * 'start' is the starting time (in hours 0-23) of the simulation.
 * If the inter-arrival time randomly chosen is too big (larger than
 * TOO_MUCH_ARRIVE_TIME) then another value is chosen.
 *
 * The algorithm (briefly):
 * A. Preparations (calculated only once in 'arrive_init()':
 * 1. foreach time interval (bucket) calculate its proportion of the number 
 *    of arriving jobs (using 'anum' and 'bnum'). This value will be the 
 *    bucket's points.
 * 2. calculate the mean number of points in a bucket ()
 * 3. divide the points in each bucket by the points mean ("normalize" the 
 *    points in all buckets)
 * B. randomly choosing a new arrival time for a job:
 * 1. get a random value from distribution gamma(aarr , barr).
 * 2. calculate the points we have.
 * 3. accumulate inter-arrival time by passing buckets (while paying them 
 *    points for that) until we do not have enough points.
 * 4. handle reminders - add the new reminder and subtract the old reminder.
 * 5. update the time variables ('current' and 'time_from_begin')
 */

/*----------------------------------------------------------------------------*/
/*     ARRIVE_INIT                                                            */
/*----------------------------------------------------------------------------*/
void arrive_init(double *aarr, double *barr, double *anum, double *bnum ,
			int start_hour , double weights[2][BUCKETS])
{
  int i,j,idx,moveto = CYCLIC_DAY_START;
  double mean[2] = {0,0};

  bzero(weights , sizeof(weights));
  current[BATCH] = current[ACTIVE] = start_hour * BUCKETS / HOURS_IN_DAY; 

  /* 
   * for both batch and interactive calculate the propotion of each bucket ,
   * and their mean */
  for (j=0 ; j<=1 ; j++) {
    for (i=moveto ; i<BUCKETS+moveto ; i++) {
      idx = (i-1)%BUCKETS; /* i-1 since array indices are 0..47 and not 1..48 */
      weights[j][idx] =  
	gamcdf(i+0.5, anum[j], bnum[j]) - gamcdf(i-0.5, anum[j],bnum[j]);
      mean[j] += weights[j][idx];
    }
    mean[j] /= BUCKETS;
  }

  /* normalize it so we associates between seconds and points correctly */
  for (j=0 ; j<=1 ; j++)
    for (i=0 ; i<BUCKETS ; i++)
      weights[j][i] /= mean[j];


  calc_next_arrive(BATCH ,weights,aarr,barr);
  calc_next_arrive(ACTIVE,weights,aarr,barr);
}

/*----------------------------------------------------------------------------*/
/*    CALC_NEXT_ARRIVE                                                        */
/*----------------------------------------------------------------------------*/
/* 
 * 'calc_next_arrive' calculates the next inter-arrival time according 
 * to the current time of the day.  
 * 'type' is the type of the next job -- interactive or batch
 * alpha and barr are the parameters of the gamma distribution of the 
 * inter-arrival time.
 * NOTE: this function changes the global variables concerning the arrival time.
 */
void calc_next_arrive(int type,double weights[2][BUCKETS] ,double aarr[2],
			     double barr[2])
{
  static double points[2] = {0,0} , reminder[2] = {0,0};
  int bucket;
  double gam , next_arrive  , new_reminder , more_time ;
  
  
  bucket = current[type]; /* the bucket of the current time*/
  do {     /* randomly choose a (not too big) number from gamma distribution */
    gam = gamrnd(aarr[type],barr[type]);
  } while (gam > TOO_MUCH_ARRIVE_TIME);
  
  points[type] += (exp(gam) / SECONDS_IN_BUCKET); /* number of points         */
  next_arrive = 0;
  while (points[type] > weights[type][bucket]) { /* while have more points    */
    points[type] -= weights[type][bucket];       /* pay points to this bucket */
    bucket = (bucket+1)  % 48;             /*   ... and goto the next bucket  */
    next_arrive += SECONDS_IN_BUCKET;      /* accumulate time in next_arrive  */
  }
  new_reminder = points[type]/weights[type][bucket];
  more_time = SECONDS_IN_BUCKET * ( new_reminder - reminder[type]);

  next_arrive += more_time;             /* add reminders         */

  reminder[type] = new_reminder;        /* save it for next call */

  /* update the global variables */
  time_from_begin[type] += next_arrive;
  current[type] = bucket;
}


/*----------------------------------------------------------------------------*/
/*     ARRIVE         (ARRIVAL TIME AND TYPE)                                 */
/*----------------------------------------------------------------------------*/
/* 
 * return the time for next job to arrive the system.
 * returns also the type of the next job , which is the type that its 
 * next arrive time (time_from_begin) is closer to the start (smaller).
 * notice that since calc_next_arrive changes time_from_begin[] we must save
 * the time_from_begin in 'res' so we would be able to return it.
 */
unsigned long arrive(int *type , double weights[2][BUCKETS] , double aarr[2] , 
		     double barr[2])
{
  unsigned long res;

  *type = (time_from_begin[BATCH] < time_from_begin[ACTIVE]) ? BATCH : ACTIVE;
  res = time_from_begin[*type];       /* save the job's arrival time     */

  /* randomly choose the next job's (of the same type) arrival time      */
  calc_next_arrive(*type,weights,aarr,barr); 

  return res;
}

/*############################################################################*/
/*                   STATISTICAL DISTRIBUTIONS                                */
/*############################################################################*/

/*----------------------------------------------------------------------------*/
/*      HYPER_GAMMA                                                           */
/*----------------------------------------------------------------------------*/
/* hyper_gamma returns a value of a random variable of mixture of 
 * two gammas. its parameters are those of the two gammas: a1,b1,a2,b2
 * and the relation between the gammas (p = the probability of the first gamma).
 * we first randomly decide which gamma will be active ((a1,b1) or (a2,b2)). 
 * then we randomly choose a number from the chosen gamma distribution.
 */
double hyper_gamma(double a1, double b1, double a2, double b2, double p)
{
  double a,b,hg, u = drand48();

  if (u <= p) { /* gamma(a1,b1) */
    a = a1;
    b = b1;
  }
  else  {          /* gamma(a2,b2) */
    a = a2;
    b = b2;
  }
  
  /* generate a value of a random variable from distribution gamma(a,b) */
  hg = gamrnd(a,b);
  return hg;
}

/*----------------------------------------------------------------------------*/
/*      GAMRND                                                                */
/*----------------------------------------------------------------------------*/
/* gamrnd returns a value of a random variable of gamma(alpha,beta).
 * gamma(alpha,beta) = gamma(int(alpha),beta) + gamma(alpha-int(alpha),beta).
 * This function and the following 3 functions were written according to 
 * Jain Raj,  'THE ART OF COMPUTER SYSTEMS PERFORMANCE ANALYSIS Techniques for 
 *    Experimental Design, Measurement, Simulation, and Modeling'.
 *   Jhon Wiley & Sons , Inc.
 *   1991.
 *   Chapter 28 - RANDOM-VARIATE GENERATION (pages 484,485,490,491)
 *
 * can be improved by getting 'diff' and 'intalpha' as function's parameters
 */

double gamrnd(double alpha , double beta)
{
  double diff,gam = 0;
  unsigned long intalpha = (unsigned long)alpha;
  if (alpha >= 1)
    gam += gamrnd_int_alpha(intalpha , beta);
  if ((diff = alpha - intalpha) > 0) 
    gam += gamrnd_alpha_smaller_1(diff,beta);
  return gam;
}

/*----------------------------------------------------------------------------*/
/*      GAMRND_INT_ALPHA                                                      */
/*----------------------------------------------------------------------------*/
/* 
 * gamrnd_int_alpha returns a value of a random variable of gamma(n,beta) 
 * distribution where n is integer(unsigned long actually).
 * gamma(n,beta) == beta*gamma(n,1) == beta* sum(1..n){gamma(1,1)} 
 * ==  beta* sum(1..n){exp(1)} ==  beta* sum(1..n){-ln(uniform(0,1))}
 */
inline double gamrnd_int_alpha(unsigned n , double beta)
{
  double acc = 0;
  unsigned i;
  for (i =0 ; i<n ; i++)
    acc += log(drand48()); /* sum the exponential random varibales */
  return (-acc * beta);
}

/*----------------------------------------------------------------------------*/
/*      GAMRND_ALPHA_SMALLER_1                                                */
/*----------------------------------------------------------------------------*/
/* 
 * gamrnd_alpha_smaller_1 returns a value of a random variable of 
 * gamma(alpha,beta) where alpha is smaller than 1.
 * This is done using the Beta distribution. 
 * (alpha<1) ==>  (1-alpha<1)  ==> we can use beta_less_1(alpha,1-alpha)
 * gamma(alpha,beta) = exponential(beta) * Beta(alpha,1-alpha)
 */
inline double gamrnd_alpha_smaller_1(double alpha, double beta)
{
  double x = betarnd_less_1(alpha , 1-alpha); /* beta random variable */
  double y = -log(drand48()); /* exponential random variable */
  return (beta * x * y);
}

/*----------------------------------------------------------------------------*/
/*      BETARND_LESS_1                                                        */
/*----------------------------------------------------------------------------*/
/* 
 * betarnd_less_1 returns a value of a random variable of beta(alpha,beta) 
 * distribution where both alpha and beta are smaller than 1 (and larger than 0)
 */
inline double betarnd_less_1(double alpha , double beta)
{
  double x,y, u1,u2;
  do {
    u1 = drand48();
    u2 = drand48();
    x = pow(u1,1/alpha);
    y = pow(u2,1/beta);
  }  while (x+y > 1);
  return (x/(x+y));
}

/*----------------------------------------------------------------------------*/
/*      GAMCDF                                                                */
/*----------------------------------------------------------------------------*/
/* 
 * return the cumulative distribution function of gamma(alpha,beta) 
 * distribution at the point 'x';
 * return -1 if an error (non-convergence) occur.
 * This function and the following two functions were written according to
 *   William H. Press , Brian P. Flannery , Saul A. Teukolsky and 
 *   William T. Vetterling.
 *   NUMERICAL RECIPES IN PASCAL The Art of Scientific Computing.
 *   Cambridge University Press
 *   1989
 *   Chapter 6 - Special Functions (pages 180-183).
 */
inline double gamcdf(double x , double alpha , double beta)
{
  x /= beta;
  if (x < (alpha + 1)) {
    return gser(x,alpha);
  }

  /* x >= a+1 */
  return 1 - gcf(x,alpha);
}

/*----------------------------------------------------------------------------*/
/*      GSER                                                                  */
/*----------------------------------------------------------------------------*/
/*
 *
 */
inline double gser(double x , double a)
{
  int i;
  double sum,monom,aa=a;
  
  sum = monom = 1/a;

  for (i=0 ; i<ITER_MAX ; i++) {
    ++aa;
    monom *= (x/aa);
    sum += monom;
    if (monom < sum * EPS)
      return (sum * exp(-x+a*log(x)-lgamma(a)));
  }
  return -1; /* error did not converged */
}


/*----------------------------------------------------------------------------*/
/*      GCF                                                                   */
/*----------------------------------------------------------------------------*/
/*
 *
 */
inline double gcf(double x , double a)
{
  int i;
  double gold=0 , g , a0=1 ,  a1=x , b0=0 , b1=1 , fac=1 , anf , ana;
  
  for (i=1 ; i<=ITER_MAX ; i++) {
    ana = i - a;
    a0 = (a1 + a0*ana) * fac;
    b0 = (b1 + b0*ana) * fac;
    anf = i * fac;
    a1 = x*a0 + anf*a1;
    b1 = x*b0 + anf*b1;
    if (a1 != 0.0) {
      fac = 1/a1;
      g = b1 * fac;
      if (fabs((g-gold)/g) < EPS)
	return ( g * exp(-x + a*log(x) - lgamma(a)));
      gold = g;
    }
  }
  return 2;  /* gamcdf will return -1 */
}


/*----------------------------------------------------------------------------*/
/*      TWO_STAGE_UNIFORM                                                     */
/*----------------------------------------------------------------------------*/
/* 
 * two_stage_uniform returns a random variable from a mixture of two uniform 
 * distributions : [low,med] and [med,hi]. 
 * 'prob' is the proportion of the first uniform.
 * first we randomly choose the active uniform according to prob.
 * then we randomly choose a value from the chosen uniform distribution.
 */
double two_stage_uniform(double low, double med, double hi, double prob)
{
  double a,b,tsu, u = drand48();

  if (u <= prob) { /* uniform(low , med) */
    a = low;
    b = med;
  }
  else  {          /* uniform(med , hi) */
    a = med;
    b = hi;
  }
  
  /* generate a value of a random variable from distribution uniform(a,b) */
  tsu = (drand48() * (b-a)) + a;
  return tsu;
}