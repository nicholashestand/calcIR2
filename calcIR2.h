// create a new subclase that inherits the gmx reader class
#include <string.h>
#include <complex.h>
#include <gmx_reader.h>

#define PSTR        "||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PWID        50


#ifndef HBondDistribution_H
#define HBondDistribution_H
class model: public gmx_reader
{
    public:
        // class variables
        string  outf="spec";                    // name for output files
        string  waterModel="e3b3";              // water model tip4p, tip4p2005, e3b2, e3b3
        string  species="HOD/H2O";              // which species to calculate
        string  HODtype="implicit";             // are HOD simulated (explicit) or H2O (implicit)
        int     nsamples=1;                     // number of samples to take
        int     ntcfpoints=200;                 // number of tcf points
        float   sampleEvery=0.;                 // number of ps to start every sample
        float   beginTime=0.;                   // time to begin for possible equilibration
        float   tcfdt=0.01;                     // time between tcf points
        float   t1=0.260;                       // T1 relaxation time
        float   avef=3400.;                     // appx average frequency to get rid 
                                                // of high frequency fluctuations;

        const int OW=0, HW1=1, HW2=2, MW=3;     // integers to access water atoms
        float charge[4];                        // charge on water atoms
        int   nchrom;                           // number of chromophores
        float map_w[3];                         // electric field map coefficients
        float map_x[2]; 
        float map_p[2];
        float map_muprime[3];
        int   *nlist;                           // neighbor list
        int   nlistmax=2000;                    // max number of neighbors in neighbor list
        float nlistcut=2.;                      // neighbor list cutoff in nm
        rvec  *lastox;                          // last o position for neighborlist update

        float *eproj;                           // electric field
        rvec  *dipole_t0;                       // transition dipole moment of each chrom at t0
        rvec  *dipole;                          // transition dipole moment of each chrom
        matrix *alpha_t0;                       // polarizabilities of each chrom at t0
        matrix *alpha;                          // polarizabiliites of each chrom

        complex<double>         *irtcf; 
        complex<double>         *vvtcf; 
        complex<double>         *vhtcf;
        complex<double>         *propigator;

        const complex<double>   img          = {0.,1.};
        const complex<double>   complex_zero = {0.,0.};
        const complex<double>   complex_one  = {1.,0.};
        const float             hbar         = 5.308837367; // hbar in cm-1 * ps
        const int               nchrom_mol   = 2;           // number of chromophores
        const int               nzeros       = 25600;

        double                  *Firtcf;
        double                  *Fvvtcf;
        double                  *Fvhtcf;
        double                  *Fomega;

        const float             sd_min = 1000., sd_max = 5000.;
        const int               sd_step=1, nsd=4001;
        float   *sd;
        float   *frequency;

    
        // Default constructor and destructor
        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        model( string _inpf_ );
        ~model();

        // functions
        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        void  get_alpha_mu();
        void  set_alpha_mu_t0();
        void  reset_propigator();
        void  get_efield();
        void  adjust_Msite();
        int   get_chrom_nx( int mol, int h );
        void  get_tcf_dilute( int tcfpoint );
        void  get_specdens();
        float get_muprime( float efield );
        float get_omega10( float efield );
        float get_x10( float omega10 );
        float get_p10( float omega10 );
        void  do_ffts();
        void  write_tcf();
        void  write_spec();
        void  update_nlist();
        void  printProgress( int currentStep, int totalSteps );
};
#endif
