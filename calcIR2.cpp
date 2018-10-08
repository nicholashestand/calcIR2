#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <xdrfile.h>
#include <gmx_reader.h>
#include <fftw3.h>
#include <omp.h>
#include <pthread.h>
#include "calcIR2.h"

#define PI 3.14159265359 

using namespace std;

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
model::model( string _inpf_ ) : gmx_reader::gmx_reader( _inpf_ )
// Default constructor
{

    // set userparams from input file
    for ( int i = 0; i < nuParams; i ++ )
    {
        if ( uParams[i] == "outf" )             outf            = uValues[i];
        if ( uParams[i] == "waterModel" )       waterModel      = uValues[i];
        if ( uParams[i] == "species" )          species         = uValues[i];
        if ( uParams[i] == "nsamples" )         nsamples        = stoi(uValues[i]);
        if ( uParams[i] == "ntcfpoints" )       ntcfpoints      = stoi(uValues[i]);
        if ( uParams[i] == "sampleEvery" )      sampleEvery     = stof(uValues[i]);
        if ( uParams[i] == "beginTime" )        beginTime       = stof(uValues[i]);
        if ( uParams[i] == "tcfdt" )            tcfdt           = stof(uValues[i]);
        if ( uParams[i] == "t1" )               t1              = stof(uValues[i]);
        if ( uParams[i] == "HODtype" )          HODtype         = uValues[i];
        if ( uParams[i] == "nlistcut" )         nlistcut        = stof(uValues[i]);
        if ( uParams[i] == "nlistmax" )         nlistmax        = stoi(uValues[i]);
    }

    cout << "Set outf to: "         << outf         << endl;
    cout << "Set waterModel to: "   << waterModel   << endl;
    cout << "Set species to: "      << species      << endl;
    cout << "Set nsamples to: "     << nsamples     << endl;
    cout << "Set ntcfpoints to: "   << ntcfpoints   << endl;
    cout << "Set sampleEvery to: "  << sampleEvery  << endl;
    cout << "Set beginTime to: "    << beginTime    << endl;
    cout << "Set T1 to: "           << t1           << endl;
    cout << "set tcfdt to: "        << tcfdt        << endl;
    cout << "set HODtype to: "      << HODtype      << endl;
    cout << "set nlistcut to: "     << nlistcut     << endl;
    cout << "set nlistmax to: "     << nlistmax     << endl;

    // determine number of chromophores
    nchrom = nmol*nchrom_mol; // number of chromophores per frame

    // set charges on atom -- note these are tip4p charges because that is what
    // the maps were parameterized for
    charge[ OW  ] = 0.; 
    charge[ HW1 ] = 0.52; 
    charge[ HW2 ] = 0.52; 
    charge[ MW  ] = -1.04;

    // allocate arrays
    eproj     = new float[nchrom]();
    dipole_t0 = new rvec[ nchrom ]();
    dipole    = new rvec[ nchrom ]();
    alpha_t0  = new matrix[ nchrom ]();
    alpha     = new matrix[ nchrom ]();
    irtcf     = new complex<double>[ ntcfpoints ]();
    vvtcf     = new complex<double>[ ntcfpoints ]();
    vhtcf     = new complex<double>[ ntcfpoints ]();
    Firtcf    = new double[ ntcfpoints + nzeros ]();
    Fvvtcf    = new double[ ntcfpoints + nzeros ]();
    Fvhtcf    = new double[ ntcfpoints + nzeros ]();
    Fomega    = new double[ ntcfpoints + nzeros ]();
    propigator= new complex<double>[ nchrom ]();
    nlist     = new int[ nmol*nlistmax ]();
    lastox    = new rvec[ nmol ]();

    // set the maps
    if ( species == "HOD/H2O" ){
        // DO stretch parameters
        map_w[0] = 2767.8; map_w[1] = -2630.3; map_w[2] = -102601.0;
        map_x[0] = 0.16593; map_x[1] = -2.0632E-5;
        map_p[0] = 2.0475; map_p[1] = 8.9108E-4;
        map_muprime[0] = 0.1646; map_muprime[1] = 11.39; map_muprime[2] = 63.41;
        avef = 2500.;
    }
    else if ( species == "HOD/D2O" ){
        // HO stretch parameters
        map_w[0] = 3760.2; map_w[1] = -3541.7; map_w[2] = -152677.0;
        map_x[0] = 0.19285; map_x[1] = -1.7261E-5;
        map_p[0] = 1.6466; map_p[1] = 5.7692E-4;
        map_muprime[0] = 0.1646; map_muprime[1] = 11.39; map_muprime[2] = 63.41;
        avef = 3400.;
    }
    else{
        cout << "WARNING:: species: " << species << " unknown. Aborting..." << endl;
        cout << "I can only calculate dilute spectra of HOD in H2O or D2O" << endl;
        exit(EXIT_FAILURE);
    }

    if ( HODtype != "explicit" and HODtype != "implicit" ){
        cout << "Warning:: HODtype: " << HODtype << " unknown. Aborting..." << endl;
        cout << "Valid options for HODtype are 'explicit' and 'implicit'" << endl;
        exit(EXIT_FAILURE);
    }

    // initialize the last ox position to large number;
    for ( int mol = 0; mol < nmol; mol ++ ){
        for ( int i = 0; i < 3; i++ ) lastox[ mol ][i] = 1E8;
    }

}

model::~model()
// Default Destructor
{
    delete [] eproj;
    delete [] dipole_t0;
    delete [] dipole;
    delete [] alpha_t0;
    delete [] alpha;
    delete [] irtcf;
    delete [] vvtcf;
    delete [] vhtcf;
    delete [] Firtcf;
    delete [] Fvvtcf;
    delete [] Fvhtcf;
    delete [] Fomega;
    delete [] propigator;
    delete [] nlist;
    delete [] lastox;
}

void model::adjust_Msite()
// set msite OH distance to tip4p geometry
{
    #pragma omp parallel for
    for ( int mol = 0; mol < nmol; mol ++ ){
        int   i;
        float om_vec[3], r;

        // the OM unit vector
        for ( i = 0; i < 3; i ++ ) om_vec[i] = x[ mol*natoms_mol + MW ][i] \
                                             - x[ mol*natoms_mol + OW ][i];
        minImage( om_vec );
        r = mag3( om_vec );
        for ( i = 0; i < 3; i ++ ) om_vec[i] /= r;

        // set the m site based on the unit vector
        for ( i = 0; i < 3; i ++ ) x[ mol*natoms_mol + 3 ][ i ] = x[ mol*natoms_mol + OW ][i] \
                                                                + om_vec[i]*0.0150;
    }
}

int model::get_chrom_nx( int mol, int h )
// return chromophore index
{
    if ( mol > nmol ){
        cout << "get_chrom_nx:: invalid argument, mol > nmol. Aborting." << endl;
        exit(EXIT_FAILURE);
    }
    if ( h > 2 or h < 1 ){
        cout << "get_chrom_nx:: invalid argument, h != 1 or 2. Aborting." << endl;
        exit(EXIT_FAILURE);
    }
    return mol*nchrom_mol + h - 1;
}

void model::update_nlist()
// calculate the electric field projection onto each H atom
{
    const float efieldCutoff = 0.7831; // cutoff radius for efield
    float pad, dispmax;
    bool update;

    pad = nlistcut - efieldCutoff;
    dispmax = 0;
    update = false;

    // find maximum oxygen displacement and update only if greater than the padding of the nlist
    for ( int mol1 = 0; mol1 < nmol; mol1 ++ ){
        float disp_vec[3], disp;
        for ( int i = 0; i < 3; i ++ ) disp_vec[i] = x[ mol1 * natoms_mol + OW ][i] \
                                                   - lastox[ mol1 ][i];
        minImage( disp_vec );
        disp = mag3( disp_vec );
        if ( disp > dispmax ) {
            dispmax = disp;
            if ( dispmax > (pad-0.09572)/2. ){
                // worst case scenario, all atoms could move as far as (pad-0.09572)/2 and all
                // relevant H atoms inside the efield cutoff would still be in the neighbor list
                // here 0.09572 is the OH bond length for tip4p geometry
                update = true;
                break;
            }
        }
    }

    if ( update ){
        cout << endl << "Updating neighbor list..." << endl;
        // reset neighborlist to negative 1
        for ( int i = 0; i < nmol*nlistmax; i ++ ) nlist[i] = -1;

        // loop over all reference chromophores
        #pragma omp parallel for
        for ( int mol1 = 0; mol1 < nmol; mol1 ++ ){
            int   mol2, i;
            float r;
            float oo_vec[3];    // oh vector of molecule 1
            int   nlisti;       // neighbor list index
            nlisti = 0;         // reset neighbor list index

            // update last positions for neighbor list update
            for (i = 0; i < 3; i ++ ) lastox[ mol1 ][i] = x[ mol1 * natoms_mol + OW ][i];

            for ( mol2 = 0; mol2 < nmol; mol2 ++ ){
                // dont care about self
                if ( mol1 == mol2 ) continue;

                // get distance between oxygen atoms
                for ( i = 0; i < 3; i ++ ) oo_vec[i] = x[ mol1 * natoms_mol + OW ][i] \
                                                     - x[ mol2 * natoms_mol + OW ][i];
                minImage( oo_vec );
                r = mag3( oo_vec );
                if ( r < nlistcut ){
                    nlist[ mol1*nlistmax + nlisti ] = mol2;
                    nlisti ++;
                }
                if ( nlisti >= nlistmax ){
                    cout << "Warning:: nlisti >= nlistmax. Aborting.\nMake sure nlilst max is big enough for nlistcut" << endl;
                    exit(EXIT_FAILURE);
                }
            }
        }
        cout << "Done." << endl;
    }
}

void model::get_efield()
// calculate the electric field projection onto each H atom
{
    const float efieldCutoff = 0.7831; // cutoff radius for efield
    const float bohr_nm      = 18.8973;// convert from nm to bohr

    // adjust the position of all msite atoms if using either e3b3 or tip4p2005
    adjust_Msite();

    // loop over all reference chromophores
    #pragma omp parallel for
    for ( int mol1 = 0; mol1 < nmol; mol1 ++ ){
        int   mol2, h1, a2, i, chrom, neigh;
        float r;
        float efield_vec[3];    // electric field vector
        float mol1oh_vec[3];    // oh vector of molecule 1
        float mol12ho_vec[3];   // the oh vector between molecule 2 and 1
        float a2_vec[3];        // vector of 2nd atom
        float dr[3];            // distance vector
        for ( h1 = 1; h1 < 3; h1 ++ ){

            //initialize electric field to zero
            for ( i = 0; i < 3; i ++ ) efield_vec[i] = 0.;

            // the oh unitvector
            for ( i = 0; i < 3; i ++ ) mol1oh_vec[i] = x[ mol1 * natoms_mol + h1 ][i] \
                                                     - x[ mol1 * natoms_mol + OW ][i];
            minImage( mol1oh_vec );
            r = mag3( mol1oh_vec );
            for ( i = 0; i < 3; i ++ ) mol1oh_vec[i] /= r;

            // loop over all other atoms to deterime the electric field
            for ( neigh = 0; neigh < nlistmax; neigh ++ ){
                mol2 = nlist[ mol1*nlistmax + neigh ];
                if ( mol2 == -1 ) break; // nlist will equal -1 when neigh > number of neighbors < nlistcut
                
                // distance between reference H and mol2 O
                for ( i = 0; i < 3; i ++ ) mol12ho_vec[i] = x[ mol1 * natoms_mol + h1 ][i] - \
                                                            x[ mol2 * natoms_mol + OW ][i];
                minImage( mol12ho_vec );
                r = mag3( mol12ho_vec );
                if ( r > efieldCutoff ) continue;

                // loop over atoms with charges on molecule 2
                for ( a2 = 1; a2 < 4; a2 ++ ){

                    for ( i = 0; i < 3; i ++ ) dr[i] = x[ mol1 * natoms_mol + h1 ][i] \
                                                     - x[ mol2 * natoms_mol + a2 ][i];
                    minImage( dr );
                    // convert to bohr so efield is in au
                    for ( i = 0; i < 3; i ++ ) dr[i] *= bohr_nm;
                    r = mag3( dr );

                    // add contribution of current atom to the electric field
                    for ( i = 0; i < 3; i ++ ) efield_vec[i] += charge[a2]*dr[i]/(r*r*r);

                }
            }
            // project the efield along the oh bond
            chrom = get_chrom_nx( mol1, h1 );
            eproj[ chrom ] = dot3( efield_vec, mol1oh_vec );
        }
    }
}

void model::get_alpha_mu()
// determine the transition dipole moment unit vectors
{
    #pragma omp parallel for
    for ( int mol = 0; mol < nmol; mol ++ ){
        int h, chrom, i, j;
        float oh_vec[3], r;
        float muprime, x10, omega10;
        for ( h = 1; h <3; h ++ ){
            for ( i = 0; i < 3; i ++ ) oh_vec[i] = x[ mol*natoms_mol + h ][i] \
                                                 - x[ mol*natoms_mol + OW][i];
            minImage( oh_vec );
            r = mag3( oh_vec );

            chrom   = get_chrom_nx( mol, h );
            omega10 = get_omega10( eproj[chrom] );
            muprime = get_muprime( eproj[chrom] );
            x10     = get_x10( omega10 );
            // mu for ir spectrum
            for ( i = 0; i < 3; i ++ ){
                dipole[ chrom ][i] = muprime*x10*oh_vec[i] / r;
                if ( HODtype == "explicit" ){
                    if ( species == "HOD/H2O" and h==1 ){ // make OH vibration dark
                        if ( h == 1 ) dipole[ chrom ][i] = 0.;
                    }
                    else if ( species == "HOD/D2O" and h==2 ){ // make OD vibration dark
                        if ( h == 2 ) dipole[ chrom ][i] = 0.;
                    }
                }
            }
            // alpha for raman spectrum
            for ( i = 0; i < 3; i ++ ){
                for ( j = 0; j < 3; j ++ ){
                    // off diagonal and diagonal
                    alpha[ chrom ][i][j] = 4.6*oh_vec[i]*oh_vec[j]*x10/(r*r);
                    if ( HODtype == "explicit" ){
                        if ( species == "HOD/H2O" and h==1 ){ // make OH vibration dark
                            if ( h == 1 ) alpha[ chrom ][i][j] = 0.;
                        }
                        else if ( species == "HOD/D2O" and h==2 ){ // make OD vibration dark
                            if ( h == 2 ) alpha[ chrom ][i][j] = 0.;
                        }
                    }
                }
                alpha[ chrom ][i][i] += x10; // diagonal have an extra factor
                if ( HODtype == "explicit" ){
                    if ( species == "HOD/H2O" and h==1 ){ // make OH vibration dark
                        if ( h == 1 ) alpha[ chrom ][i][j] = 0.;
                    }
                    else if ( species == "HOD/D2O" and h==2 ){ // make OD vibration dark
                        if ( h == 2 ) alpha[ chrom ][i][j] = 0.;
                    }
                }
            }
        }
    }
}

void model::set_alpha_mu_t0()
// set transition dipole moment vector at t0
{
    #pragma omp parallel for
    for ( int chrom = 0; chrom < nchrom; chrom ++ ){
        for ( int i = 0; i < 3; i ++ ){
            dipole_t0[ chrom ][i] = dipole[chrom][i];
            for ( int j = 0; j < 3; j ++ ) alpha_t0[ chrom ][i][j] = alpha[chrom][i][j];
        }
    }
}

float model::get_omega10( float efield ){
// get omega from efield map
    float omega10;

    omega10 = map_w[0] + map_w[1] * efield + map_w[2] * efield * efield;
    return omega10;
}

float model::get_x10( float omega10 ){
// get x10 from efield map
    float x10;
    x10 = map_x[0] + map_x[1] * omega10;
    return x10;
}

float model::get_p10( float omega10){
// get p10 from efield map
    float p10;
    p10 = map_p[0] + map_p[1] * omega10;
    return p10;
}

float model::get_muprime( float efield ){
// get muprime from efield map
    float muprime;
    muprime = map_muprime[0] + map_muprime[1]*efield + map_muprime[2]*efield*efield;
    return muprime;
}

void model::get_tcf_dilute( int tcfpoint )
// determine dipole time correlation function for dilute HOD in D2O or H2O
{
    double irReal, irImag, vvReal, vvImag, vhReal, vhImag;
    irReal = 0.; irImag=0.; vvReal=0.; vvImag=0.; vhReal=0.; vhImag=0.;

    #pragma omp parallel for reduction(+:irReal, irImag, vvReal, vvImag, vhReal, vhImag)
    for ( int chrom = 0; chrom < nchrom; chrom ++ ){
        int i, j;
        complex<double> ir_prefactor, arg, vv_prefactor, vh_prefactor;
        float omega10;
        float dipole_t0_vec[3], dipole_vec[3];

        // update the propigator
        if ( tcfpoint != 0 ){ // at t=0, the propigator is 1
            omega10 = get_omega10( eproj[chrom] ) - avef; // subtract off average frequency 
                                                      // to avoid high frequency oscillations 
                                                      // -- need to add it back later
            arg = (complex<double>){ 0., omega10 * tcfdt / hbar };
            propigator[ chrom ] *= exp(arg);
        }

        // ir spectrum
        for ( i = 0; i < 3; i ++ ){
            dipole_t0_vec[i] = dipole_t0[chrom][i];
            dipole_vec[i]    = dipole[chrom][i];
        }
        ir_prefactor = (complex<double>){dot3( dipole_t0_vec, dipole_vec ), 0};
        irReal += real(ir_prefactor*propigator[ chrom ]);
        irImag += imag(ir_prefactor*propigator[ chrom ]);

        // vv raman spectrum
        vv_prefactor = complex_zero;
        for ( i = 0; i < 3; i ++ ){
            for ( j = 0; j < 3; j ++ ){
                vv_prefactor += (complex<double>) \
                                { (alpha_t0[chrom][i][i]*alpha[chrom][j][j]  \
                              + 2.*alpha_t0[chrom][i][j]*alpha[chrom][i][j]) \
                              / 15., 0};
            }
        }
        vvReal += real(vv_prefactor * propigator[chrom]);
        vvImag += imag(vv_prefactor * propigator[chrom]);
        
        // vh raman spectrum
        vh_prefactor = complex_zero;
        for ( i = 0; i < 3; i ++ ){
            for ( j = 0; j < 3; j ++ ){
                vh_prefactor += (complex<double>) \
                                { (3.*alpha_t0[chrom][i][j]*alpha[chrom][i][j]  \
                                     -alpha_t0[chrom][i][i]*alpha[chrom][j][j]) \
                                /30.,0.};
            }
        }
        vhReal += real(vh_prefactor * propigator[chrom]);
        vhImag += imag(vh_prefactor * propigator[chrom]);
    }

    // collect everything at the end
    irtcf[ tcfpoint ] += complex<double>{ irReal, irImag };
    vvtcf[ tcfpoint ] += complex<double>{ vvReal, vvImag };
    vhtcf[ tcfpoint ] += complex<double>{ vhReal, vhImag };
}

void model::reset_propigator()
// reset the propigator to ones for t=0
{
    #pragma omp parallel for
    for ( int chrom = 0; chrom < nchrom; chrom ++ ){
        propigator[ chrom ] = complex_one;
    }
}

void model::do_ffts()
// fourier transform tcf to get the spectrum
{
    fftw_plan plan;
    complex<double> tcf[nzeros + ntcfpoints];
    double Ftcf[nzeros + ntcfpoints];
    double convert;
    int i;

    // ir spectrum
    for ( i = 0; i < ntcfpoints + nzeros; i++ ) tcf[i] = complex_zero;
    for ( i = 0; i < ntcfpoints; i++ ) tcf[i] = irtcf[ i ]*pow(-1.,i);// last mult puts zero 
                                                                      // freq in center of array
    plan = fftw_plan_dft_c2r_1d( nzeros + ntcfpoints, \
            reinterpret_cast<fftw_complex*>(tcf), Ftcf, FFTW_ESTIMATE );
    fftw_execute(plan);
    // C2R transform is alway inverse, so the spectrum is "backwards", here make it forwards
    // and normalize it
    convert = 2.*PI*hbar/(tcfdt*(ntcfpoints+nzeros));
    for ( i = 0; i < ntcfpoints + nzeros; i ++ ){
        Firtcf[i] = Ftcf[ ntcfpoints + nzeros - i - 1 ]/(convert*(ntcfpoints+nzeros));
        Fomega[i] = (i-(ntcfpoints+nzeros)/2+1)*convert + avef;
    }


    // vv spectrum
    for ( i = 0; i < ntcfpoints + nzeros; i++ ) tcf[i] = complex_zero;
    for ( i = 0; i < ntcfpoints; i++ ) tcf[i] = vvtcf[ i ]*pow(-1.,i);// last mult puts zero 
                                                                      // freq in center of array
    plan = fftw_plan_dft_c2r_1d( nzeros + ntcfpoints, \
            reinterpret_cast<fftw_complex*>(tcf), Ftcf, FFTW_ESTIMATE );
    fftw_execute(plan);
    for ( i = 0; i < ntcfpoints + nzeros; i ++ ){
        Fvvtcf[i] = Ftcf[ ntcfpoints + nzeros - i - 1 ]/(convert*(ntcfpoints+nzeros));
    }

    // vh spectrum
    for ( i = 0; i < ntcfpoints + nzeros; i++ ) tcf[i] = complex_zero;
    for ( i = 0; i < ntcfpoints; i++ ) tcf[i] = vhtcf[ i ]*pow(-1.,i);// last mult puts zero 
                                                                      // freq in center of array
    plan = fftw_plan_dft_c2r_1d( nzeros + ntcfpoints, \
            reinterpret_cast<fftw_complex*>(tcf), Ftcf, FFTW_ESTIMATE );
    fftw_execute(plan);
    for ( i = 0; i < ntcfpoints + nzeros; i ++ ){
        Fvhtcf[i] = Ftcf[ ntcfpoints + nzeros - i - 1 ]/(convert*(ntcfpoints+nzeros));
    }
}

void model::write_tcf()
// write time correlation functions to a file
{
    int tcfpoint;
    string fname;
    FILE *file;
    
    // ir spec
    fname = outf+"-irrtcf.dat";
    file = fopen(fname.c_str(),"w");
    fprintf( file, "#t (ps) irtcf.real\n");
    for ( tcfpoint = 0; tcfpoint < ntcfpoints; tcfpoint ++ ){
        fprintf( file, "%g %g \n", tcfpoint*tcfdt, irtcf[tcfpoint].real() );
    }
    fclose( file );

    fname = outf+"-iritcf.dat";
    file = fopen(fname.c_str(),"w");
    fprintf( file, "#t (ps) irtcf.imag\n");
    for ( tcfpoint = 0; tcfpoint < ntcfpoints; tcfpoint ++ ){
        fprintf( file, "%g %g \n", tcfpoint*tcfdt, irtcf[tcfpoint].imag() );
    }
    fclose( file );

    // vv spec
    fname = outf+"-vvrtcf.dat";
    file = fopen(fname.c_str(),"w");
    fprintf( file, "#t (ps) vvtcf.real\n");
    for ( tcfpoint = 0; tcfpoint < ntcfpoints; tcfpoint ++ ){
        fprintf( file, "%g %g \n", tcfpoint*tcfdt, vvtcf[tcfpoint].real() );
    }
    fclose( file );

    fname = outf+"-vvitcf.dat";
    file = fopen(fname.c_str(),"w");
    fprintf( file, "#t (ps) vvtcf.imag\n");
    for ( tcfpoint = 0; tcfpoint < ntcfpoints; tcfpoint ++ ){
        fprintf( file, "%g %g \n", tcfpoint*tcfdt, vvtcf[tcfpoint].imag() );
    }
    fclose( file );

    // vh spec
    fname = outf+"-vhrtcf.dat";
    file = fopen(fname.c_str(),"w");
    fprintf( file, "#t (ps) vhtcf.real\n");
    for ( tcfpoint = 0; tcfpoint < ntcfpoints; tcfpoint ++ ){
        fprintf( file, "%g %g \n", tcfpoint*tcfdt, vhtcf[tcfpoint].real() );
    }
    fclose( file );

    fname = outf+"-vhitcf.dat";
    file = fopen(fname.c_str(),"w");
    fprintf( file, "#t (ps) vhtcf.imag\n");
    for ( tcfpoint = 0; tcfpoint < ntcfpoints; tcfpoint ++ ){
        fprintf( file, "%g %g \n", tcfpoint*tcfdt, vhtcf[tcfpoint].imag() );
    }
    fclose( file );

}

void model::write_spec()
// write spectra to a file
{
    int i;
    string fname;
    FILE *file;
    
    // ir spectrum
    fname = outf+"-irls.dat";
    file = fopen(fname.c_str(),"w");
    fprintf( file, "#omega (cm-1) lineshape\n");
    for ( i = 0; i < ntcfpoints + nzeros; i ++ ){
        fprintf( file, "%g %g\n", Fomega[i], Firtcf[i] );
    }
    fclose(file);

    // vv spectrum
    fname = outf+"-vvls.dat";
    file = fopen(fname.c_str(),"w");
    fprintf( file, "#omega (cm-1) lineshape\n");
    for ( i = 0; i < ntcfpoints + nzeros; i ++ ){
        fprintf( file, "%g %g\n", Fomega[i], Fvvtcf[i] );
    }
    fclose(file);

    // vh spectrum
    fname = outf+"-vhls.dat";
    file = fopen(fname.c_str(),"w");
    fprintf( file, "#omega (cm-1) lineshape\n");
    for ( i = 0; i < ntcfpoints + nzeros; i ++ ){
        fprintf( file, "%g %g\n", Fomega[i], Fvhtcf[i] );
    }
    fclose(file);

}

void printProgress( int currentStep, int totalSteps )
// print a progress bar to keep updated on usage
{
    float percentage = (float) currentStep / (float) totalSteps;
    int lpad = (int) (percentage*PWID);
    int rpad = PWID - lpad;
    fprintf(stderr, "\r [%.*s%*s]%3d%%", lpad, PSTR, rpad, "",(int) (percentage*100));
}

// ************************************************************************ 
int main( int argc, char* argv[] )
{

    int     currentSample, frameno, tcfpoint;
    float   currentTime, tcfTime;

    // Check program input
    if ( argc != 2 ){
        printf("Program expects only one argument, which is the name of \n\
                an input file containing the details of the analysis.\nAborting...\n");
        exit(EXIT_FAILURE);
    }

    // get filename for parameters
    string inpf(argv[1]);

    // initialize class 
    model reader( inpf );

    // calculate the spectra
    cout << endl \
         << "************************************************************************" << endl;
    for ( currentSample = 0; currentSample < reader.nsamples; currentSample ++ ){
        currentTime = currentSample * reader.sampleEvery + reader.beginTime;
        fprintf(stderr, "\n    Now processing sample %d/%d starting at %.2f ps\n", \
                currentSample + 1, reader.nsamples, currentTime );
        fflush(stderr);
        for ( tcfpoint = 0; tcfpoint < reader.ntcfpoints; tcfpoint ++ ){
           
            // get current time and read time frame
            tcfTime = currentTime + tcfpoint * reader.tcfdt;
            frameno = reader.get_frame_number( tcfTime );
            reader.read_frame( frameno );
            if ( reader.checktime( tcfTime ) == false ){
                cout << endl << "Warning:: checktime failed. gmxtime is: " << reader.gmxtime << endl;
                cout << "tcfTime is: " << tcfTime << endl;
                cout << "Aborting." << endl;
                exit(EXIT_FAILURE);
            }

            // update neighbor list
            reader.update_nlist();

            reader.get_efield();
            reader.get_alpha_mu();
            if ( tcfpoint == 0 ){
                reader.set_alpha_mu_t0();
                reader.reset_propigator();
            }
            reader.get_tcf_dilute( tcfpoint );

            printProgress( tcfpoint , reader.ntcfpoints );
        }   
    }

    // normalize the time correlation function and multiply by relaxation time
    for ( tcfpoint = 0; tcfpoint < reader.ntcfpoints; tcfpoint ++ ){
        reader.irtcf[ tcfpoint ] *= (complex<double>)\
                {exp(-1.*tcfpoint*reader.tcfdt/(2.0*reader.t1))/(1.*reader.nsamples),0.};
        reader.vvtcf[ tcfpoint ] *= (complex<double>)\
                {exp(-1.*tcfpoint*reader.tcfdt/(2.0*reader.t1))/(1.*reader.nsamples),0.};
        reader.vhtcf[ tcfpoint ] *= (complex<double>)\
                {exp(-1.*tcfpoint*reader.tcfdt/(2.0*reader.t1))/(1.*reader.nsamples),0.};
    }

    // perform the fft to get the spectrum
    reader.do_ffts();

    // write output files
    reader.write_tcf(); // write tcf to file
    reader.write_spec();

    cout << endl << endl << endl << "DONE!" << endl;
}
