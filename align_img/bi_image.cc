#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include "bi_image.hh"
#include "tiffio.h"
#include "png_output.hh"

#ifdef _OPENMP
#include "omp.h"
inline double bi_image::wtime() {return omp_get_wtime();}
inline int bi_image::thread_num() {return omp_get_thread_num();}
#else
#include <ctime>
inline double bi_image::wtime() {return double(clock())/CLOCKS_PER_SEC;}
inline int bi_image::thread_num() {return 0;}
#endif

/** Global function used to link the GSL optimizer back to the class routine
 * for evaluating the fit function S. */
double bi_f(const gsl_vector *v,void *params) {
    return reinterpret_cast<bi_image*>(params)->fun(v->data);
}

/** Global function used to link the GSL optimizer back to the class routine
 * for evaluating the derivative of the fit function S. */
void bi_df(const gsl_vector *v,void *params,gsl_vector *df) {
    reinterpret_cast<bi_image*>(params)->dfun(v->data,df->data);
}

/** Global function used to link the GSL optimizer back to the class routine
 * for evaluating the fit function S and its derivative. */
void bi_fdf(const gsl_vector *v,void *params,double *f,gsl_vector *df) {
    *f=reinterpret_cast<bi_image*>(params)->fun_dfun(v->data,df->data);
}

/** Converts a floating point value into a byte for a PNG color channel at a
 * pixel.
 * \param[in] val the value to consider.
 * \return The PNG byte. */
inline png_byte png_scale(double val) {
    if(val<0) return 0;
    else if(val>255) return 255;
    return static_cast<png_byte>(val);
}

/** Constructs the image fitting class by reading in a grid of field values
 * from a MATLAB MAT file.
 * \param[in] mfile the filename of the TIF file.
 * \param[in] q_ the maximum degree polynomial allowing in fitting. */
bi_image::bi_image(const char* tfile,int q_) : q(q_), dof(2*q*q), ftype(0),
    pin(false), g(NULL), of(NULL), al(new double[dof]) {
    

    // Open the TIF file using the TIFF library
    TIFF* tif = TIFFOpen(tfile, "r");
    // #define uint32 unsigned long
    uint32 width, height;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);           // uint32 width;
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);        // uint32 height;
    // Get space to store the image
    size_t npixels=width*height;
    uint32 *tmp;
    tmp=(uint32 *) _TIFFmalloc(npixels *sizeof(uint32));

    if(TIFFReadRGBAImage(tif, width, height, tmp, 0)){
        puts("read img");
    }
    else{
        fprintf(stderr,"Error opening TIF file %s\n",tfile);
        exit(1);
    }

    // Begin custom edit to convert to transpose
    m=(int) width;
    n=(int) height;
    f=new vec3[mn=m*n];
    int v;
    for(int i=0;i<m;i++) for(int j=0;j<n;j++){
        v=tmp[j*m+i];
        f[j*m+i]=vec3(v&255,(v>>8)&255,(v>>16)&255);
    }
    _TIFFfree(tmp);

    //******** f is a m*n double array

    // The images sometimes have NaNs in them. Convert them to zeros.
    //for(double *fp=f;fp<f+mn;fp++) if(isnan(*fp)) *fp=0.;
    setup_common();

    TIFFClose(tif);
}

/** Constructs the image fitting class setting up a rectangular grid of field
 * values without initializing them.
 * \param[in] (m_,n_) the dimensions of the grid.
 * \param[in] q_ the maximum degree polynomial allowing in fitting. */
bi_image::bi_image(int m_,int n_,int q_) : q(q_), dof(2*q*q), ftype(0), m(m_),
    n(n_), mn(m*n), pin(false), f(new vec3[mn]), g(NULL), of(NULL),
    al(new double[dof]) {
    setup_common();
}

/** Constructs the image fitting class by copying the quantities from another.
 * \param[in] b the class to copy. */
bi_image::bi_image(bi_image &b) : q(b.q), dof(b.dof), ftype(b.ftype), m(b.m),
    n(b.n), mn(b.mn), nt(b.nt), pin(b.pin), dx(b.dx), dy(b.dy), ax(b.ax),
    ay(b.ay), f(new vec3[mn]), g(NULL), of(NULL), al(new double[dof]),
    ilo(b.ilo), ihi(b.ihi), jlo(b.jlo), jhi(b.jhi) {
    memcpy(f,b.f,mn*sizeof(vec3));
    setup_common(false);
}
/** Constructs the image fitting class by copying the quantities from another
 * and downsampling the grid.
 * \param[in] b the class to copy.
 * \param[in] d the downsampling factor. */
bi_image::bi_image(bi_image &b,int d) : q(b.q), dof(b.dof), ftype(b.ftype),
    m(b.m/d), n(b.n/d), mn(m*n), nt(b.nt), pin(b.pin), dx(d*b.dx),
    dy(d*b.dy), ax(b.ax+0.5*b.dx*(d-1)), ay(b.ay+0.5*b.dy*(d-1)),
    f(new vec3[mn]), g(NULL), of(NULL), al(new double[dof]),
    ilo(b.ilo/d), ihi(b.ihi/d), jlo(b.jlo/d), jhi(b.jhi/d) {

    // Downsample the field values by a factor of d
    double fac=1./(d*d);
#pragma omp parallel for
    for(int j=0;j<n;j++) {
        vec3 *fp=f+j*m;
        for(int i=0;i<m;i++) {
            vec3 *bf=b.f+d*(b.m*j+i);
            *fp=0;
            for(int l=0;l<b.m*d;l+=b.m) for(int k=0;k<d;k++)
                *fp+=bf[l+k];
            *(fp++)*=fac;
        }
    }
    setup_common(false);
}

/** The class destructor frees the dynamically allocated memory. */
bi_image::~bi_image() {
    if(g!=NULL) delete [] g;
    delete [] Ty;
    delete [] Tx;
    for(int k=nt-1;k>=0;k--) {
        delete bic[k];
        delete grl[k];
    }
    delete [] bic;
    delete [] grl;
    delete [] al;
    delete [] f;
}

/** Sets various constants and initializes various arrays that are common
 * for all class constructors.
 * \param[in] grid_constants whether to set up the grid constants as well. */
void bi_image::setup_common(bool grid_constants) {

    if(grid_constants) {

        // Set the number of threads
#ifdef _OPENMP
        nt=omp_get_max_threads();
        //nt=16;
#else
        nt=1;
#endif

        // Set the grid size
        dx=2./m;dy=2./n;
        ax=-1+0.5*dx;ay=-1+0.5*dy;

        // Set the range over gridpoints over which to optimize the mapping
        // function
        ilo=0;ihi=m;
        jlo=0;jhi=n;
    }

    // Allocate workspace and bicubic interpolation classes for
    // multithreaded computations
    grl=new double*[nt];
    bic=new bicubic_interp*[nt];
#pragma omp parallel
    {
        int tn=thread_num();
        grl[tn]=new double[dof];
        bic[tn]=new bicubic_interp(f,m,n,ax,-ax,ay,-ay);
    }

    // Set up the Chebyshev tables
    chebyshev_setup(Tx,m,ax,dx);
    chebyshev_setup(Ty,n,ay,dy);
}

/** Sets up a table of Chebyshev polynomial values evaluated at regular
 * intervals.
 * \param[in] T a reference to a pointer in which to set up the table.
 * \param[in] r the number of values to write.
 * \param[in] a the first position to evaluate at.
 * \param[in] d the step size between positions. */
void bi_image::chebyshev_setup(double *&T,int r,double a,double d) {
    T=new double[q*r];

    // Set up the zeroth Chebyshev polynomial
    for(int i=0;i<r;i++) T[i]=1.;
    if(q>1) {

        // Set up the first Chebyshev polynomial
        for(int i=0;i<r;i++) T[r+i]=a+i*d;

        // Set up the remaining Chebyshev polynomials using the
        // recursion relation
        for(double *Tp=T+2*r;Tp<T+q*r;Tp+=r) for(int i=0;i<r;i++)
            Tp[i]=2*T[r+i]*Tp[-r+i]-Tp[-2*r+i];
    }

    // Apply multiplication by a stretched factor of (1-x*x) if pinning is
    // enabled
    if(pin) chebyshev_pin(T,r,a,d);
}

/** Multiplies a table of Chebyshev polynomials by a stretched version of
 * (1-x*x) in order to pin the map to the end points.
 * \param[in] T a reference to the table.
 * \param[in] r the number of values to write.
 * \param[in] a the first position to evaluate at.
 * \param[in] d the step size between positions. */
void bi_image::chebyshev_pin(double *&T,int r,double a,double d) {
    for(int i=0;i<r;i++) {
        double z=d*d*(i+0.5)*(r-i-0.5);
        for(int k=0;k<q;k++) T[k*r+i]*=z;
    }
}

/** Smoothes the field values by applying a five-point averaging stencil.
 * \param[in] s the size of the smoothing to apply. */
void bi_image::smooth(double s) {
    if(g==NULL) g=new vec3[mn];

    // Use the g array to create a smoothed version of the f array, using a
    // five-point stencil
#pragma omp parallel for
    for(int j=0;j<n;j++) {
        vec3 *fp=f+j*m,*fs=fp,*fe=fp+m,*fpen=fe-1,*gp=g+j*m;
        while(fp<fe) {
            vec3 s1(0);int s2=0;
            if(j>0) {s1+=fp[-m];s2++;}
            if(j<n-1) {s1+=fp[m];s2++;}
            if(fp>fs) {s1+=fp[-1];s2++;}
            if(fp<fpen) {s1+=fp[1];s2++;}
            *(gp++)=*(fp++)*(1-s2*s)+s*s1;
        }
    }

    // Swap the pointers to the f and g array, so that smoothed field is
    // now in the main f array
    vec3 *h=g;g=f;f=h;
}

/** Smoothes the image by integrating the bicubic interpolant exactly along
 * lines of length 2r and orientation theta centered about each pixel.
 * \param[in] b a bi_image class with the same dimensions in which to store the
 *        output.
 * \param[in] r the half-length of the line.
 * \param[in] thetat the orientation of the line. */
void bi_image::line_smooth(bi_image &b,double r,double theta) {
    double cth=r*cos(theta),sth=r*sin(theta),fac=0.5/r;
#pragma omp parallel
    {
        bicubic_interp *bip=bic[thread_num()];
#pragma omp for
        for(int j=0;j<n;j++) {
            vec3 *fp=b.f+j*m;
            double x,y=ay+j*dy,ylo=y-sth,yhi=y+sth;
            for(int i=0;i<m;i++,fp++) {
                x=ax+i*dx;
                *fp=fac*(bip->line_integral(x-cth,ylo,x+cth,yhi));
            }
        }
    }
}

/** Initalizes the field values to be a sum of Gaussians in a ring, with
 * slightly perturbed positions. */
void bi_image::init_test() {

    // Initialize Gaussian centers
    const int o=10;
    const double step=2*M_PI/o;
    double z[2*o],*zp=z;
    for(int k=0;k<o;k++) {
        *(zp++)=0.6*cos(step*k)+rshift(0.02);
        *(zp++)=0.6*sin(step*k)+rshift(0.02);
    }

    // Initialize the field as a sum of Gaussians
#pragma omp parallel
    for(int j=0;j<n;j++) {
        double y=ay+j*dy;
        vec3 *fp=f+j*m;
        for(int i=0;i<m;i++,fp++) {
            double x=ax+i*dx,delx,dely;
            *fp=vec3(0);
            for(double *zp=z;zp<z+2*o;) {
                delx=x-*(zp++);
                dely=y-*(zp++);
                fp->x+=exp(-50*(delx*delx+dely*dely));
                fp->y+=exp(-100*(delx*delx+dely*dely));
                fp->z+=exp(-200*(delx*delx+dely*dely));
            }
        }
    }
}

/** Finds the best fit mapping function between this field and another.
 * \param[in] other_ the other bi_image class to fit to.
 * \param[in] al_ a starting guess for the mapping.
 * \return The total wall clock time to do the fitting. */
double bi_image::fit(bi_image &b,double *al_) {
    int iter=0,status,piter=0;
    double tstart=wtime(),t0=0;
    of=b.f;

    // Inititalize the function to be minimized, linking the elements of
    // my_func to the parabolic function example
    gsl_multimin_function_fdf my_func;
    my_func.n=dof;
    my_func.f=bi_f;
    my_func.df=bi_df;
    my_func.fdf=bi_fdf;
    my_func.params=this;

    // Set the starting point for the minimization
    gsl_vector *x=gsl_vector_alloc(dof);
    if(al_==NULL) gsl_vector_set_zero(x);
    else memcpy(x->data,al_,dof*sizeof(double));

    // Initialize the BFGS minimizer
    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;
    T=gsl_multimin_fdfminimizer_vector_bfgs2;
    s=gsl_multimin_fdfminimizer_alloc(T,dof);
    gsl_multimin_fdfminimizer_set(s,&my_func,x,0.01,1e-3);

    // Do the BFGS iterations
    piter=0;
    do {
        if(iter%1==0) message(piter,iter,s,t0);
        iter++;
        if(gsl_multimin_fdfminimizer_iterate(s)) break;
        status=gsl_multimin_test_gradient(s->gradient,1e-3);
    } while(status==GSL_CONTINUE&&iter<10000);

    // Print final information
    message(piter,iter,s,t0);
    puts(status==GSL_SUCCESS?"Minimum found":"Minimization failed");

    // Copy the vector contents into the coefficient array within the class
    memcpy(al,s->x->data,dof*sizeof(double));

    // Free the dynamically allocated memory
    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x);
    return t0-tstart;
}

/** Prints the coefficients of the map. */
void bi_image::print_map() {
    for(int i=0;i<2;i++) {
        printf("%s map coefficients:\n",i==0?"x":"\ny");
        double *as=al+i;
        for(int l=0;l<q;l++) {
            for(int k=0;k<q-1;k++) printf("% 10.4g ",as[2*(l*q+k)]);
            printf("% 10.4g\n",as[2*(l*q+q-1)]);
        }
    }
}

/** Prints a message about the status of the BFGS minimization.
 * \param[in] iter the iteration count.
 * \param[in] s a pointer to the minimization class. */
void bi_image::message(int &piter,int iter,gsl_multimin_fdfminimizer *s,double &t0) {
    if(piter==iter) {
        printf("Iteration %d, residual %.10f\n",iter,s->f);
        t0=wtime();
    } else {
        double t1=wtime();
        printf("Iteration %d, residual %.10f [%.6g s/iter]\n",iter,s->f,(t1-t0)/(iter-piter));
        t0=t1;
        piter=iter;
    }
}

/** Evaluates the square difference between the field values in this class
 * and the mapped field values in the other class.
 * \param[in] c an array of coefficients describing the mapping to use.
 * \return The value of the difference. */
double bi_image::fun(double *c) {
    switch(ftype) {
        case 0: return t_fun<0>(c);
        case 1: return t_fun<1>(c);
        default: return 0;
    }
}

/** Evaluates the gradient of the difference between the field values in this
 * class and the mapped field values in the other class.
 * \param[in] c an array of coefficients describing the mapping to use.
 * \param[in] gr an array in which to store the gradient. */
void bi_image::dfun(double *c,double *df) {
    switch(ftype) {
        case 0: t_dfun<0>(c,df);break;
        case 1: t_dfun<1>(c,df);
    }
}

/** Evaluates the square difference between the field values in this class
 * and the mapped field values in the other class, plus the gradient of
 * the difference with respect to the coefficients in the mapping.
 * \param[in] c an array of coefficients describing the mapping to use.
 * \param[in] gr an array in which to store the gradient.
 * \return The value of the difference. */
double bi_image::fun_dfun(double *c,double *df) {
    switch(ftype) {
        case 0: return t_fun_dfun<0>(c,df);
        case 1: return t_fun_dfun<1>(c,df);
        default: return 0;
    }
}

/** Evaluates the square difference between the field values in this class
* and the mapped field values in the other class.
* \param[in] c an array of coefficients describing the mapping to use.
* \return The value of the difference. */
template<int ft>
double bi_image::t_fun(double *c) {
    double S=0;
#pragma omp parallel
    {
        bicubic_interp* bip=bic[thread_num()];

#pragma omp for reduction(+:S)
        for(int j=jlo;j<jhi;j++) {
            double u,v;
            for(int i=ilo;i<ihi;i++) {
                pos(c,u,v,i,j);
                S+=gfunc(ft,of[j*m+i],bip->f(u,v));
            }
        }
    }
    output_state(c);
    printf("f = %g\n",4./mn*S);
    return 4./mn*S;
}

/** Evaluates the gradient of the difference between the field values in this
 * class and the mapped field values in the other class.
 * \param[in] c an array of coefficients describing the mapping to use.
 * \param[in] gr an array in which to store the gradient. */
template<int ft>
void bi_image::t_dfun(double *c,double *gr) {
    puts("dfun");
#pragma omp parallel
    {

        // Zero the accumulators used by this thread
        int tn=thread_num();
        double *gt=grl[tn];
        for(double *gp=gt;gp<gt+dof;gp++) *gp=0;
        bicubic_interp *bip=bic[tn];

        // Loop over the rows of the image
#pragma omp for
        for(int j=jlo;j<jhi;j++) {
            double *gp,u,v,z,grad_x,grad_y;
            vec3 d,fx,fy;
            for(int i=ilo;i<ihi;i++) {
                pos(c,u,v,i,j);
                gbfunc(ft,of[j*m+i],bip->f_grad_f(u,v,fx,fy),d);
                grad_x=dot(fx,d);
                grad_y=dot(fy,d);
                gp=gt;
                for(int l=0;l<q;l++) for(int k=0;k<q;k++) {
                    z=Tx[k*m+i]*Ty[l*n+j];
                    *(gp++)+=grad_x*z;
                    *(gp++)+=grad_y*z;
                }
            }
        }
    }

    // Sum and scale the results
    for(int l=0;l<dof;l++) {
        gr[l]=grl[0][l];
        for(int k=1;k<nt;k++) gr[l]+=grl[k][l];
        gr[l]*=4./mn;
    }
    output_state(c);
    double gr2[2];
    dfun_check(c,gr2,1e-5);
    printf("df = [ %g, %g]\n",gr[0],gr[1]);
    printf("df2 = [ %g, %g]\n",gr2[0],gr2[1]);
}

/** Evaluates the square difference between the field values in this class
 * and the mapped field values in the other class, plus the gradient of
 * the difference with respect to the coefficients in the mapping.
 * \param[in] c an array of coefficients describing the mapping to use.
 * \param[in] gr an array in which to store the gradient.
 * \return The value of the difference. */
template<int ft>
double bi_image::t_fun_dfun(double *c,double *gr) {
    puts("fun dfun");
    double S=0;
#pragma omp parallel
    {

        // Zero the accumulators used by this thread
        int tn=thread_num();
        double *gt=grl[tn];
        for(double *gp=gt;gp<gt+dof;gp++) *gp=0;
        bicubic_interp *bip=bic[tn];

        // Loop over the rows of the image
#pragma omp for reduction(+:S)
        for(int j=jlo;j<jhi;j++) {
            double *gp,u,v,z,grad_x,grad_y;
            vec3 d,fx,fy;
            for(int i=ilo;i<ihi;i++) {
                pos(c,u,v,i,j);
                S+=gbfunc(ft,of[j*m+i],bip->f_grad_f(u,v,fx,fy),d);
                grad_x=dot(fx,d);
                grad_y=dot(fy,d);
                gp=gt;
                for(int l=0;l<q;l++) for(int k=0;k<q;k++) {
                    z=Tx[k*m+i]*Ty[l*n+j];
                    *(gp++)+=grad_x*z;
                    *(gp++)+=grad_y*z;
                }
            }
        }
    }

    // Sum and scale the results
    for(int l=0;l<dof;l++) {
        gr[l]=grl[0][l];
        for(int k=1;k<nt;k++) gr[l]+=grl[k][l];
        gr[l]*=4./mn;
    }
    output_state(c);
    printf("f = %g\n",4./mn*S);
    printf("df = [ %g, %g]\n",gr[0],gr[1]);
    return 4./mn*S;
}

/** Returns the function of the two pixel values that is to be minimized.
 * \param[in] ft the function type to use.
 * \param[in] a the pixel value in the first image.
 * \param[in] b the pixel value in the second image.
 * \return The function evaluation. */
inline double bi_image::gfunc(int ft,vec3 a,vec3 b) {
    switch(ft) {
        case 0: return sq_diff(a,b);
        default: {
            fputs("Only zero type supported\n",stderr);
            exit(1);
            return 0;
        }
    }
}

/** Returns the function of the two pixel values that is to be minimized, plus
 * the partial derivative of the function with respect to the second argument.
 * \param[in] ft the function type to use.
 * \param[in] a the pixel value in the first image.
 * \param[in] b the pixel value in the second image.
 * \param[out] gb the partial derivative.
 * \return The function evaluation. */
inline double bi_image::gbfunc(int ft,vec3 a,vec3 b,vec3 &gb) {
    switch(ft) {
        case 0:
            {
                vec3 d=b-a;
                gb=2.*d;
                return mod_sq(d);
            }
        default: {
            fputs("Only zero type supported\n",stderr);
            exit(1);
            return 0;
        }
    }
}

/* Evaluates the gradient of the difference between the field values in this
 * class and the mapped field values in the other class. It uses centered
 * differences of the function itself, and can be used for checking the
 * main derivative function is working.
 * \param[in] c an array of coefficients describing the mapping to use.
 * \param[in] gr an array in which to store the gradient.
 * \param[in] eps the step length to use in the centered difference stencil. */
void bi_image::dfun_check(double *c,double *gr,double eps) {
    double nor=0.5/eps,t;
    for(int k=0;k<dof;k++) {
        c[k]-=eps;
        t=fun(c);
        c[k]+=2*eps;
        gr[k]=(fun(c)-t)*nor;
        c[k]-=eps;
    }
}

/** Calculates the mapped position of a gridpoint.
 * \param[in] c the coefficients of the mapping.
 * \param[out] (u,v) the mapped position.
 * \param[in] (i,j) the gridpoint index. */
void bi_image::pos(double *c,double &u,double &v,int i,int j) {
    u=ax+i*dx;v=ay+j*dy;
    double *cp=c;
    for(int l=0;l<q;l++) {
        double ul=0,vl=0;
        for(int k=0;k<q;k++) {
            ul+=*(cp++)*Tx[k*m+i];
            vl+=*(cp++)*Tx[k*m+i];
        }
        u+=ul*Ty[l*n+j];
        v+=vl*Ty[l*n+j];
    }
}

/** Sets the secondary grid to be a mapped version of the primary grid.
 * \param[in] c the coefficients in the mapping to use. */
void bi_image::compute_map(double *c) {
    if(g==NULL) g=new vec3[mn];

#pragma omp parallel for
    for(int j=0;j<n;j++) {
        int k=thread_num();
        vec3 *gp=g+j*m;
        double u,v;
        for(int i=0;i<m;i++) {
            pos(c,u,v,i,j);
            *(gp++)=bic[k]->f(u,v);
        }
    }
}

/** Outputs a grid of field values in a binary format that can be read by Gnuplot
 * \param[in] filename the filename to save to.
 * \param[in] coords whether to include the coordinate information.
 * \param[in] primary whether to output the primary f grid or the secondary g
 *            grid.
 * \param[in] chan the color channel to output (0: red, 1: green, 2: blue). */
void bi_image::output_gnuplot(const char *filename,bool coords,bool primary,int chan) {
    int i,j,l=coords?m+1:m;

    // Open the output file and check that the operation was successful
    FILE *outf=fopen(filename,"wb");
    if(outf==NULL) {
        fprintf(stderr,"Error opening file \"%s\"",filename);
        exit(1);
    }

    // Set up pointers and allocate temporary memory
    if(!primary&&g==NULL) {
        fputs("Secondary grid not allocated\n",stderr);
        exit(1);
    }
    vec3 *fp=primary?f:g;
    float *buf=new float[l],*bp,*be=buf+l;

    // Output the x coordinate information if requested
    if(coords) {
        *buf=m;
        bp=buf+1;
        for(i=0;i<m;i++) *(bp++)=ax+i*dx;
        fwrite(buf,sizeof(float),l,outf);
    }

    // Output the field values, converting them to single precision
    // floating point numbers
    for(j=0;j<n;j++) {
        bp=buf;
        if(coords) *(bp++)=ay+j*dy;
        switch(chan) {
            case 0: while(bp<be) *(bp++)=static_cast<float>((fp++)->x);break;
            case 1: while(bp<be) *(bp++)=static_cast<float>((fp++)->y);break;
            case 2: while(bp<be) *(bp++)=static_cast<float>((fp++)->z);
        }
        fwrite(buf,sizeof(float),l,outf);
    }

    // Close the file and remove the temporary memory
    fclose(outf);
    delete [] buf;
}

/** Outputs a color bitmap of the field information in PNG format, with one
 * pixel corresponding to each field value.
 * \param[in] filename the name of the file to write to. */
void bi_image::write_image(const char* filename,bool primary) {

    // Create PNG bitmap array
    png_bytep *rowp=new png_bytep[n];

    // Set up pointers and allocate temporary memory
    if(!primary&&g==NULL) {
        fputs("Secondary grid not allocated\n",stderr);
        exit(1);
    }
    vec3 *fp=primary?f:g;

    // Assemble bitmap information
    int i,j;
    for(j=0;j<n;j++) {
        png_byte* rp=(rowp[n-1-j]=new png_byte[3*m]);
        for(i=0;i<m;i++,fp++) {
            *(rp++)=png_scale(fp->x);
            *(rp++)=png_scale(fp->y);
            *(rp++)=png_scale(fp->z);
        }
    }

    // Output the file
    png_write(m,n,rowp,filename);

    // Free dynamically allocated memory
    for(j=0;j<n;j++) delete [] rowp[j];
    delete [] rowp;
}


/** Project transmitted image to reflected light image using matrix c
 * \param[in] a the pixel values of the image.
 * \param[in] c the mapping coefficients. */
void bi_image::project(double &a,double *c){
//    a=c*a;

}

// Explicit instantiation
template double bi_image::t_fun<0>(double*);
template double bi_image::t_fun<1>(double*);
template void bi_image::t_dfun<0>(double*,double*);
template void bi_image::t_dfun<1>(double*,double*);
template double bi_image::t_fun_dfun<0>(double*,double*);
template double bi_image::t_fun_dfun<1>(double*,double*);
