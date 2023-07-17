#ifndef BI_IMAGE_HH
#define BI_IMAGE_HH

#include <cstdlib>

#include "vec3.hh"
#include "bi_interp.hh"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_multimin.h"

double bi_f(const gsl_vector *v,void *params);
void bi_df(const gsl_vector *v,void *params,gsl_vector *df);
void bi_fdf(const gsl_vector *x,void *params,double *f,gsl_vector *df);

class bi_image {
	public:
		/** The maximum degree of polynomial to consider. */
		const int q;
		/** The total number of degrees of freedom in the fitting
		 * function. */
		const int dof;
		/** The minimization function type. */
		int ftype;
		/** The number of gridpoints in the horizontal direction. */
		int m;
		/** The number of gridpoints in the vertical direction. */
		int n;
		/** The total number of gridpoints. */
		int mn;
		/** The number of threads. */
		int nt;
		/** Whether the Chebyshev polynomials are pinned at the end
		 * points. */
		bool pin;
		/** The horizontal grid spacing. */
		double dx;
		/** The vertical grid spacing. */
		double dy;
		/** The lower x coordinate of the grid. */
		double ax;
		/** The lower y coordinate of the grid. */
		double ay;
		/** The array of field values. */
		vec3 *f;
		/** A second array of field values, used for field smoothing. */
		vec3 *g;
		/** A pointer to the function values in the other class. */
		vec3 *of;
		/** A array holding the coefficients in the best fit mapping. */
		double *al;
		/** The lower x-index of the box on which to consider fitting. */
		int ilo;
		/** The upper x-index of the box on which to consider fitting. */
		int ihi;
		/** The lower y-index of the box on which to consider fitting. */
		int jlo;
		/** The upper y-index of the box on which to consider fitting. */
		int jhi;
		bi_image(const char* mfile,int q_);
		bi_image(int m_,int n_,int q_);
		bi_image(bi_image &b);
		bi_image(bi_image &b,int k);
		~bi_image();
		void init_test();
		void smooth(double s);
		double fit(bi_image &b,double *al_=NULL);
		double fun(double *c);
		void dfun(double *c,double *df);
		double fun_dfun(double *c,double *df);
		void dfun_check(double *c,double *gr,double eps);
		inline void compute_map() {
			compute_map(al);
		}
		inline void chebyshev_pin() {
			pin=true;
			chebyshev_pin(Tx,m,ax,dx);
			chebyshev_pin(Ty,n,ay,dy);
		}
		void compute_map(double *c);
		void print_map();
		void line_smooth(bi_image &b,double r,double theta);
		void output_gnuplot(const char *filename,bool coords,bool primary,int chan);

        void write_image(const char *filename,bool primary);
		void project(double &proj,double *c);
	private:
		template<int ft>
		double t_fun(double *c);
		template<int ft>
		void t_dfun(double *c,double *df);
		template<int ft>
		double t_fun_dfun(double *c,double *df);
		void setup_common(bool grid_constants=true);
		void chebyshev_setup(double *&T,int r,double a,double d);
		void chebyshev_pin(double *&T,int r,double a,double d);
		void message(int &piter,int iter,gsl_multimin_fdfminimizer *s,double &t0);
		void pos(double *c,double &u,double &v,int i,int j);
		inline double gfunc(int ft,vec3 a,vec3 b);
		inline double gbfunc(int ft,vec3 a,vec3 b,vec3 &gb);
        void output_state(double *c) {
            printf("[%g",*c);
            for(int i=1;i<dof;i++) printf(",%g",c[i]);
            puts("]");
        }
		inline double rshift(double s) {
			return -s+static_cast<double>(rand())*(2*s/RAND_MAX);
		}
		inline int thread_num();
		inline double wtime();
		/** Arrays for accumulating the gradients of the optimization
		 * function, containing separate space for each thread. */
		double **grl;
		/** An array of bicubic interpolation classes. */
		bicubic_interp** bic;
		/** A table of Chebyshev polynomial values on the x grid
		 * coordinates. */
		double *Tx;
		/** A table of Chebyshev polynomial values on the y grid
		 * coordinates. */
		double *Ty;
};

#endif
