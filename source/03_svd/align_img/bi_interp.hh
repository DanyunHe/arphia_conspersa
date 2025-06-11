#ifndef BI_INTERP_HH
#define BI_INTERP_HH

#include "vec3.hh"

class bicubic_interp {
	public:
		/** The number of gridpoints in the horizontal direction. */
		int m;
		/** The number of gridpoints in the vertical direction. */
		int n;
		/** The lower x coordinate of the grid. */
		double ax;
		/** The lower y coordinate of the grid. */
		double ay;
		/** The reciprocal of the horizontal grid spacing. */
		double xsp;
		/** The reciprocal of the vertical grid spacing. */
		double ysp;
		/** The table of coefficients from which to compute the bicubic
		 * interpolation. */
		vec3 a[16];
		/** A reference to the field to consider. */
		vec3 *&u;
		/** The index of the grid square that the coefficient table is
		 * currently set up for. */
		int ijc;
		bicubic_interp(vec3 *&u_,int m_,int n_,double ax_,double bx_,double ay_,double by_);
		~bicubic_interp() {}
		vec3 f(double x,double y);
		vec3 f_grad_f(double x,double y,vec3 &fx,vec3 &fy);
		vec3 line_integral(double x0,double y0,double x1,double y1);
	private:
		/** Whether to evaluate the integral line segments using
		 * vertical (true) or horizontal (false) distances. */
		bool vert;
		void table_setup(int i,int j,int ij);
		void compute_x(int i,vec3 *up,vec3 &c0,vec3 &c1,vec3 &c2,vec3 &c3);
		/** Calculates the horizontal index of the grid cell that an x
		 * coordinate is within, taking into account boundary cases.
		 * \param[in] x the x coordinate.
		 * \return The index. */
		inline int x_index(double x) {
			int i=static_cast<int>(x);
			return i<0?0:(i>m-2?m-2:i);
		}
		/** Calculates the horizontal index of the grid cell that an x
		 * coordinate is within, taking into account boundary cases.
		 * \param[in] y the y coordinate.
		 * \return The index. */
		inline int y_index(double y) {
			int j=static_cast<int>(y);
			return j<0?0:(j>n-2?n-2:j);
		}
		/** Computes the bicubic interpolation of the function at a point,
		 * assuming the interpolation coefficient table is already set up.
		 * \param[in] (x,y) the position of the point, mapped into the
		 *		    unit square. */
		inline vec3 fmap(double x,double y) {
			return yl(a,y)+x*(yl(a+4,y)+x*(yl(a+8,y)+x*yl(a+12,y)));
		}
		/** Calculates the function value. */
		inline vec3 yl(vec3 *ap,double y) {
			return *ap+y*(ap[1]+y*(ap[2]+y*ap[3]));
		}
		/** Calculates the first derivative of the function. */
		inline vec3 dyl(vec3 *ap,double y) {
			return ap[1]+y*(2.*ap[2]+3.*y*ap[3]);
		}
		void grid_index(double &x,double &y,int &i,int &j);
		void grid_setup(double &x,double &y);
		vec3 row_integral(int i,int ie,int j,double x,double y,double xe,double ye);
		vec3 patch_integral(int i,int j,double x,double y,double xe,double ye);
		void fill_ad(vec3 *ap,vec3 c1,vec3 c2,vec3 c3);
		void fill_au(vec3 *ap,vec3 c0,vec3 c1,vec3 c2);
		void fill_a(vec3 *ap,vec3 c0,vec3 c1,vec3 c2,vec3 c3);
};

#endif
