#include "bi_interp.hh"

#include <cstdio>
#include <cmath>

/** The bicubic interpolation class constructor sets up the grid constants and
 * a reference to the field to interpolate. \param[in] u a reference to the
 * field to interpolate
 * \param[in] (m,n) the dimensions of the grid.
 * \param[in] (ax,bx) the horizontal range of the grid.
 * \param[in] (ay,by) the vertical range of the grid. */
bicubic_interp::bicubic_interp(vec3 *&u_,int m_,int n_,double ax_,double bx_,double ay_,double by_)
	: m(m_), n(n_), ax(ax_), ay(ay_), xsp((m-1)/(bx_-ax)),
	ysp((n-1)/(by_-ay)), u(u_), ijc(-1) {

	// Clear the intermediate table of coefficients that is used in the
	// calculations
	*a=a[1]=a[2]=a[3]=vec3(0);
	a[4]=a[5]=a[6]=a[7]=vec3(0);
	a[8]=a[9]=a[10]=a[11]=vec3(0);
	a[12]=a[13]=a[14]=a[15]=vec3(0);
}

/** Maps a physical point onto the unit grid, and computes the grid cell that
 * it is within.
 * \param[in,out] (x,y) the coordinates for the point to be mapped.
 * \param[out] (i,j) the grid cell coordinates. */
void bicubic_interp::grid_index(double &x,double &y,int &i,int &j) {
	x=(x-ax)*xsp;i=x_index(x);
	y=(y-ay)*ysp;j=y_index(y);
}

/** Calculates which grid square a given position is within, and sets up the
 * coefficient table if needed.
 * \param[in,out] (x,y) the position to consider. This is remapped to a
 *			fractional position within the grid square upon exit.
 */
void bicubic_interp::grid_setup(double &x,double &y) {
	int i,j,ij;

	// Find which grid square the given position is in
	grid_index(x,y,i,j);

	// Compute the index of the grid square. If the table of coefficients
	// is already set up for this grid square (as indicated by the value of
	// ijc), then skip the table setup routine.
	ij=i+m*j;
	if(ijc!=ij) table_setup(i,j,ij);

	// Map the position to a coordinates in [0,1]^2 within the grid square
	x-=i;y-=j;
}

/** Sets up the table of coefficients of the bicubic interpolation function. */
void bicubic_interp::table_setup(int i,int j,int ij) {
	ijc=ij;
	vec3 *up=u+ij;
	vec3 c00,c01,c02,c03;
	vec3 c10,c11,c12,c13;
	vec3 c20,c21,c22,c23;
	vec3 c30,c31,c32,c33;

	// Bicubic interpolation requires considering a 4x4 grid of field
	// values. Compute the interpolation coefficients for the central two
	// lines of this grid.
	compute_x(i,up,c01,c11,c21,c31);
	compute_x(i,up+m,c02,c12,c22,c32);

	// Set up the table coefficients. Different routines must be used if
	// the grid square is on the top or bottom of the grid.
	if(j==0) {

		// Setup routines for the bottom row
		compute_x(i,up+2*m,c03,c13,c23,c33);
		fill_ad(a,c01,c02,c03);
		fill_ad(a+4,c11,c12,c13);
		fill_ad(a+8,c21,c22,c23);
		fill_ad(a+12,c31,c32,c33);
	} else if(j==n-2) {

		// Setup routines for the top row
		compute_x(i,up-m,c00,c10,c20,c30);
		fill_au(a,c00,c01,c02);
		fill_au(a+4,c10,c11,c12);
		fill_au(a+8,c20,c21,c22);
		fill_au(a+12,c30,c31,c32);
	} else {

		// Setup routines for a middle row
		compute_x(i,up-m,c00,c10,c20,c30);
		compute_x(i,up+2*m,c03,c13,c23,c33);
		fill_a(a,c00,c01,c02,c03);
		fill_a(a+4,c10,c11,c12,c13);
		fill_a(a+8,c20,c21,c22,c23);
		fill_a(a+12,c30,c31,c32,c33);
	}
}

/** Computes the intermediate values (from a row of four gridpoints) that are
 * needed to calculate the table coefficients.
 * \param[in] i the horizontal grid index.
 * \param[in] up a pointer the grid square to consider.
 * \param[out] (c0,c1,c2,c3) the computed values. */
void bicubic_interp::compute_x(int i,vec3 *up,vec3 &c0,vec3 &c1,vec3 &c2,vec3 &c3) {
	c0=*up;
	if(i==0) {

		// Setup for the leftmost column
		c1=-1.5*(*up)+2.*up[1]-0.5*up[2];
		c2=0.5*(*up)-up[1]+0.5*up[2];
		c3=0;
	} else if(i==m-2) {

		// Setup for the rightmost column
		c1=-0.5*up[-1]+0.5*up[1];
		c2=0.5*up[-1]-*up+0.5*up[1];
		c3=0;
	} else {

		// Setup for a middle column
		c1=-0.5*up[-1]+0.5*up[1];
		c2=up[-1]-2.5*(*up)+2.*up[1]-0.5*up[2];
		c3=-0.5*up[-1]+1.5*(*up)-1.5*up[1]+0.5*up[2];
	}
}

/** Calculates the bicubic interpolation of the field at a given position.
 * \param[in] (x,y) the position to consider.
 * \return The bicubic interpolation. */
vec3 bicubic_interp::f(double x,double y) {
	grid_setup(x,y);
	return fmap(x,y);
}

/** Calculates the bicubic interpolation of the field and its gradient at a
 * given position.
 * \param[in] (x,y) the position to consider.
 * \param[out] (fx,fy) the gradient of the interpolation.
 * \return The bicubic interpolation. */
vec3 bicubic_interp::f_grad_f(double x,double y,vec3 &fx,vec3 &fy) {
	grid_setup(x,y);
	fx=xsp*(yl(a+4,y)+x*(2.*yl(a+8,y)+3.*x*yl(a+12,y)));
	fy=ysp*(dyl(a,y)+x*(dyl(a+4,y)+x*(dyl(a+8,y)+x*dyl(a+12,y))));
	return fmap(x,y);
}

/** Sets up four entries of the table coefficients for the case when the grid
 * square is on the bottom row.
 * \param[in] ap a pointer in the table in which to set the coefficients.
 * \param[in] (c1,c2,c3) the intermediate values from which to compute the
 *			 table coefficients. */
void bicubic_interp::fill_ad(vec3 *ap,vec3 c1,vec3 c2,vec3 c3) {
	*ap=c1;
	ap[1]=-1.5*c1+2.*c2-0.5*c3;
	ap[2]=0.5*c1-c2+0.5*c3;
	ap[3]=0;
}

/** Sets up four entries of the table coefficients for the case when the grid
 * square is on the top row.
 * \param[in] ap a pointer in the table in which to set the coefficients.
 * \param[in] (c0,c1,c2) the intermediate values from which to compute the
 *			 table coefficients. */
void bicubic_interp::fill_au(vec3 *ap,vec3 c0,vec3 c1,vec3 c2) {
	*ap=c1;
	ap[1]=-0.5*c0+0.5*c2;
	ap[2]=0.5*c0-c1+0.5*c2;
	ap[3]=0;
}

/** Sets up four entries of the table coefficients for the case when the grid
 * square is in a middle row.
 * \param[in] ap a pointer in the table in which to set the coefficients.
 * \param[in] (c0,c1,c2,c3) the intermediate values from which to compute the
 *			    table coefficients. */
void bicubic_interp::fill_a(vec3 *ap,vec3 c0,vec3 c1,vec3 c2,vec3 c3) {
	*ap=c1;
	ap[1]=-0.5*c0+0.5*c2;
	ap[2]=c0-2.5*c1+2.*c2-0.5*c3;
	ap[3]=-0.5*c0+1.5*c1-1.5*c2+0.5*c3;
}

/** Calculates the exact line integral of the bicubic interpolation using
 * four-point Gaussian quadrature applied to each patch that the line crosses.
 * \param[in] (x,y) the start position of the integral.
 * \param[in] (xe,ye) the end position of the integral.
 * \return The line integral result. */
vec3 bicubic_interp::line_integral(double x,double y,double xe,double ye) {

	// Prepare the integration prefactor
	double lx=xe-x,ly=ye-y,ifac=sqrt(lx*lx+ly*ly),xr;

	// Scale the coordinates onto a unit grid, and compute the grid squares
	// of the start and end poistions. Determine whether to use horizontal
	// or vertical displacements to count the relative contributions of
	// each part of the integral.
	int i,j,ie,je,ir;
	grid_index(x,y,i,j);
	grid_index(xe,ye,ie,je);
	ifac/=((vert=fabs(ye-y)>fabs(xe-x))?ye-y:xe-x);

	// If needed, switch the two ends of the integral so that it goes in
	// the positive x-direction
	if(ie<i) {
		double z=x;x=xe;xe=z;z=y;y=ye;ye=z;
		int k=i;i=ie;ie=k;k=j;j=je;je=k;
		ifac=-ifac;
	}

	// Loop over rows in the grid that the line passes
	vec3 ans(0);
	if(j<je) {
		do {
			xr=(xe*((j+1)-y)+x*(ye-(j+1)))/(ye-y);
			ir=x_index(xr);
			ans+=row_integral(i,ir,j,x,y,xr,j+1);
			x=xr;y=++j;i=ir;
		} while(j<je);
		ans+=row_integral(i,ie,j,x,y,xe,ye);
	} else {
		while(j>je) {
			xr=(xe*(j-y)+x*(ye-j))/(ye-y);
			ir=x_index(xr);
			ans+=row_integral(i,ir,j,x,y,xr,j);
			x=xr;y=j--;i=ir;
		}
		ans+=row_integral(i,ie,j,x,y,xe,ye);
	}

	// Apply normalizing factor to take into account line gradient and grid
	// scaling
	return ans*ifac;
}

/** Computes the integral across a line that is wholly within a row in the grid.
 * \param[in] (i,ie) the range of horizontal boxes that the line crosses.
 * \param[in] j the vertical index of the row.
 * \param[in] (x,y) the start position of the line.
 * \param[in] (xe,ye) the end position of the line.
 * \param[in,out] (xans,yans) the accumulators for the integrals. */
vec3 bicubic_interp::row_integral(int i,int ie,int j,double x,double y,double xe,double ye) {
	vec3 ans(0);
	while(i<ie) {
		double yr=(ye*((i+1)-x)+y*(xe-(i+1)))/(xe-x);
		ans+=patch_integral(i,j,x,y,i+1,yr);
		x=++i;y=yr;
	}
	return ans+patch_integral(i,j,x,y,xe,ye);
}

/** Computes the integral across a line within a single grid patch.
 * \param[in] (i,j) the index of the patch.
 * \param[in] (x,y) the start position of the line.
 * \param[in] (xe,ye) the end position of the line.
 * \param[in,out] (xans,yans) the accumulators for the integrals. */
vec3 bicubic_interp::patch_integral(int i,int j,double x,double y,double xe,double ye) {

	// Four point Gaussian quadrature points and weights, capable of
	// exactly integrating polynomials up to order 7
	const double g0=0.5-0.5*sqrt(3./7.+2./7.*sqrt(6./5.)),
	             g1=0.5-0.5*sqrt(3./7.-2./7.*sqrt(6./5.)),
		     g2=0.5+0.5*sqrt(3./7.-2./7.*sqrt(6./5.)),
		     g3=0.5+0.5*sqrt(3./7.+2./7.*sqrt(6./5.)),
		     w0=(18.-sqrt(30.))/72.,w1=(18.+sqrt(30.))/72.;

	// Remap coordinates and set up the interpolation table if needed
	double dx=xe-x,dy=ye-y;
	x-=i;y-=j;
	int ij=i+m*j;
	if(ijc!=ij) table_setup(i,j,ij);

	// Sample the function at the quadrature points
	double x0=x+g0*dx,y0=y+g0*dy,x1=x+g1*dx,y1=y+g1*dy,
	       x2=x+g2*dx,y2=y+g2*dy,x3=x+g3*dx,y3=y+g3*dy;
    vec3 f0=fmap(x0,y0),f1=fmap(x1,y1),
	     f2=fmap(x2,y2),f3=fmap(x3,y3);

	// Compute integral contributions
	return (vert?dy:dx)*(w0*(f0+f3)+w1*(f1+f2));
}
