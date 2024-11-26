#include <cstdio>
#include <cstring>

#include "bi_image.hh"

int main() {
	int exp=43;
	int t=3;
	char buf[128];

	// q-1 is the maximum degree of polynomial to consider. dof sets the
	// total number of degrees of freedom in the minimization.
	const int q=1,dof=2*q*q;

	// Read in the two images. Set the optimization extent to be 40 pixels
	// smaller than the full image, to prevent the method applying too much
	// weight to the boundaries.
	
	bi_image b("../imgs/population_34/population_34+FMNH_4669526+stack_0_hw_crop.tif",q);
	bi_image a("../imgs/population_34/population_34+FMNH_4669526+stack_1_hw_crop.tif",q);

	puts("1");
	// Choose function minimization type. 0: (a-b)^2, 1: a*(a-b).
	b.ftype=0;

	// Pin Chebyshev polynomials to match the boundary
	b.chebyshev_pin();
	puts("2");
	// Truncate fitting region
	b.ilo+=40;b.ihi-=40;

	b.jlo+=40;b.jhi-=40;

	// Create smoothed versions for fitting
	bi_image a2(a),b2(b);
	for(int i=0;i<150;i++) a2.smooth(0.05); //a2.smooth(0.125);
	for(int i=0;i<150;i++) b2.smooth(0.05); //b2.smooth(0.125);

    puts("3");
	// Project the image to the same lighting condition
	// b2.project(b2,c);

	// Initialize mapping coefficients
	double al[dof],tot_time=0;
	for(int k=0;k<dof;k++) al[k]=0;

	// Do repeated fittings at coarser levels
	int d[3]={4,2};
	for(int k=0;k<2;k++) {
		printf("Fit with downsample factor %d\n",d[k]);
		bi_image a3(a2,d[k]),b3(b2,d[k]);

		tot_time+=b3.fit(a3,al);
		b3.print_map();
		memcpy(al,b3.al,dof*sizeof(double));
		puts("");
	}

	// Do final fitting at full resolution
	puts("Final fit iteration");
	tot_time+=b2.fit(a2,al);
	b2.print_map();
	printf("\nTotal time: %g s\n",tot_time);

	// Apply fit to original image
	b.compute_map(b2.al);
	//b.output_gnuplot("orig_bmap.gnu",true,false,0);
    a.write_image("target.png",true);
    b.write_image("orig.png",true);
    b.write_image("mapped.png",false);

	// Apply fit to thresholded image
	//c.compute_map(b2.al);
	//c.output_gnuplot("test_bmap.gnu",true,false);
}
