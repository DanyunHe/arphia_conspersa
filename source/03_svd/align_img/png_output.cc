#include "png_output.hh"

#include <cstdlib>

/** Prints an error message due to outputting a PNG image
 * \param[in] err_msg the error message to print. */
void png_abort(const char* err_msg) {
    fprintf(stderr,"PNG output: %s\n",err_msg);
    exit(1);
}

/** Writes a PNG image using the libpng library.
 * \param[in] (m_,n_) the dimensions of the image.
 * \param[in] rowp a pointer to the PNG bitmap information.
 * \param[in] filename the output file name. */
void png_write(int m_,int n_,png_bytep *rowp,const char *filename) {
    FILE *fp=fopen(filename,"wb");
    if(fp==NULL) png_abort("can't open output file");

    png_structp p=png_create_write_struct(PNG_LIBPNG_VER_STRING,NULL,NULL,NULL);
    if(!p) png_abort("error creating write structure");

    png_infop info=png_create_info_struct(p);
    if(!p) png_abort("error creating info structure");

    if(setjmp(png_jmpbuf(p))) png_abort("setjmp error");
    png_init_io(p,fp);
    png_set_IHDR(p,info,m_,n_,8,PNG_COLOR_TYPE_RGB,PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,PNG_FILTER_TYPE_DEFAULT);

    png_write_info(p,info);
    png_write_image(p,rowp);
    png_write_end(p,NULL);
    fclose(fp);
    png_destroy_write_struct(&p,&info);
}
