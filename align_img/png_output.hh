#ifndef PNG_OUTPUT_HH
#define PNG_OUTPUT_HH

#include "png.h"

void png_abort(const char* err_msg);

void png_write(int m_,int n_,png_bytep *rowp,const char *filename);

#endif
