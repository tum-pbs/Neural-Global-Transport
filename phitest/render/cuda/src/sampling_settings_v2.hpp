#pragma once

#ifndef _INCLUDE_SAMPLING_SETTINGS_2
#define _INCLUDE_SAMPLING_SETTINGS_2

//#include<limits>

namespace Sampling{

enum BoundaryMode{BOUNDARY_BORDER=0, BOUNDARY_CLAMP=1, BOUNDARY_WRAP=2}; //, BOUNDARY_MIRROR=3
enum FilterMode{FILTERMODE_NEAREST=0, FILTERMODE_LINEAR=1, FILTERMODE_MIN=2, FILTERMODE_MAX=3}; //, BOUNDARY_MIRROR=3

} //Sampling


#endif //_INCLUDE_SAMPLING_SETTINGS_2