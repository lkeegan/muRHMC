#define CATCH_CONFIG_MAIN
#include "catch.hpp"

// NOTE: modified the following two functions in catch header file
// to get doubles printed in scientific notation with many digits:

/*
std::string StringMaker<double>::convert(double value) {
    return fpToString(value, 17);
}
*/

/*
template<typename T>
std::string fpToString( T value, int precision ) {
    if (std::isnan(value)) {
        return "nan";
    }

    ReusableStringStream rss;
    rss << std::setprecision( precision )
        << std::scientific
        << value;
    return rss.str();
}
*/