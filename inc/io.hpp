#ifndef LATTICE_IO_H
#define LATTICE_IO_H
#include <fstream>
#include <string>
#include "4d.hpp"
#include "su3.hpp"

// Read a gauge field from file in Philippe's fortran format
void read_fortran_gauge_field(field<gauge>& U, const std::string& fileName);

// read gauge field from file
void read_gauge_field(field<gauge>& U, const std::string& fileName);

// write gauge field to file
void write_gauge_field(field<gauge>& U, const std::string& fileName);

// calculate average plaqutte for use as checksum
double checksum_plaquette (const field<gauge> &U);

#endif //LATTICE_IO_H