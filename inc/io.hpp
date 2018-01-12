#ifndef LATTICE_IO_H
#define LATTICE_IO_H
#include <fstream>
#include <string>
#include <complex>
#include "4d.hpp"
#include "su3.hpp"
#include "hmc.hpp"

// output message and variables to log file
void log(const std::string& message);
void log(const std::string& message, const std::string& value);
void log(const std::string& message, double value);
void log(const std::string& message, std::complex<double> value);

// read hmc and run parameters from input file
void read_input_file(const std::string& filename, hmc_params& hmc_params, run_params& run_params);

// Read a gauge field from file in Philippe's fortran binary format
void read_fortran_gauge_field(field<gauge>& U, const std::string& filename);

// Read a gauge field from file in Massimo's text format
void read_massimo_gauge_field(field<gauge>& U, const std::string& filename);

// returns a filename for config with given number
std::string make_fileName (const std::string& base_name, int config_number);

// read gauge field from file
void read_gauge_field(field<gauge>& U, const std::string& base_name, int config_number);

// write gauge field to file
void write_gauge_field(field<gauge>& U, const std::string& base_name, int config_number);

// calculate average plaqutte for use as checksum
double checksum_plaquette (const field<gauge> &U);

#endif //LATTICE_IO_H