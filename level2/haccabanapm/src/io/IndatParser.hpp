#pragma once
// src/io/IndatParser.hpp
// Parser for HACC's `indat.params` configuration files.
//
// Format:
//   - One key-value pair per line: `KEY VALUE`
//   - Whitespace-separated (any amount).
//   - `#` starts a comment (line or end-of-line).
//   - Blank lines are ignored.
//   - Keys are case-sensitive.  Values are stored as strings, converted
//     at access time to int/double/bool.
//
// HACC reference:
//   nbody/initializer/Initializer.cxx and HaccConfig parsing in
//   nbody/common/HACCParameter.cxx — same file format, simpler parser here.
//
// Spec: ../analysis/prompts/12_implement_drivers_and_demo.md (Deliverable 1)

#include <iosfwd>
#include <map>
#include <set>
#include <string>

namespace pmk {

class IndatParser {
public:
    // Read and parse the file.  Throws std::runtime_error on unreadable
    // file or malformed line.
    explicit IndatParser(const std::string& path);

    // Typed accessors.  Throw std::runtime_error if the key is missing
    // or the value cannot be parsed as the requested type.
    int         get_int   (const std::string& key) const;
    double      get_double(const std::string& key) const;
    bool        get_bool  (const std::string& key) const;
    std::string get_string(const std::string& key) const;

    // Typed accessors with defaults — return the default if the key is
    // missing, but still throw on parse failure.
    int    get_int   (const std::string& key, int default_value) const;
    double get_double(const std::string& key, double default_value) const;

    // Was the key in the file?
    bool has(const std::string& key) const;

    // Report keys present in the file but never accessed.  Useful for
    // catching typos in user config.
    void report_unused(std::ostream& out) const;

private:
    std::map<std::string, std::string> entries_;
    mutable std::set<std::string>      accessed_;
};

} // namespace pmk
