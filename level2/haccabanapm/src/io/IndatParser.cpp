// src/io/IndatParser.cpp
// HACC indat.params parser.  See IndatParser.hpp for the format spec.

#include "IndatParser.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace pmk {

namespace {

void rtrim(std::string& s)
{
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
        s.pop_back();
}
void ltrim(std::string& s)
{
    std::size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
        ++i;
    s.erase(0, i);
}

// Strip end-of-line `# ...` comment, leave value otherwise untouched.
void strip_comment(std::string& s)
{
    auto h = s.find('#');
    if (h != std::string::npos) s.erase(h);
}

bool parse_int(const std::string& v, int& out)
{
    try {
        std::size_t pos = 0;
        long x = std::stol(v, &pos);
        // Allow trailing whitespace only.
        for (std::size_t i = pos; i < v.size(); ++i)
            if (!std::isspace(static_cast<unsigned char>(v[i]))) return false;
        out = static_cast<int>(x);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_double(const std::string& v, double& out)
{
    try {
        std::size_t pos = 0;
        double x = std::stod(v, &pos);
        for (std::size_t i = pos; i < v.size(); ++i)
            if (!std::isspace(static_cast<unsigned char>(v[i]))) return false;
        out = x;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_bool(const std::string& v, bool& out)
{
    std::string u = v;
    std::transform(u.begin(), u.end(), u.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    if (u == "YES" || u == "TRUE" || u == "ON"  || u == "1") { out = true;  return true; }
    if (u == "NO"  || u == "FALSE"|| u == "OFF" || u == "0") { out = false; return true; }
    return false;
}

} // namespace

IndatParser::IndatParser(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("IndatParser: cannot open " + path);

    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        strip_comment(line);
        ltrim(line); rtrim(line);
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string key;
        if (!(iss >> key))
            continue;  // whitespace-only after comment strip

        // Anything after the key (and any whitespace) is the value.
        std::string value;
        std::getline(iss, value);
        ltrim(value); rtrim(value);
        // value may legitimately be empty (e.g., a flag like SMALL_DUMP).

        entries_[key] = value;
    }
}

bool IndatParser::has(const std::string& key) const
{
    return entries_.find(key) != entries_.end();
}

std::string IndatParser::get_string(const std::string& key) const
{
    auto it = entries_.find(key);
    if (it == entries_.end())
        throw std::runtime_error("IndatParser: missing key " + key);
    accessed_.insert(key);
    return it->second;
}

int IndatParser::get_int(const std::string& key) const
{
    auto it = entries_.find(key);
    if (it == entries_.end())
        throw std::runtime_error("IndatParser: missing key " + key);
    accessed_.insert(key);
    int x = 0;
    if (!parse_int(it->second, x))
        throw std::runtime_error("IndatParser: cannot parse '" + it->second +
                                 "' as int (key " + key + ")");
    return x;
}

double IndatParser::get_double(const std::string& key) const
{
    auto it = entries_.find(key);
    if (it == entries_.end())
        throw std::runtime_error("IndatParser: missing key " + key);
    accessed_.insert(key);
    double x = 0.0;
    if (!parse_double(it->second, x))
        throw std::runtime_error("IndatParser: cannot parse '" + it->second +
                                 "' as double (key " + key + ")");
    return x;
}

bool IndatParser::get_bool(const std::string& key) const
{
    auto it = entries_.find(key);
    if (it == entries_.end())
        throw std::runtime_error("IndatParser: missing key " + key);
    accessed_.insert(key);
    bool b = false;
    if (!parse_bool(it->second, b))
        throw std::runtime_error("IndatParser: cannot parse '" + it->second +
                                 "' as bool (key " + key + ")");
    return b;
}

int IndatParser::get_int(const std::string& key, int default_value) const
{
    auto it = entries_.find(key);
    if (it == entries_.end()) return default_value;
    accessed_.insert(key);
    int x = 0;
    if (!parse_int(it->second, x))
        throw std::runtime_error("IndatParser: cannot parse '" + it->second +
                                 "' as int (key " + key + ")");
    return x;
}

double IndatParser::get_double(const std::string& key, double default_value) const
{
    auto it = entries_.find(key);
    if (it == entries_.end()) return default_value;
    accessed_.insert(key);
    double x = 0.0;
    if (!parse_double(it->second, x))
        throw std::runtime_error("IndatParser: cannot parse '" + it->second +
                                 "' as double (key " + key + ")");
    return x;
}

void IndatParser::report_unused(std::ostream& out) const
{
    bool any = false;
    for (auto& [k, v] : entries_) {
        if (accessed_.find(k) == accessed_.end()) {
            if (!any) {
                out << "IndatParser: unused keys (present in file, never read):\n";
                any = true;
            }
            out << "  " << k << " = " << v << "\n";
        }
    }
}

} // namespace pmk
