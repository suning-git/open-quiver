#!/bin/bash
# Patches for qme-ng: 6 documented fixes + compilation compatibility fixes
set -e

# === Critical fix: Makefile -D__cplusplus breaks C++ standard detection ===
# Remove -D__cplusplus (overrides __cplusplus to 1, breaking stdlib and Boost)
# Add -std=gnu++17 for proper C++17 support
sed -i 's/-D__STDC_LIMIT_MACROS -D__cplusplus/-D__STDC_LIMIT_MACROS -std=gnu++17/' Makefile

# === Documented fix 1: ATILDE out-of-bounds write ===
sed -i 's/for(i=0;i<nbSommets;i++)/for(i=0;i<nbSommets-1;i++)/' src/carquois.cpp

# === Documented fix 2: uninitialized pointers in main ===
sed -i 's/Carquois \*carquois;/Carquois *carquois = NULL;/' qme-ng.cpp
sed -i 's/GreenFinder \*gf;/GreenFinder *gf = NULL;/' qme-ng.cpp
sed -i 's/MutExploratorSeq \*explorator;/MutExploratorSeq *explorator = NULL;/' qme-ng.cpp
sed -i 's/PrincipalExtension \*pt;/PrincipalExtension *pt = NULL;/' qme-ng.cpp
sed -i 's/^    delete pt;/    if(pt) delete pt;/' qme-ng.cpp

# === Documented fix 3: missing return in estDansC() ===
sed -i '/bool MutExplorator::estDansC/,/^}/ {
    /^}/ i\    return false;
}' src/mutexplorator.cpp

# === Documented fix 4: uninitialized ret in greenfinder ===
sed -i 's/int ret;/int ret = -1;/' src/greenfinder.cpp

# === Documented fix 5: Windows sys/times.h (add _WIN32 branch) ===
sed -i '/#ifndef NAUTY_CPU_DEFINED/a\
#ifdef _WIN32\
#include <time.h>\
#define CPUDEFS\
#define CPUTIME ((double)clock() \/ CLOCKS_PER_SEC)\
#else' include/naututil.h
sed -i '/#endif \/\*NAUTY_CPU_DEFINED\*\//i\
#endif \/* _WIN32 *\/' include/naututil.h

# === Documented fix 6: file parsing CRLF and line concatenation ===
sed -i '/while(!f.eof())/,/^    }/ {
    s/while(!f.eof())/while(true)/
}' src/principalExtension.cpp
sed -i '/while(true)/{n; s/{/{\
        if(!std::getline(f, ligne)) break;\
        if(!ligne.empty() \&\& ligne.back() == 0x0d) ligne.pop_back();/}' src/principalExtension.cpp
sed -i 's/contenu+=ligne;/contenu+=ligne+" ";/' src/principalExtension.cpp

# === Extra fix: ss.str("") incompatible with newer C++ ===
sed -i 's/ss\.str("")/ss.str(std::string())/g' src/greenSizeHash.cpp

echo "All patches applied successfully."
