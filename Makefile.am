bin_PROGRAMS = nb nbdist

nb_SOURCES = \
        src/naive-bayes.cc \
        src/util.cc \
	src/dettools.cc

nbdist_SOURCES = \
        src/nbdist.cc \
        src/util.cc \
	src/dettools.cc

nb_CXXFLAGS = -O3 -Iinclude
nbdist_CXXFLAGS = -O3 -Iinclude

#-std=c++11
#-I@FBINC@ 

#nb_LDADD = \
#	-lboost_system \
#	-lboost_program_options \
#	-lboost_filesystem \
#    -ldb_cxx -lpthread 
