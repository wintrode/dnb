AC_INIT(dnb, 0.1, jonathan.wintrode@gmail.com)
AM_INIT_AUTOMAKE([foreign subdir-objects -Wall])

if test -z $CXXFLAGS; then
    CXXFLAGS='-O3 -g'
fi
AC_PROG_CXX

AC_CHECK_PROGS([DOXYGEN], [doxygen])
if test -z "$DOXYGEN"; then
   AC_MSG_WARN([Doxygen not found - continue without Doxygen support])
fi
AM_CONDITIONAL([HAVE_DOXYGEN], [test -n "$DOXYGEN"])

AM_COND_IF([HAVE_DOXYGEN], [AC_CONFIG_FILES([docs/Doxyfile])])

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile])

AC_OUTPUT
