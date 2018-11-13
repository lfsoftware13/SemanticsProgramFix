import sys
import trace

# create a Trace object, telling it what to ignore, and whether to
# do tracing or line-counting or both.
tracer = trace.Trace(
    ignoredirs=[sys.prefix, sys.exec_prefix],
    trace=1,
    count=1,
    countfuncs=0, countcallers=0, ignoremods=(), timing=False)

code = r'''
from base_program import main
main()'''

# run the new command using the given tracer
tracer.run(code)

# make a report, placing output in the current directory
r = tracer.results()
r.write_results(show_missing=True, coverdir=".")